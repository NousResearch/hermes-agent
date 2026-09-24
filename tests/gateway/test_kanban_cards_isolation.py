"""Installed cards through the real multiplex resolver, boards and watcher.

Fixtures supply disposable filesystem roots, colliding task IDs, a one-tick clock
and the external Telegram API boundary. No route/service replacement is used.
"""
import asyncio
from contextlib import closing
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
pytest.importorskip(
    "hermes_telegram_experience",
    reason="optional external-plugin integration; exercised by plugin repository CI",
)

from agent.secret_scope import set_multiplex_active
from gateway.run import GatewayRunner
from hermes_constants import set_hermes_home_override, reset_hermes_home_override

spec = importlib.util.spec_from_file_location(
    'card_isolation_harness', Path(__file__).with_name('test_kanban_cards_integration.py'))
assert spec is not None and spec.loader is not None
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


@pytest_asyncio.fixture
async def multiplex(tmp_path, monkeypatch):
    root = tmp_path/'.hermes'
    homes = {'default': root, 'alpha': root/'profiles/alpha', 'denied': root/'profiles/denied'}
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: root)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(root))
    managers = {}
    for name, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        h.config(home, quiet=name == 'alpha', profile=name,
                 board='south' if name == 'alpha' else 'north', task_id='t_aaaaaaaa')
        token = set_hermes_home_override(home)
        try:
            managers[name] = h.get_plugin_manager()
            managers[name].discover_and_load()
            assert managers[name]._task_card_registration.active
        finally:
            reset_hermes_home_override(token)
    import hermes_telegram_experience
    assert 'site-packages' in hermes_telegram_experience.__file__
    assert len({id(m) for m in managers.values()}) == len(homes)
    bots, adapters = {}, {}
    for name in ('default', 'alpha'):
        bots[name] = h.Bot()
        adapters[name] = h.TelegramAdapter(h.PlatformConfig(
            enabled=True, token='123:synthetic', typing_indicator=False))
        adapters[name]._bot = bots[name]
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=[])
    runner.adapters = {h.Platform.TELEGRAM: adapters['default']}
    runner._profile_adapters = {'alpha': {h.Platform.TELEGRAM: adapters['alpha']}, 'denied': {}}
    runner._primary_profile_name = runner._kanban_notifier_profile = 'default'
    runner._profile_failed_platforms = {}
    runner._kanban_dispatcher_lock_handle = object()
    # ID collision is intentional synthetic input, not a patched authority service.
    monkeypatch.setattr(h.kb, '_new_task_id', lambda: 't_aaaaaaaa')
    for board, owner in [('north', 'default'), ('south', 'alpha')]:
        h.kb.create_board(board)
        with closing(h.kbc.connect(board=board)) as c:
            tid = h.kb.create_task(c, title='Same task-like name', board=board)
            subscribe(c, tid, owner)
    set_multiplex_active(True)
    result = SimpleNamespace(root=root, homes=homes, managers=managers, bots=bots,
                             adapters=adapters, runner=runner, tid='t_aaaaaaaa')
    try:
        yield result
    finally:
        for bot in bots.values(): bot.release.set()
        for manager in managers.values(): manager.unload()
        for adapter in adapters.values():
            for source in tuple(adapter._live_todo_sources): await source.finish()
        set_multiplex_active(False)


def subscribe(conn, tid, owner, chat='-100'):
    h.notify.add_notify_sub(conn, task_id=tid, platform='telegram', chat_id=chat,
                            thread_id='7', notifier_profile=owner, chat_type='group')


def rows(board):
    with closing(h.kbc.connect(board=board)) as c:
        return h.receipts.list_delivery_receipts(c, task_id='t_aaaaaaaa')


async def watcher(rig, monkeypatch, drain=True):
    real_sleep = asyncio.sleep
    async def clock(delay):
        if delay == 5: return
        rig.runner._running = False
        await real_sleep(0)
    rig.runner._running = True
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(asyncio, 'sleep', clock)
        await rig.runner._kanban_notifier_watcher(interval=1)
    if drain:
        tasks = [s.task for m in rig.managers.values()
                 for s in tuple(m._task_card_registration.sources)]
        if tasks: await asyncio.wait_for(asyncio.gather(*tasks), 5)


@pytest.mark.asyncio
async def test_two_profiles_two_boards_effective_config_and_explicit_authority(multiplex, monkeypatch):
    r = multiplex
    with closing(h.kbc.connect(board='north')) as c:
        # A second explicit destination must work; an unserved profile must not
        # borrow the primary bot even for the same task/topic names.
        subscribe(c, r.tid, 'default', '-200')
        subscribe(c, r.tid, 'denied', '-300')
        with monkeypatch.context() as ids:
            ids.setattr(h.kb, '_new_task_id', lambda: 't_unsubscribed')
            h.kb.create_task(c, title='Same task-like name', board='north')
    await watcher(r, monkeypatch)
    assert {(m['chat_id'], m['message_thread_id']) for m in r.bots['default'].sent} == {(-100, 7), (-200, 7)}
    assert not r.bots['alpha'].sent  # alpha's effective quiet config, not launch config
    assert len(rows('north')) == 2
    assert len(rows('south')) == 1 and rows('south')[0].attempt_count == 0
    with closing(h.kbc.connect(board='north')) as c:
        assert not h.receipts.list_delivery_receipts(c, task_id='t_unsubscribed')
    h.config(r.homes['alpha'], profile='alpha', board='south', task_id=r.tid)
    await watcher(r, monkeypatch)
    assert len(r.bots['alpha'].sent) == 1
    expected_card = 'Same task-like name\nQueued'
    assert r.bots['alpha'].sent[0]['text'] == expected_card
    assert all(m['text'] == expected_card for m in r.bots['default'].sent)
    with closing(h.kbc.connect(board='south')) as c:
        assert h.kb.edit_task(c, r.tid, title='Same task-like name south')
    await watcher(r, monkeypatch)
    assert len(r.bots['alpha'].edited) == 1 and not r.bots['default'].edited
    assert r.bots['alpha'].edited[0]['message_id'] == 701
    # A -> B -> A effective config does not grant a second profile the first's state.
    h.config(r.homes['alpha'], quiet=True, profile='alpha', board='south', task_id=r.tid)
    with closing(h.kbc.connect(board='north')) as c:
        assert h.kb.edit_task(c, r.tid, title='Same task-like name north')
    await watcher(r, monkeypatch)
    assert len(r.bots['default'].edited) == 2 and len(r.bots['alpha'].edited) == 1
    assert all(m['chat_id'] != -300 for b in r.bots.values() for m in b.sent + b.edited)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['profile-a-b-a', 'incarnation'])
@pytest.mark.parametrize('dispatched', [False, True])
async def test_stale_writer_fenced_across_profile_and_incarnation(multiplex, monkeypatch, change, dispatched):
    r = multiplex
    bot, adapter = r.bots['default'], r.adapters['default']
    lock = adapter._chat_send_lock('-100')
    if dispatched:
        bot.pause = True
    else:
        await lock.__aenter__()
    await watcher(r, monkeypatch, drain=False)
    source = next(iter(r.managers['default']._task_card_registration.sources))
    if dispatched:
        await asyncio.wait_for(bot.entered.wait(), 5)
    else:
        # Wait for the durable claim, without replacing admission or route ownership.
        async def claimed():
            while source.lease is None: await asyncio.sleep(0)
        await asyncio.wait_for(claimed(), 5)
    old = rows('north')[0]
    with closing(h.kbc.connect(board='north')) as c:
        if change == 'profile-a-b-a':
            h.notify.remove_notify_sub(c, task_id=r.tid, platform='telegram', chat_id='-100', thread_id='7')
            subscribe(c, r.tid, 'alpha')
            assert not source.admitted()
            h.notify.remove_notify_sub(c, task_id=r.tid, platform='telegram', chat_id='-100', thread_id='7')
            subscribe(c, r.tid, 'default')
            assert h.kb.edit_task(c, r.tid, title='Same task-like name rebound')
        else:
            assert h.kb.delete_task(c, r.tid)
            assert h.kb.create_task(c, title='Same task-like name', board='north') == r.tid
            subscribe(c, r.tid, 'default')
            assert h.receipts.get_task_source(c, r.tid).task_incarnation != old.task_incarnation
        token = c.execute('SELECT binding_token FROM kanban_notify_subs WHERE task_id=?', (r.tid,)).fetchone()[0]
        assert token != source.data['binding_token']
    assert not source.admitted()
    if dispatched: bot.release.set()
    else: await lock.__aexit__(None, None, None)
    await asyncio.wait_for(source.task, 5)
    assert not source.admitted()
    assert len(bot.sent) == int(dispatched) and not bot.edited
    if dispatched:
        # Exact late evidence belongs to the old attempt, not new writer authority.
        settled = next(row for row in rows('north') if row.id == old.id)
        assert settled.destination_message_id == '701'
        assert settled.delivered_revision == old.attempted_revision
    with closing(h.kbc.connect(board='north')) as c:
        c.execute('UPDATE kanban_delivery_receipts SET retry_at=0')
    await watcher(r, monkeypatch)
    await watcher(r, monkeypatch)
    expected_creates = 2 if dispatched and change == 'incarnation' else 1
    assert len(bot.sent) == expected_creates
    if dispatched and change == 'profile-a-b-a':
        assert len(bot.edited) == 1 and bot.edited[0]['message_id'] == 701
    else:
        assert not bot.edited
    assert not r.bots['alpha'].sent and not r.bots['alpha'].edited
    assert rows('south')[0].attempt_count == 0
    assert not source.admitted()
