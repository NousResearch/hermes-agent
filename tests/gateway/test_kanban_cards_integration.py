"""Installed-wheel -> native loader -> canonical watcher/DB -> real adapter.

Only external Telegram API calls and watcher clock are fixtures. No listeners.
"""
import asyncio
from contextlib import closing

from types import SimpleNamespace

import pytest
import pytest_asyncio
pytest.importorskip(
    "hermes_telegram_experience",
    reason="optional external-plugin integration; exercised by plugin repository CI",
)
import yaml
from telegram.error import BadRequest, RetryAfter, TimedOut
from gateway.config import Platform, PlatformConfig
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.kanban_watchers_notifier import _notifier_collect, _KanbanNotification
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as notify, kanban_db_surface as receipts
from hermes_cli.plugins import get_plugin_manager
from plugins.platforms.telegram.adapter import TelegramAdapter


class Bot:
    def __init__(self):
        self.sent, self.edited = [], []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.pause = False
        self.failure = None
        self.edit_failure = None

    async def send_message(self, **kw):
        self.sent.append(kw)
        self.entered.set()
        if self.pause:
            await self.release.wait()
        if self.failure:
            raise self.failure
        return SimpleNamespace(message_id=700 + len(self.sent))

    async def edit_message_text(self, **kw):
        self.edited.append(kw)
        if self.edit_failure:
            raise self.edit_failure
        return SimpleNamespace(message_id=kw['message_id'])


class Runner(GatewayKanbanWatchersMixin):
    def __init__(self, home, adapter):
        self.home = home
        self.adapters = {Platform.TELEGRAM: adapter}
        self._profile_adapters = {}
        self.config = SimpleNamespace(multiplex_profiles=False)
        self._kanban_notifier_profile = 'default'
        self._kanban_dispatcher_lock_handle = object()
        self._running = True

    def _active_profile_name(self): return 'default'
    def _resolve_profile_home_for_source(self, source): return self.home
    def _authorization_adapter(self, platform, profile): return self.adapters.get(platform)


def config(home, cards=True, quiet=False, enabled=True, *, task_id, profile='default',
           board='default', routes=(('-100', '7'), ('-100', '8'), ('-200', '7'))):
    scope = {
        'routes': [dict(profile=profile, platform='telegram', chat_id=chat, thread_id=thread)
                   for chat, thread in routes],
        'task_resources': [dict(board=board, task_id=task_id)],
    }
    payload = dict(plugins=dict(enabled=['hermes-telegram-experience'], entries={
        'hermes-telegram-experience': dict(settings=dict(
            enabled=enabled, durable_cards=cards, scope=scope))}))
    if quiet:
        payload['display'] = {'tool_progress': 'off'}
    (home/'config.yaml').write_text(yaml.safe_dump(payload))


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    home = tmp_path/'profile'; home.mkdir()
    path = tmp_path/'board.db'
    monkeypatch.setenv('HERMES_HOME',str(home)); monkeypatch.setenv('HERMES_KANBAN_DB',str(path))
    with closing(kbc.connect(path)) as c:
        tid = kb.create_task(c, title='Synthetic title', assignee='worker')
        notify.add_notify_sub(c, task_id=tid, platform='telegram',chat_id='-100',thread_id='7',notifier_profile='default',chat_type='group')
    config(home, task_id=tid)
    manager = get_plugin_manager(); manager.discover_and_load()
    assert manager._task_card_registration.active
    import hermes_telegram_experience
    assert 'site-packages' in hermes_telegram_experience.__file__
    bot = Bot()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='123:synthetic', typing_indicator=False))
    adapter._bot = bot
    runner = Runner(home, adapter)
    result=SimpleNamespace(home=home,path=path,manager=manager,bot=bot,adapter=adapter,runner=runner,tid=tid)
    yield result
    bot.release.set()
    manager.unload()
    for source in tuple(adapter._live_todo_sources): await source.finish()
    await asyncio.sleep(0)


async def tick(rig, drain=True):
    deliveries = await asyncio.to_thread(_notifier_collect, rig.runner, kb, notifier_profile='default',gc_due=False,gc_retention_days=30)
    for d in deliveries:
        await _KanbanNotification(rig.runner,d,platform_cls=Platform,sub_fail_counts={}).deliver()
    if drain:
        sources=tuple(rig.manager._task_card_registration.sources)
        if sources: await asyncio.wait_for(asyncio.gather(*(s.task for s in sources)),5)
    return deliveries


def receipt(rig):
    with closing(kbc.connect(rig.path)) as c:
        return receipts.list_delivery_receipts(c,task_id=rig.tid)[0]


def advance(rig):
    with closing(kbc.connect(rig.path)) as c:
        task = kb.get_task(c, rig.tid)
        assert task is not None
        assert kb.edit_task(c, rig.tid, title=task.title + ' revised')
        return receipts.get_task_source(c,rig.tid).current_revision


@pytest.mark.asyncio
@pytest.mark.parametrize('fence',['none','expired','unload','replacement'])
async def test_paused_create_new_desire_exact_late_id_same_message(rig,fence):
    rig.bot.pause=True
    await tick(rig,False)
    await asyncio.wait_for(rig.bot.entered.wait(),5)
    source=next(iter(rig.manager._task_card_registration.sources))
    old=receipt(rig); assert old.attempted_revision==old.desired_revision
    new=advance(rig)
    await tick(rig,False)
    assert receipt(rig).desired_revision==new
    if fence=='expired':
        with closing(kbc.connect(rig.path)) as c:
            c.execute('UPDATE kanban_delivery_receipts SET lease_expires_at=0')
            with pytest.raises(receipts.DeliveryReceiptUnknown):
                receipts.claim_delivery_receipt(c,old.id,owner_id='other')
    if fence=='unload': rig.manager.unload('hermes-telegram-experience')
    if fence=='replacement': rig.adapter._fence_live_todo_transport()
    rig.bot.release.set()
    await asyncio.wait_for(source.task,5)
    settled=receipt(rig)
    assert settled.destination_message_id=='701'
    assert settled.delivered_revision in (old.desired_revision,new)
    assert settled.desired_revision==new
    if fence=='unload': rig.manager.discover_and_load(force=True)
    await tick(rig)
    assert len(rig.bot.sent)==1
    assert rig.bot.edited[-1]['message_id']==701
    assert receipt(rig).delivered_revision==new
    assert 'message_thread_id' not in rig.bot.edited[-1]
    assert rig.bot.sent[0]['message_thread_id']==7


@pytest.mark.asyncio
async def test_duplicate_ticks_reordering_and_committed_states(rig):
    await tick(rig); await tick(rig)
    assert len(rig.bot.sent)==1 and not rig.bot.edited
    with closing(kbc.connect(rig.path)) as c: kb.block_task(c,rig.tid,reason='synthetic blocker',kind='needs_input')
    d=await tick(rig)
    assert rig.bot.edited[-1]['text'] == 'Synthetic title\nWaiting'
    with closing(kbc.connect(rig.path)) as c:
        kb.complete_task(c,rig.tid,summary='synthetic verified fixture result')
    await tick(rig)
    latest=receipt(rig).delivered_revision
    await _KanbanNotification(rig.runner,d[0],platform_cls=Platform,sub_fail_counts={}).deliver()
    await asyncio.sleep(0.05)
    assert receipt(rig).delivered_revision==latest
    assert 'Completed' in rig.bot.edited[-1]['text']
    assert len(rig.bot.sent)==1


@pytest.mark.asyncio
async def test_comment_payload_coalesces_then_block_edits_same_card(rig):
    await tick(rig)
    first = receipt(rig)
    assert first.state == 'sent' and first.destination_message_id == '701'

    rig.bot.edit_failure = BadRequest('Bad Request: message is not modified')
    with closing(kbc.connect(rig.path)) as c:
        kb.add_comment(c, rig.tid, 'synthetic', 'new canonical event')
        comment_revision = receipts.get_task_source(c, rig.tid).current_revision
    await tick(rig)
    after_comment = receipt(rig)
    assert after_comment.state == 'sent'
    assert after_comment.delivered_revision == comment_revision
    assert not rig.bot.edited

    rig.bot.edit_failure = None
    with closing(kbc.connect(rig.path)) as c:
        kb.block_task(c, rig.tid, reason='synthetic blocker', kind='needs_input')
        blocked_revision = receipts.get_task_source(c, rig.tid).current_revision
    await tick(rig)
    after_block = receipt(rig)
    assert after_block.delivered_revision == blocked_revision
    assert len(rig.bot.sent) == 1 and len(rig.bot.edited) == 1
    assert rig.bot.edited[0]['message_id'] == 701
    assert 'Waiting' in rig.bot.edited[0]['text']


@pytest.mark.asyncio
@pytest.mark.parametrize('error',[TimedOut('ambiguous'),BadRequest('invalid content'),RetryAfter(0)])
async def test_bounded_failures_never_blind_resend(rig,error):
    rig.bot.failure=error
    for _ in range(5):
        await tick(rig)
        # Advance the durable retry clock only (no production policy override).
        with closing(kbc.connect(rig.path)) as c: c.execute('UPDATE kanban_delivery_receipts SET retry_at=0')
        rig.adapter.__dict__.setdefault('_telegram_send_cooldown_until', {}).clear()
    r=receipt(rig)
    assert len(rig.bot.sent)==(3 if isinstance(error,RetryAfter) else 1)
    assert r.state==('unknown' if isinstance(error,TimedOut) else 'failed')
    with closing(kbc.connect(rig.path)) as c: assert kb.get_task(c,rig.tid).status=='ready'


@pytest.mark.asyncio
async def test_deleted_known_card_one_replacement_unknown_stays_unknown(rig):
    await tick(rig)
    rig.bot.edit_failure=BadRequest('Message to edit not found')
    advance(rig); await tick(rig)
    assert receipt(rig).state=='deleted'
    rig.bot.failure=TimedOut('ambiguous replacement')
    await tick(rig); await tick(rig)
    r=receipt(rig)
    assert len(rig.bot.sent)==2 and r.state=='unknown' and r.replacement_budget==0
    advance(rig); await tick(rig)
    assert receipt(rig).replacement_budget==0 and len(rig.bot.sent)==2


@pytest.mark.asyncio
async def test_quiet_disable_reenable_and_normal_handoff(rig):
    config(rig.home,quiet=True,task_id=rig.tid)
    await tick(rig)
    assert not rig.bot.sent
    config(rig.home,task_id=rig.tid); await tick(rig)
    assert len(rig.bot.sent)==1
    rig.manager.unload('hermes-telegram-experience')
    with closing(kbc.connect(rig.path)) as c: kb.block_task(c,rig.tid,reason='normal notifier',kind='needs_input')
    await tick(rig)
    assert len(rig.bot.sent)==2 and 'blocked' in rig.bot.sent[-1]['text']
    config(rig.home,task_id=rig.tid); rig.manager.discover_and_load(force=True)
    await tick(rig)
    assert len(rig.bot.sent)==2 and rig.bot.edited[-1]['message_id']==701


@pytest.mark.asyncio
@pytest.mark.parametrize('fence', ['none', 'quiet', 'unload', 'replacement'])
async def test_narrowed_active_replacement_declines_late_surface_handoff(rig, monkeypatch, fence):
    with closing(kbc.connect(rig.path)) as c:
        kb.block_task(c, rig.tid, reason='ordinary notifier handoff', kind='needs_input')
    deliveries = await asyncio.to_thread(
        _notifier_collect, rig.runner, kb, notifier_profile='default',
        gc_due=False, gc_retention_days=30,
    )
    assert len(deliveries) == 1 and deliveries[0]['surface'] is not None

    # Replace the active consumer after collection with an exact scope that no
    # longer owns this resource. The already-claimed terminal event must return
    # to the ordinary notifier, without constructing a plugin card source.
    config(rig.home, task_id='t_deadbeef', quiet=fence == 'quiet')
    rig.manager.discover_and_load(force=True)
    replacement = rig.manager._task_card_registration
    assert replacement.active and not replacement.sources

    # Deterministically let lifecycle changes win after lookup and before the
    # handoff's locked admission, as an executor-thread reload can do.
    from gateway import kanban_surfaces
    lookup = kanban_surfaces.subscription_registration
    def lookup_then_fence(*args, **kwargs):
        result = lookup(*args, **kwargs)
        if fence == 'unload':
            rig.manager.unload('hermes-telegram-experience')
        elif fence == 'replacement':
            config(rig.home, task_id=rig.tid)
            rig.manager.discover_and_load(force=True)
        return result
    monkeypatch.setattr(kanban_surfaces, 'subscription_registration', lookup_then_fence)

    await _KanbanNotification(
        rig.runner, deliveries[0], platform_cls=Platform, sub_fail_counts={},
    ).deliver()

    assert len(rig.bot.sent) == (1 if fence == 'none' else 0)
    if fence == 'none':
        assert 'blocked' in rig.bot.sent[0]['text'].lower()
    assert not replacement.sources


@pytest.mark.asyncio
async def test_actual_watcher_tick_not_a_private_loader_shortcut(rig,monkeypatch):
    real=asyncio.sleep
    async def clock(delay):
        if delay==5: return
        rig.runner._running=False
        await real(0)
    monkeypatch.setattr(asyncio,'sleep',clock)
    await rig.runner._kanban_notifier_watcher(interval=1)
    sources=tuple(rig.manager._task_card_registration.sources)
    if sources: await asyncio.gather(*(s.task for s in sources))
    assert len(rig.bot.sent)==1 and receipt(rig).state=='sent'


@pytest.mark.asyncio
async def test_native_default_off_missing_capability_and_idempotence(rig,monkeypatch,caplog):
    from hermes_cli.plugins import PluginContext
    registration=rig.manager._task_card_registration
    rig.manager.discover_and_load()
    assert rig.manager._task_card_registration is registration
    rig.manager.unload()
    # Native discovery with omitted opt-in settings must also stay off, not
    # merely the explicit durable_cards: false case below.
    for settings in ('', '      settings:\n        enabled: true\n'):
        (rig.home/'config.yaml').write_text(
            'plugins:\n  enabled: [hermes-telegram-experience]\n  entries:\n'
            '    hermes-telegram-experience:\n' + settings)
        rig.manager.discover_and_load(force=True)
        assert not rig.manager._task_card_registration.active
        await tick(rig)
        assert not rig.bot.sent
        rig.manager.unload()
    config(rig.home,cards=False,task_id=rig.tid)
    rig.manager.discover_and_load(force=True)
    assert not rig.manager._task_card_registration.active
    await tick(rig)
    assert not rig.bot.sent
    rig.manager.unload()
    config(rig.home,task_id=rig.tid)
    monkeypatch.setattr(PluginContext,'task_card_capability',None)
    rig.manager.discover_and_load(force=True)
    assert not rig.manager._task_card_registration.active
    assert 'task_card capability 2' in caplog.text
    assert not rig.manager._live_todo_registration.active


@pytest.mark.asyncio
async def test_known_edit_uncertainty_has_durable_retry_budget(rig):
    await tick(rig)
    rig.bot.edit_failure=TimedOut('known edit uncertainty')
    advance(rig)
    for _ in range(5): await tick(rig)
    assert len(rig.bot.sent)==1 and len(rig.bot.edited)==3
    assert receipt(rig).state=='unknown' and receipt(rig).failure_count==3
