"""A Kanban notify+wake subscription made from a local (CLI/TUI/Desktop) session wakes that session.

Local session recovery builds the LocalSessionAdapter itself; ``_create_adapter`` (which sets
``gateway_runner`` on every other adapter) never sees it. ``deliver_wake`` reads
``adapter.gateway_runner`` to commit the wake into the authority FIFO (``admit_route_wake``'s LOCAL
branch); with it unset the wake fell to the adapter path, which a local adapter never accepts, and
the notifier rewound the claim every tick: the passive ping was recorded, the session never woke.
Real GatewayRunner, SessionAuthority, local session, Kanban board and notifier tick over SQLite;
only the turn itself is a stand-in.
"""
import asyncio
from contextlib import closing
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


def _hermes_root(tmp_path, monkeypatch):
    """A temp ``~/.hermes`` whose profile name both the subscriber and the notifier resolve."""
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'board.db'))
    return root


def _subscribe_from_session(live, title):
    """``kanban_create`` called by the local session's own turn: its session vars, its auto-subscribe."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.kanban_tools import _maybe_auto_subscribe
    tokens = set_session_vars(platform=live.source.platform.value, chat_id=live.source.chat_id,
                              chat_type=live.source.chat_type, user_id=live.source.user_id or '',
                              profile=live.source.profile or '', session_key=live.route)
    try:
        with closing(kbc.connect()) as conn:
            task_id = kb.create_task(conn, title=title, assignee='worker')
            assert _maybe_auto_subscribe(conn, task_id)
            [sub] = kbn.list_notify_subs(conn, task_id)
            assert (sub['platform'], sub['chat_id'], sub['delivery_mode']) == ('local', live.source.chat_id,
                                                                              'notify+wake'), sub
            kb.complete_task(conn, task_id, summary='handoff for the creator')
    finally:
        clear_session_vars(tokens)
    return task_id


async def _tick(runner, failures):
    deliveries = await asyncio.to_thread(_notifier_collect, runner, kb,
                                         notifier_profile=runner._kanban_notifier_profile,
                                         gc_due=False, gc_retention_days=30)
    for delivery in deliveries:
        await _KanbanNotification(runner, delivery, platform_cls=Platform, sub_fail_counts=failures).deliver()
    return deliveries


def _unseen(task_id, chat_id):
    with closing(kbc.connect()) as conn:
        return kbn.unseen_events_for_sub(conn, task_id=task_id, platform='local', chat_id=chat_id,
                                         kinds=['completed'])[1]


def _wakes(authority, session_id):
    from hermes_state_runtime import list_session_admissions
    return [row for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False)
            if row['principal_id'].startswith('automation:')]


async def _assert_wakes(runner, authority, ref, task_id):
    """One notifier tick commits one wake to the session's FIFO, advances the cursor, and the drain runs it."""
    live = authority.sessions[ref.session_id]
    failures = {}
    assert await _tick(runner, failures), 'the notifier must claim the completed event'
    assert not _unseen(task_id, live.source.chat_id), 'the wake was not accepted: the claim was rewound'
    [wake] = _wakes(authority, ref.session_id)
    assert wake['status'] == 'queued' and 'local_automation_v1' in wake['payload'], wake
    assert task_id in wake['payload']['text'], wake
    assert failures == {}
    # A second tick finds nothing left to deliver: the cursor advanced once, no duplicate wake.
    assert not await _tick(runner, failures)
    assert len(_wakes(authority, ref.session_id)) == 1
    seen = []

    async def turn(event):
        seen.append(event.text)
        return 'ok'
    runner._handle_message = turn
    await authority._drain(ref)
    assert len(seen) == 1 and task_id in seen[0], seen


@pytest.mark.asyncio
async def test_kanban_wake_reaches_a_local_session(tmp_path, monkeypatch):
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import LocalSessionAdapter, create_local_session
    from hermes_state import AsyncSessionDB, SessionDB

    root = _hermes_root(tmp_path, monkeypatch)
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # drained by hand below
    runner = GatewayRunner(GatewayConfig(sessions_dir=root / 'sessions'))
    db = SessionDB(root / 'state.db')
    runner.session_store._db = db
    runner._session_db = AsyncSessionDB(db)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner', db=db)
    try:
        actor = Principal('uid:501', 'default', frozenset({'session:create', 'session:submit', 'session:read'}),
                          'fixture')
        ref = create_local_session(authority, actor, dict(request_id='desk', source='gui', model='fixture',
                                                          toolsets=[]))
        assert isinstance(runner.adapters.get(Platform.LOCAL), LocalSessionAdapter)
        task_id = _subscribe_from_session(authority.sessions[ref.session_id], 'local wake')
        await _assert_wakes(runner, authority, ref, task_id)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_kanban_wake_reaches_a_served_profiles_local_session(tmp_path, monkeypatch):
    """The same card from a served secondary's Desktop session (``multiplex_profiles``): its adapter
    lives in ``runner._profile_adapters[name]`` and its wake commits to that profile's authority."""
    import hermes_state
    from gateway import run
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authority import SessionAuthority
    from gateway.session_authorities import owner_scope
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import LocalSessionAdapter, create_local_session

    root = _hermes_root(tmp_path, monkeypatch)
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {'platform_toolsets': {'cli': [], 'tui': []}})
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # drained by hand below
    home = root / 'profiles' / 'beta'
    home.mkdir(parents=True, mode=0o700)
    (home / 'config.yaml').write_text('{}')
    homes = (('default', root), ('beta', home))
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = homes
    runner = GatewayRunner(config)
    process_ownership.reserve([h for _, h in homes])
    connection = None
    try:
        await initialize_gateway_runtime(runner)
        secondary = runner.session_authorities.for_home(home)
        connection = AuthorityConnection(secondary, object(), {'user_id': 'human'})
        with owner_scope(secondary):
            ref = create_local_session(secondary, connection.actor, dict(
                request_id='desk', source='gui', cwd=str(tmp_path), model='fixture', toolsets=[]))
        live = secondary.sessions[ref.session_id]
        assert live.source.profile == 'beta'
        assert isinstance(runner._profile_adapters['beta'].get(Platform.LOCAL), LocalSessionAdapter)
        task_id = _subscribe_from_session(live, 'served local wake')
        await _assert_wakes(runner, secondary, ref, task_id)
    finally:
        if connection is not None:
            await connection.close()
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, h in homes:
            process_ownership.release(h)
