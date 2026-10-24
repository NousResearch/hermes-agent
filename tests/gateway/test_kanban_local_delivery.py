"""Local Kanban delivery uses canonical viewer events and the owning authority FIFO."""
import asyncio

import pytest
import pytest_asyncio

from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_local_producer_admission import _Gateway


@pytest_asyncio.fixture
async def gateway(tmp_path, monkeypatch):
    # Native-package discovery is irrelevant here; never probe a live install manifest.
    monkeypatch.setattr('pm.paths.store_root', lambda: tmp_path / 'tools')
    monkeypatch.setattr('pm.paths.writable_store_root', lambda: tmp_path / 'tools')
    value = _Gateway(tmp_path, monkeypatch)
    await value.start()
    try:
        yield value
    finally:
        await value.stop()


async def tick(runner):
    deliveries = await asyncio.to_thread(_notifier_collect, runner, kb,
        notifier_profile='default', gc_due=False, gc_retention_days=30)
    for delivery in deliveries:
        await _KanbanNotification(runner, delivery, platform_cls=Platform, sub_fail_counts={}).deliver()


def cursor(tid):
    with kbc.connect() as conn:
        return dict(conn.execute('SELECT last_event_id, last_ping_event_id FROM kanban_notify_subs '
                                 'WHERE task_id=?', (tid,)).fetchone())


@pytest.mark.asyncio
@pytest.mark.parametrize('profile', ['default', 'beta'])
@pytest.mark.parametrize('mode', ['notify', 'notify+wake'])
async def test_local_terminal_event_is_visible_once_and_settles_its_cursor(gateway, tmp_path, monkeypatch, profile, mode):
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'board.db'))
    authority = gateway.primary if profile == 'default' else gateway.secondary
    ref = gateway.create(authority, gateway.connect(authority), 'kanban-origin', 'gui')
    sid = ref.session_id
    live = authority.sessions[sid]
    frames = []
    live.event_stream.observers.add(frames.append)
    await gateway.watch(authority, sid)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='local receipt', assignee='worker', session_id=live.route)
        kbn.add_notify_sub(conn, task_id=tid, platform='local', chat_id=sid, user_id='uid:1000',
                          chat_type='dm', notifier_profile=profile, delivery_mode=mode)
        kb.complete_task(conn, tid, summary='isolated handoff')
        completed_event_id = conn.execute(
            "SELECT id FROM task_events WHERE task_id=? AND kind='completed'", (tid,)
        ).fetchone()[0]
    before = cursor(tid)
    if mode == 'notify+wake':
        original = authority.admit_automation

        async def refuse(*args, **kwargs):
            raise RuntimeStoreError('runtime_draining')

        monkeypatch.setattr(authority, 'admit_automation', refuse)
        await tick(gateway.runner)
        assert cursor(tid) == before, 'a refused local wake must not settle or ACK a phantom ping'
        assert gateway.turns == []
        assert not [f for f in frames if f['params']['type'] == 'notification.show']
        monkeypatch.setattr(authority, 'admit_automation', original)
    await tick(gateway.runner)
    if mode == 'notify+wake':
        await gateway.settled(authority, sid, 1)
        [row] = gateway.rows(authority, sid)
        assert 'kanban-wake' in row['request_id']
        assert row['payload']['local_automation_v1']['owner'] == sid
        [turn] = gateway.turns
        assert (turn.chat, turn.profile) == (sid, live.source.profile)
        assert 'isolated handoff' in turn.text
        assert not [f for f in frames if f['params']['type'] == 'notification.show']
        [reply] = [f for f in frames if f['params']['type'] == 'message.complete']
        assert reply['params']['session_id'] == sid
        assert reply['params']['payload']['text'] == 'ack'
    else:
        assert gateway.turns == []
        [notice] = [f for f in frames if f['params']['type'] == 'notification.show']
        assert notice['params']['session_id'] == sid
        assert 'isolated handoff' in notice['params']['payload']['text']
    after = cursor(tid)
    assert after['last_event_id'] == completed_event_id
    # The wake cursor settles accepted delivery; the ping checkpoint records only
    # passive publication, deliberately absent for a local streamed wake response.
    assert after['last_ping_event_id'] == (
        completed_event_id if mode == 'notify' else before['last_ping_event_id']
    )
    # Admission settlement can precede its trailing session.info frame. Compare only
    # visible deliveries: a late metadata refresh is not a duplicate notification.
    visible_types = {'notification.show', 'message.complete'}
    delivered = [f for f in frames if f['params']['type'] in visible_types]
    await tick(gateway.runner)
    assert cursor(tid) == after
    assert [f for f in frames if f['params']['type'] in visible_types] == delivered
    assert len(gateway.turns) == (mode == 'notify+wake')
