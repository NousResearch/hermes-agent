import contextlib
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from test_supervision import supervised, review
from test_progress import setup, report
from tools import delegate_tool_registry as registry


def lease_fields(child):
    lease = child._delegate_reviewed_deadline
    return tuple(getattr(lease, key) for key in ('deadline', 'pending', 'latest', 'closed', 'renewals'))


@pytest.mark.parametrize('operation', ['report', 'review'])
@pytest.mark.parametrize('failure', ['insert', 'commit'])
def test_sql_failure_rolls_back_lease_and_allows_retry(supervised, monkeypatch, operation, failure):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)['checkpoint_id'] if operation == 'review' else None
    clock.now = 650
    before = lease_fields(child)
    original_db = plugin.db
    table = 'reviews' if operation == 'review' else 'reports'
    if failure == 'insert':
        with plugin.db() as db:
            db.execute(f"CREATE TRIGGER fail_write BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT, 'fault'); END")
    else:
        @contextlib.contextmanager
        def failing_commit():
            with original_db() as db:
                yield db
                if db.in_transaction:
                    raise sqlite3.OperationalError('commit fault')
        monkeypatch.setattr(plugin, 'db', failing_commit)
    action = lambda: review(plugin, sup, parent, checkpoint) if checkpoint else report(plugin)
    with pytest.raises(sqlite3.DatabaseError):
        action()
    assert lease_fields(child) == before
    monkeypatch.setattr(plugin, 'db', original_db)
    with plugin.db() as db:
        assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0
        db.execute('DROP TRIGGER IF EXISTS fail_write')
    assert action()['success']


def test_report_holds_lease_lock_until_commit(supervised, monkeypatch):
    plugin, ctx, parent, child, sup, clock = supervised
    original_db = plugin.db
    acquired = []
    @contextlib.contextmanager
    def checking_commit():
        with original_db() as db:
            statements = []
            db.set_trace_callback(statements.append)
            yield db
            if not any('INSERT INTO reports' in s for s in statements):
                return
            def compete():
                lease = child._delegate_reviewed_deadline
                ok = lease.lock.acquire(blocking=False)
                acquired.append(ok)
                if ok:
                    lease.lock.release()
            thread = threading.Thread(target=compete)
            thread.start()
            thread.join(1)
    monkeypatch.setattr(plugin, 'db', checking_commit)
    assert report(plugin)['success']
    assert acquired and not any(acquired)


def test_report_survives_dead_parent_weakref(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    child._delegate_parent_ref = lambda: None
    assert report(plugin)['success']


def test_report_requires_exact_registry_child(supervised, monkeypatch):
    plugin, ctx, parent, child, sup, clock = supervised
    monkeypatch.setitem(registry._active_subagents, child._subagent_id,
                        {'agent': SimpleNamespace(), 'owner_agent_session_id': parent.session_id})
    assert not report(plugin)['success']


def test_expired_duplicate_rejected(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    assert report(plugin)['success']
    clock.now = 701
    assert not report(plugin)['success']


def test_stop_uses_stable_id_not_compressed_session(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    report(plugin)
    plugin.stop(child_session_id='child-a', child_subagent_id='unrelated', child_status='completed')
    with plugin.db() as db:
        assert db.execute('SELECT status FROM children').fetchone()[0] == 'running'
    plugin.stop(child_session_id='compressed-child', child_subagent_id=child._subagent_id, child_status='completed')
    with plugin.db() as db:
        assert db.execute('SELECT status FROM children').fetchone()[0] == 'completed'
        assert json.loads(db.execute('SELECT payload FROM reports ORDER BY id DESC').fetchone()[0])['kind'] == 'terminal_checkpoint'


def test_supervisor_isolates_entire_child_path(supervised, monkeypatch):
    plugin, ctx, parent, child, sup, clock = supervised
    child.get_activity_summary = lambda: (_ for _ in ()).throw(RuntimeError('broken child'))
    healthy = SimpleNamespace(**child.__dict__)
    healthy._subagent_id = 'healthy'
    healthy.get_activity_summary = lambda: {}
    plugin.start(parent_session_id=parent.session_id, child_session_id='healthy-session', child_subagent_id='healthy')
    monkeypatch.setitem(registry._active_subagents, 'healthy', {'agent': healthy, 'owner_agent_session_id': parent.session_id})
    clock.now = 450
    sup.tick()
    with plugin.db() as db:
        assert db.execute('SELECT subagent FROM supervision_checks').fetchone()[0] == 'healthy'
    assert len(ctx.wakes) == 1


def test_unload_stops_timer_and_inflight_observation(supervised, monkeypatch):
    plugin, ctx, parent, child, sup, clock = supervised
    handles = []
    class Handle:
        cancelled = False
        def cancel(self):
            self.cancelled = True
    def schedule(*args):
        handles.append(Handle())
        return handles[-1]
    monkeypatch.setattr('agent.periodic_scheduler.schedule', schedule)
    sup.start()
    sup.start()
    assert len(handles) == 1
    child.get_activity_summary = lambda: (sup.close() or {})
    clock.now = 450
    sup.tick()
    assert handles[0].cancelled
    assert not ctx.wakes and not parent.notices
    sup.tick()
    assert not ctx.wakes
