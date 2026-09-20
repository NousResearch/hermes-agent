"""Owner-fenced Output eligibility: canonical admission and real SQLite reads.

Two private owners, inert socket/model boundary, manual prepare only. No runtime
coordinator, listener, recovery, or provider execution is started by these tests.
"""
import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from gateway import hosted_room_driver as tasks
from gateway.hosted_room_artifacts import RoomArtifactError
from gateway.run import _profile_runtime_scope
from hermes_state import StateDbCorruptError
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_retry import clock, pending, tick
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


async def hold_output(c, monkeypatch):
    from gateway.session_peer_output_custody import PeerOutputCustody
    clock(c)
    def unavailable(self, *args):
        raise PeerRunsHTTPError('held import', retryable=True, status_code=503)
    monkeypatch.setattr(PeerOutputCustody, 'read', unavailable)
    await tick(c)
    assert len(pending(c)) == 1


def retained_rows(conn):
    return {
        table: [tuple(row) for row in conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
        for table in ('hosted_room_driver_tasks', 'hosted_room_artifact_retries',
                      'hosted_room_artifact_completions', 'session_admissions')
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('loss', ['drain', 'owner'])
async def test_mid_callback_owner_refusal_precedes_queued_admission(files_target, monkeypatch, loss):
    async with peer_case(files_target, monkeypatch) as c:
        await hold_output(c, monkeypatch)
        before = retained_rows(c.db._conn)
        wire = list(c.wire.calls)
        checkpoint = c.service.policy_checkpoint
        snapshot, metadata = checkpoint.snapshot, c.service._output_metadata
        selections = []
        injected = []
        def select(**kwargs):
            selections.append(1)
            return snapshot(**kwargs)
        def lose_owner(conn, key):
            # The second canonical prepare snapshot controls NEW admission. Its
            # initial callback owner check has succeeded before reaching here.
            if len(selections) == 2 and not injected:
                injected.append(1)
                if loss == 'drain':
                    c.authority.runner._draining = True
                else:
                    c.authority.runner.session_authority = None
            return metadata(conn, key)
        monkeypatch.setattr(checkpoint, 'snapshot', select)
        monkeypatch.setattr(c.service, '_output_metadata', lose_owner)
        error = None
        try:
            with _profile_runtime_scope(c.home, hydrate_secrets=False):
                await asyncio.to_thread(c.service.send, room_id='room-one', event_id='new-same-thread',
                    payload=dict(thread_id='thread-one', text='@reader Reply to the newer request.'))
        except (RoomArtifactError, RuntimeStoreError) as exc:
            error = exc
        assert injected == [1], 'must reach metadata after the callback owner check'
        assert retained_rows(c.db._conn) == before, 'owner refusal must precede NEW queued admission'
        assert error is not None, 'observed owner loss must propagate, not become no hold'
        if loss == 'drain':
            assert isinstance(error, RuntimeStoreError) and error.reason == 'runtime_draining'
        assert c.wire.calls == wire
        assert len(c.launched) == len(c.executions) == 1
        assert not tasks.list_tasks(c.db.db_path, room_id='room-one', status='queued')
        assert not c.db._conn.in_transaction


@pytest.mark.asyncio
@pytest.mark.parametrize('loss', ['closed', 'closed-after-sync', 'replaced', 'quarantined', 'stale-epoch', 'swapped-db'])
async def test_eligibility_refuses_unavailable_owner_without_path_reopen(files_target, monkeypatch, loss):
    async with peer_case(files_target, monkeypatch) as c:
        await hold_output(c, monkeypatch)
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            room = c.service._room('room-one')
        old_conn = c.db._conn
        before = retained_rows(old_conn)
        wire = list(c.wire.calls)
        path = c.db.db_path
        displaced = path.with_name('displaced-state.db')
        replacement = path.with_name('replacement-state.db')
        if loss == 'closed':
            c.db.close()
            assert c.db._conn is None and c.db._read_conns_closed
        elif loss == 'closed-after-sync':
            sync = c.service.policy_checkpoint.sync
            def close_after_sync(**kwargs):
                result = sync(**kwargs)
                c.db.close()
                return result
            monkeypatch.setattr(c.service.policy_checkpoint, 'sync', close_after_sync)
        elif loss == 'swapped-db':
            c.authority.db = c.target.authority.db
        elif loss == 'replaced':
            with sqlite3.connect(replacement) as new:
                new.execute('CREATE TABLE replacement_marker(value TEXT)')
            new.close()
            path.rename(displaced)
            replacement.rename(path)
            assert c.db._db_file_was_replaced()
        elif loss == 'quarantined':
            with pytest.raises(StateDbCorruptError):
                c.db._halt_db_corrupt(sqlite3.DatabaseError('inert structural corruption'))
        else:
            c.db._execute_write(lambda conn: conn.execute('UPDATE runtime_epoch SET epoch=epoch+1'))
        opened = []
        connect = sqlite3.connect
        def record_open(*args, **kwargs):
            if loss != 'closed-after-sync' or c.db._read_conns_closed:
                opened.append(args[0])
            return connect(*args, **kwargs)
        error = None
        try:
            with monkeypatch.context() as observed:
                observed.setattr(sqlite3, 'connect', record_open)
                with _profile_runtime_scope(c.home, hydrate_secrets=False):
                    try:
                        c.service._policy_snapshot(room)
                    except (RoomArtifactError, RuntimeStoreError, sqlite3.Error) as exc:
                        error = exc
            assert not opened, 'unavailable owner must refuse before checkpoint pathname opens'
            assert error is not None, 'closed/quarantined/replaced/stale owner is not eligibility authority'
            if loss in {'closed', 'closed-after-sync'}:
                assert c.db._conn is None and c.db._read_conns_closed
            else:
                assert c.db._conn is old_conn and not old_conn.in_transaction
                assert retained_rows(old_conn) == before
            assert c.wire.calls == wire
            assert len(c.launched) == len(c.executions) == 1
        finally:
            if loss == 'replaced':
                path.rename(replacement)
                displaced.rename(path)


@pytest.mark.asyncio
@pytest.mark.parametrize('independent', [False, True])
async def test_metadata_and_selection_share_one_committed_read_view(files_target, monkeypatch, independent):
    import gateway.session_hosted_output_retry as output
    async with peer_case(files_target, monkeypatch) as c:
        await hold_output(c, monkeypatch)
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            await asyncio.to_thread(c.service.send, room_id='room-one', event_id='snapshot-follower',
                payload=dict(thread_id='snapshot-independent' if independent else 'thread-one',
                             text='@reader Reply after the held output.'))
            room = c.service._room('room-one')
        before = retained_rows(c.db._conn)
        original_result = c.db._conn.execute('SELECT result_json FROM hosted_room_driver_tasks WHERE task_id=?',
                                            (c.task['identity'].task_id,)).fetchone()[0]
        changed_result = json.dumps(dict(json.loads(original_result), peer_result_digest='0' * 64))
        first_read, committed = Event(), Event()
        observed = []
        require = output.require_output_task
        def between_task_reads(conn, scope, cancel_generation, **kwargs):
            # _output_metadata fetched result/payload, but its repeated task proof
            # and policy selection have not run. Commit from a distinct WAL writer.
            first_read.set()
            assert committed.wait(5), 'bounded writer did not commit'
            observed.append(conn.execute('SELECT result_json FROM hosted_room_driver_tasks WHERE task_id=?',
                                         (c.task['identity'].task_id,)).fetchone()[0])
            return require(conn, scope, cancel_generation, **kwargs)
        def writer():
            assert first_read.wait(5), 'metadata edge not reached'
            with sqlite3.connect(c.db.db_path, timeout=5) as conn:
                conn.execute('UPDATE hosted_room_driver_tasks SET result_json=? WHERE task_id=?',
                             (changed_result, c.task['identity'].task_id))
                conn.execute('UPDATE hosted_room_policy_cursors SET stopped_through_seq=through_seq WHERE room_id=?', ('room-one',))
                conn.execute("UPDATE hosted_room_policy_threads SET completed=1 WHERE thread_id='snapshot-independent'")
                conn.execute("INSERT OR REPLACE INTO hosted_room_policy_watermarks VALUES ('room-one','snapshot-independent','reader',999)")
                conn.commit()
            conn.close()
            committed.set()
        def select():
            with _profile_runtime_scope(c.home, hydrate_secrets=False):
                return c.service._policy_snapshot(room)
        monkeypatch.setattr(output, 'require_output_task', between_task_reads)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(writer)
            selected = await asyncio.to_thread(select)
            future.result(timeout=5)
        assert observed == [original_result], 'metadata proof mixed pre/post-commit task rows'
        assert selected.stopped_through_seq == 0, 'cursor and metadata must share the same read view'
        if independent:
            assert [e['event_id'] for e in selected.events] == ['snapshot-follower'], 'selection mixed policy generations'
            assert selected.watermarks == {}, 'watermarks escaped the selection read transaction'
        else:
            assert selected.events == (), 'the unchanged settled hold must defer its causal thread'
        assert not c.db._conn.in_transaction, 'snapshot must release the borrowed transaction'
        assert c.db._conn.execute('SELECT result_json FROM hosted_room_driver_tasks WHERE task_id=?',
                                 (c.task['identity'].task_id,)).fetchone()[0] == changed_result
        after = retained_rows(c.db._conn)
        assert after['hosted_room_artifact_retries'] == before['hosted_room_artifact_retries']
        assert after['session_admissions'] == before['session_admissions']
        assert len(after['hosted_room_driver_tasks']) == len(before['hosted_room_driver_tasks'])
        assert len(c.launched) == len(c.executions) == 1
