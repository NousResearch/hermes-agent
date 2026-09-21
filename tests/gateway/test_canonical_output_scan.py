"""Bounded inventory on real driver rows; no execution/settlement witness implied."""
import json
import pytest
from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn
from gateway import hosted_room_driver as tasks


@pytest.mark.asyncio
async def test_room_scan_resumes_interruption_and_revisits_late_terminal(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
        async def handle(event):
            return ''
        runner._handle_message = handle
        rpc, request, receipt, task, binding = await execute_group_turn(authority, service)
        # Inventory-only queued rows, not fabricated accepted or completed execution.
        def populate(conn):
            row = dict(conn.execute('SELECT * FROM hosted_room_driver_tasks').fetchone())
            for i in range(70):
                changed = dict(row, task_id=f'inventory-{i:03}', thread_id=f't-{i}', turn_id=f'turn-{i}',
                    status='queued', execution_generation=0, result_json=None, settlement_id=None,
                    settlement_status=None, terminal_at=None)
                conn.execute('INSERT INTO hosted_room_driver_tasks (' + ','.join(changed) + ') VALUES (' +
                             ','.join('?' for _ in changed) + ')', tuple(changed.values()))
        authority.db._execute_write(populate)
        def exhaustive(*args, **kwargs):
            raise AssertionError('new Output consumer must not enumerate the room exhaustively')
        monkeypatch.setattr(service, '_list_tasks', exhaustive)
        from gateway import hosted_room_task_scan as scan_module
        original_page, sizes = scan_module.page, []
        def bounded_page(*args):
            saved, batch = original_page(*args)
            sizes.append(len(batch))
            return saved, batch
        monkeypatch.setattr(scan_module, 'page', bounded_page)
        room = service._room('room')
        service._prepare_terminal_tasks(room)
        from gateway.hosted_room_task_scan import scan_state
        with authority.db._read_ctx() as conn:
            first = scan_state(conn, 'room')
        assert first['pending'] and '' < first['after'] < first['highwater']
        from hermes_state_mutation_retirement import retire_prunable
        assert authority.db._execute_write(lambda c: retire_prunable(c, [rpc.ref.session_id])) == []
        assert tasks.prune_published_terminal_tasks(service.db_path, room_id='room', clock=lambda: 10**12, retain=0) == 0
        from gateway.session_group_retirement import require_room_retired
        from hermes_state_runtime import RuntimeStoreError
        with authority.db._read_ctx() as conn, pytest.raises(RuntimeStoreError):
            require_room_retired(conn, 'room')
        # A row already passed becomes terminal; a new row arrives beyond highwater.
        authority.db._execute_write(lambda c: c.execute(
            "UPDATE hosted_room_driver_tasks SET status='cancelled' WHERE task_id='inventory-000'"))
        def late(conn):
            row = dict(conn.execute("SELECT * FROM hosted_room_driver_tasks WHERE task_id='inventory-000'").fetchone())
            row.update(task_id='zz-late', thread_id='late-thread', turn_id='late-turn', status='cancelled')
            conn.execute('INSERT INTO hosted_room_driver_tasks (' + ','.join(row) + ') VALUES (' +
                         ','.join('?' for _ in row) + ')', tuple(row.values()))
        authority.db._execute_write(late)
        visited = []
        monkeypatch.setattr(service, '_publish_one_output', lambda room, task, progress: visited.append(task['identity'].task_id) or False)
        # A failure at the cursor commit cannot skip the batch on the next tick.
        authority.db._execute_write(lambda c: c.execute("""CREATE TRIGGER interrupted_scan
            BEFORE UPDATE ON state_meta WHEN NEW.key LIKE 'gateway.hosted.output_scan.v1:%'
            BEGIN SELECT RAISE(ABORT,'interrupted scan'); END"""))
        with pytest.raises(Exception, match='interrupted scan'):
            service._prepare_terminal_tasks(room)
        with authority.db._read_ctx() as conn:
            assert scan_state(conn, 'room') == first
        authority.db._execute_write(lambda c: c.execute('DROP TRIGGER interrupted_scan'))
        for _ in range(8):  # explicit bounded manual ticks, not a background retry loop
            service._prepare_terminal_tasks(room)
        assert 'inventory-000' in visited
        assert 'zz-late' in visited
        with authority.db._read_ctx() as conn:
            final = scan_state(conn, 'room')
        assert final['revision'] > first['revision']
        assert first['highwater'] < final['highwater']
        assert not final['pending']
        assert max(sizes) <= scan_module.BUDGET and len(sizes) > 1
        # Inventory-only terminal states: isolate the scan refusal from the
        # existing active-task/output guards (not execution settlement evidence).
        authority.db._execute_write(lambda c: c.execute("UPDATE hosted_room_driver_tasks SET status='settled'"))
        with authority.db._read_ctx() as conn, pytest.raises(RuntimeStoreError, match='output_cleanup_pending'):
            require_room_retired(conn, 'room')
        for _ in range(3):
            service._prepare_terminal_tasks(room)
        with authority.db._read_ctx() as conn:
            require_room_retired(conn, 'room')
        authority.db._execute_write(lambda c: c.execute('DROP TRIGGER hosted_task_scan_update'))
        changes = authority.db._conn.total_changes
        with authority.db._read_ctx() as conn, pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            scan_module.pending(conn, 'room')
        assert authority.db._conn.total_changes == changes
