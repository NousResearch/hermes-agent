"""Kanban run_task's claim check and owner_admitted marker run off the owner loop (B1-26): the
Kanban DB is the cross-profile dispatch bus (busy_timeout up to 120 s), so a dispatcher holding
its writer must not freeze every session on the loop. Real Kanban SQLite, a real held writer."""
import asyncio
from itertools import pairwise
from contextlib import closing
import sqlite3
import threading
import time
from types import SimpleNamespace

HOLD_S = 3.0


def test_run_task_kanban_write_never_blocks_the_loop(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from gateway.session_kanban import run_task
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env',
                        lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned', assignee='default', workspace_kind='dir', workspace_path=str(workspace))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
        board_db = conn.execute('PRAGMA database_list').fetchone()[2]
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    monkeypatch.setattr('gateway.session_local.create_local_session', lambda *a, **k: SimpleNamespace(
        session_id='owner-session', profile_id=str(tmp_path)))
    calls = []

    async def submit(actor, submission):
        calls.append('submit')
        return SimpleNamespace(admission_id='a')

    async def resume(ref, params):
        calls.append('resume')
    db = SimpleNamespace(get_session=lambda sid: None, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board='owned', task_id=task_id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    held = threading.Event()

    def dispatcher_holds_the_bus():
        rival = sqlite3.connect(board_db, isolation_level=None)
        rival.execute('BEGIN IMMEDIATE')
        held.set()
        time.sleep(HOLD_S)  # health: allow HX013 -- the held writer's duration IS the behaviour under test
        rival.execute('ROLLBACK')
        rival.close()
    monkeypatch.setattr('gateway.session_kanban.asdict', lambda receipt: {})

    async def scenario():
        writer = threading.Thread(target=dispatcher_holds_the_bus, daemon=True)
        writer.start()
        assert await asyncio.to_thread(held.wait, 10)
        ticks = []

        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(.05)  # health: allow HX013 -- the loop heartbeat probe; its gaps are the measurement
        beat = asyncio.create_task(ticker())
        try:
            async with asyncio.timeout(30):
                await run_task(connection, params)
        finally:
            beat.cancel()
        ticks.append(time.monotonic())  # a loop blocked from the first tick still yields one gap
        await asyncio.to_thread(writer.join, 10)
        return max(b - a for a, b in pairwise(ticks))
    gap = asyncio.run(scenario())
    assert calls == ['resume', 'submit']
    with closing(connect(board='owned')) as conn:
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='owner_admitted'", (task_id,)).fetchone()[0] == 1
    assert gap < HOLD_S / 2, f'the owner loop stalled {gap:.2f}s behind a held Kanban writer'
