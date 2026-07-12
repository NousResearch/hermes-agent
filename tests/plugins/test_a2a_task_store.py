from __future__ import annotations

import asyncio
import logging
import pytest
import stat

from plugins.platforms.a2a import task_store


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _context(owner: str):
    from a2a.server.context import ServerCallContext
    from plugins.platforms.a2a.server import AuthenticatedA2AUser

    return ServerCallContext(user=AuthenticatedA2AUser(owner))


def _task(task_id: str, state: int):
    from a2a.types.a2a_pb2 import Task, TaskStatus

    return Task(id=task_id, context_id="context", status=TaskStatus(state=state))


@pytest.mark.asyncio
async def test_database_store_is_scoped_to_authenticated_sdk_user(hermes_home):
    from a2a.types.a2a_pb2 import TASK_STATE_COMPLETED

    store = task_store.create_task_store()
    await store.save(_task("task-1", TASK_STATE_COMPLETED), _context("alice"))

    assert (await store.get("task-1", _context("alice"))).id == "task-1"
    assert await store.get("task-1", _context("bob")) is None
    assert task_store.tasks_path() == hermes_home / "a2a" / "tasks.db"
    assert stat.S_IMODE(task_store.tasks_path().stat().st_mode) == 0o600
    await store.close()


@pytest.mark.asyncio
async def test_cross_owner_task_id_overwrite_is_rejected(hermes_home):
    from a2a.types.a2a_pb2 import TASK_STATE_COMPLETED

    store = task_store.create_task_store()
    await store.save(_task("shared-id", TASK_STATE_COMPLETED), _context("alice"))

    with pytest.raises(PermissionError, match="different owner"):
        await store.save(_task("shared-id", TASK_STATE_COMPLETED), _context("bob"))

    assert (await store.get("shared-id", _context("alice"))).id == "shared-id"
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_cross_owner_save_is_atomic(hermes_home):
    from a2a.types.a2a_pb2 import TASK_STATE_COMPLETED

    first = task_store.create_task_store()
    second = task_store.create_task_store()
    await first.initialize()
    await second.initialize()
    start = asyncio.Event()

    async def save(store, owner):
        await start.wait()
        await store.save(_task("raced-id", TASK_STATE_COMPLETED), _context(owner))
        return owner

    left = asyncio.create_task(save(first, "alice"))
    right = asyncio.create_task(save(second, "bob"))
    start.set()
    results = await asyncio.gather(left, right, return_exceptions=True)

    assert sum(isinstance(result, PermissionError) for result in results) == 1
    assert sum(isinstance(result, str) for result in results) == 1
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_unauthenticated_context_is_rejected(hermes_home):
    from a2a.server.context import ServerCallContext
    from a2a.types.a2a_pb2 import TASK_STATE_COMPLETED

    store = task_store.create_task_store()
    with pytest.raises(PermissionError, match="authenticated"):
        await store.save(_task("task-1", TASK_STATE_COMPLETED), ServerCallContext())
    await store.close()


@pytest.mark.asyncio
async def test_restart_reconciliation_marks_nonterminal_tasks_failed(hermes_home):
    from a2a.types.a2a_pb2 import TASK_STATE_FAILED, TASK_STATE_WORKING

    store = task_store.create_task_store()
    await store.save(_task("orphan", TASK_STATE_WORKING), _context("alice"))

    assert await task_store.reconcile_orphaned_tasks(store) == 1
    reconciled = await store.get("orphan", _context("alice"))
    assert reconciled.status.state == TASK_STATE_FAILED
    assert reconciled.metadata["interrupted"] == "server_restart"
    await store.close()


@pytest.mark.asyncio
async def test_sdk_task_store_logs_never_include_task_ids(hermes_home, caplog):
    from a2a.types.a2a_pb2 import TASK_STATE_COMPLETED

    store = task_store.create_task_store()
    secret_task_id = "secret-task-id-must-not-log"
    with caplog.at_level(logging.DEBUG):
        await store.save(_task(secret_task_id, TASK_STATE_COMPLETED), _context("alice"))
        await store.get(secret_task_id, _context("alice"))
    sdk_logs = "\n".join(
        record.getMessage() for record in caplog.records if record.name.startswith("a2a.server")
    )
    assert secret_task_id not in sdk_logs
    await store.close()
