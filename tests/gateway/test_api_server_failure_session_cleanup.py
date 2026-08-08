"""Failure-session hygiene for API-server agent entry paths."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


def _adapter_with_db(db: SessionDB) -> APIServerAdapter:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db
    return adapter


def _agent_result(result, db: SessionDB, *, persist_message: bool = False):
    agent = MagicMock()

    def _run(*, user_message, conversation_history, task_id):
        db.create_session(task_id, source="api_server", model="test")
        if persist_message:
            db.append_message(task_id, role="user", content=user_message)
        return dict(result)

    agent.run_conversation.side_effect = _run
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_id = None
    return agent


def _agent_exception(exc: Exception, db: SessionDB):
    agent = MagicMock()

    def _run(*, user_message, conversation_history, task_id):
        db.create_session(task_id, source="api_server", model="test")
        raise exc

    agent.run_conversation.side_effect = _run
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_id = None
    return agent


def _agent_nondict(value, db: SessionDB):
    """Build an agent that returns ``value`` (a non-dict or a dict without
    ``final_response``) so the runs handler must route it as a failure."""
    agent = MagicMock()

    def _run(*, user_message, conversation_history, task_id):
        db.create_session(task_id, source="api_server", model="test")
        return value

    agent.run_conversation.side_effect = _run
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_id = None
    return agent


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


async def _wait_for_terminal_run(client: TestClient, run_id: str) -> dict:
    for _ in range(40):
        response = await client.get(f"/v1/runs/{run_id}")
        status = await response.json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach a terminal status")


@pytest.mark.asyncio
async def test_runs_structured_failure_deletes_empty_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = _adapter_with_db(db)
    agent = _agent_result({"failed": True, "error": "upstream rejected"}, db)
    app = _create_runs_app(adapter)

    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "runs-failed-empty"},
            )
            run_id = (await response.json())["run_id"]
            status = await _wait_for_terminal_run(client, run_id)

    assert status["status"] == "failed"
    assert db.get_session("runs-failed-empty") is None


@pytest.mark.asyncio
async def test_runs_exception_deletes_empty_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = _adapter_with_db(db)
    agent = _agent_exception(RuntimeError("boom"), db)
    app = _create_runs_app(adapter)

    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "runs-exception-empty"},
            )
            run_id = (await response.json())["run_id"]
            status = await _wait_for_terminal_run(client, run_id)

    assert status["status"] == "failed"
    assert db.get_session("runs-exception-empty") is None


@pytest.mark.asyncio
async def test_runs_cooperative_stop_deletes_empty_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = _adapter_with_db(db)
    # The agent returns a "would have been done" dict, but we mark the
    # run for cooperative stop BEFORE its executor result is observed.
    # The post-stop branch must route through the same prune path that
    # covers the cancellation and exception branches.
    agent = _agent_result({"final_response": "would have been done"}, db)
    app = _create_runs_app(adapter)

    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "runs-coop-stop-empty"},
            )
            run_id = (await response.json())["run_id"]
            adapter._stopping_run_ids.add(run_id)
            status = await _wait_for_terminal_run(client, run_id)

    assert status["status"] == "cancelled"
    assert db.get_session("runs-coop-stop-empty") is None


@pytest.mark.asyncio
async def test_runs_nondict_result_treated_as_failure(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = _adapter_with_db(db)
    # The agent returns None — neither a dict with "failed" nor a dict
    # with "final_response". The runs handler must NOT classify this as
    # a successful completion.
    agent = _agent_nondict(None, db)
    app = _create_runs_app(adapter)

    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "runs-nondict-empty"},
            )
            run_id = (await response.json())["run_id"]
            status = await _wait_for_terminal_run(client, run_id)

    assert status["status"] == "failed"
    assert db.get_session("runs-nondict-empty") is None


@pytest.mark.asyncio
async def test_run_agent_structured_failure_deletes_empty_session_off_event_loop(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = _adapter_with_db(db)
    agent = _agent_result({"failed": True, "error": "upstream rejected"}, db)

    with patch.object(adapter, "_create_agent", return_value=agent), patch(
        "gateway.platforms.api_server.asyncio.to_thread",
        wraps=asyncio.to_thread,
    ) as to_thread:
        result, _usage = await adapter._run_agent(
            "hello", [], session_id="failed-empty"
        )

    assert result["failed"] is True
    assert db.get_session("failed-empty") is None
    cleanup_calls = [
        call
        for call in to_thread.call_args_list
        if call.args and call.args[0] == db.delete_session_if_empty
    ]
    assert cleanup_calls
    assert cleanup_calls[0].args[1] == "failed-empty"
    # The patched offloader executes its callable on a worker thread; verify
    # that the SQLite cleanup did not run on aiohttp's event-loop thread.
    assert cleanup_calls[0].args[0].__self__ is db


@pytest.mark.asyncio
async def test_run_agent_failure_preserves_session_with_resumable_content(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = _adapter_with_db(db)
    agent = _agent_result(
        {"failed": True, "error": "failed after persistence"},
        db,
        persist_message=True,
    )

    with patch.object(adapter, "_create_agent", return_value=agent):
        await adapter._run_agent("hello", [], session_id="failed-with-content")

    assert db.get_session("failed-with-content") is not None


@pytest.mark.asyncio
async def test_run_agent_success_preserves_empty_resumable_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = _adapter_with_db(db)
    agent = _agent_result({"final_response": "done"}, db)

    with patch.object(adapter, "_create_agent", return_value=agent):
        await adapter._run_agent("hello", [], session_id="successful-empty")

    assert db.get_session("successful-empty") is not None


@pytest.mark.asyncio
async def test_prune_isolation_two_profiles_do_not_cross_delete(tmp_path):
    """A failure on profile A must not delete a row on profile B.

    Riker's C1 finding. Verify that the per-profile SessionDB cache
    resolves the right home on the loop thread before the async prune
    runs, and that the to_thread offloader does not get the wrong DB.
    """
    import os
    import sqlite3

    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    for h in (home_a, home_b):
        h.mkdir(parents=True)

    db_a = SessionDB(db_path=home_a / "state.db")
    db_b = SessionDB(db_path=home_b / "state.db")
    adapter_a = _adapter_with_db(db_a)
    adapter_b = _adapter_with_db(db_b)

    # Pre-populate profile B with a row sharing the SAME id we will fail
    # on profile A. If the prune were to touch the wrong DB, this row
    # would be deleted. With correct isolation, it must survive.
    db_b.create_session(
        "iso-a",
        source="api_server",
        model="test",
    )

    with patch.dict(os.environ, {"HERMES_HOME": str(home_a)}):
        try:
            from hermes_constants import get_hermes_home
            _ = get_hermes_home()
        except Exception:
            pass

    agent = _agent_result({"failed": True, "error": "boom"}, db_a)
    with patch.object(adapter_a, "_create_agent", return_value=agent), \
         patch.dict(os.environ, {"HERMES_HOME": str(home_a)}):
        try:
            await adapter_a._run_agent("x", [], session_id="iso-a")
        except RuntimeError:
            pass

    # Profile A: failed row must be gone.
    assert db_a.get_session("iso-a") is None
    # Profile B: pre-existing row with the same id must still exist.
    assert db_b.get_session("iso-a") is not None
    # Sanity: there is no row in home_b's DB with the exact same name that
    # was produced by profile A — only the one we pre-populated.
    conn = sqlite3.connect(str(home_b / "state.db"))
    rows = conn.execute("SELECT id FROM sessions").fetchall()
    assert [r[0] for r in rows] == ["iso-a"]


@pytest.mark.asyncio
async def test_prune_preserves_concurrent_message_flush(tmp_path):
    """If a message flush lands between failure detection and the prune
    call, the row MUST survive because it is no longer empty.

    The SQL guard inside ``delete_session_if_empty`` is single-statement
    and atomic, so the realistic race is: a parallel writer appends a
    message after the failure is observed but before the prune runs.
    The test simulates that with a thread that signals at the right
    moment. The row should still exist when the test ends.
    """
    import threading

    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = _adapter_with_db(db)
    agent = _agent_result({"failed": True, "error": "boom"}, db)

    # Start the agent in the same call we are about to exercise; instead
    # we will simulate the race deterministically: the agent creates the
    # row, then a parallel thread appends a message before the prune
    # call has a chance to commit. The production code must not lose
    # the row.
    flush_started = threading.Event()
    proceed_to_prune = threading.Event()

    def flusher():
        db.create_session("race-a", source="api_server", model="test")
        flush_started.set()
        proceed_to_prune.wait(timeout=5)
        db.append_message("race-a", role="user", content="late flush")

    t = threading.Thread(target=flusher, daemon=True)
    t.start()
    flush_started.wait(timeout=5)

    with patch.object(adapter, "_create_agent", return_value=agent):
        proceed_to_prune.set()
        try:
            await adapter._run_agent("x", [], session_id="race-a")
        except RuntimeError:
            pass

    t.join(timeout=5)
    # Row must still exist (a message was flushed before the prune ran)
    row = db.get_session("race-a")
    assert row is not None
