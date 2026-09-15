"""On-demand Kanban worker context occupancy."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import hermes_state
from agent.usage_anchor import capture_usage_anchor, set_usage_anchor
from cli import HermesCLI
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.active_sessions import active_session_registry_snapshot, try_acquire_active_session
from hermes_state import SessionDB
from run_agent import AIAgent


def _load_plugin():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_context_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def context_client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    mod = _load_plugin()
    monkeypatch.setattr(mod, "_worker_profile_home", lambda _profile: home)
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    return TestClient(app), home


def _running_task(title: str) -> tuple[str, int]:
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title=title, assignee="default", initial_status="running")
        assert kb.claim_task(conn, task_id) is not None
        kbd._set_worker_pid(conn, task_id, os.getpid())
        run_id = kb.get_task(conn, task_id).current_run_id
    assert run_id is not None
    return task_id, run_id


def _worker_lease(home: Path, session_id: str, task_id: str, run_id: int, *, context_max: int = 200_000):
    return try_acquire_active_session(
        session_id=session_id,
        surface="cli",
        config={},
        registry_home=home,
        metadata={
            "live_session_id": session_id,
            "kanban_task_id": task_id,
            "kanban_run_id": str(run_id),
            "kanban_board": "default",
            "context_max": context_max,
        },
    )


def test_context_endpoint_uses_the_current_worker_session_not_task_origin(context_client, monkeypatch):
    client, home = context_client
    messages = [
        {"role": "user", "content": "worker prompt"},
        {"role": "assistant", "content": "worker response"},
        {"role": "user", "content": "new tool result"},
    ]
    worker_session = "worker-session"
    parent_session = "parent-session"

    db = SessionDB(home / "state.db")
    try:
        db.create_session(worker_session, source="kanban", model="worker-model")
        for message in messages:
            db.append_message(worker_session, message["role"], content=message["content"])
        anchor = capture_usage_anchor(41_000, 800, messages[:2])
        db.patch_session_model_config(worker_session, {"_usage_anchor": anchor})
        db.create_session(parent_session, source="desktop", model="parent-model")
    finally:
        db.close()

    task_id, run_id = _running_task("worker context")
    lease, refusal = _worker_lease(home, worker_session, task_id, run_id)
    assert lease is not None and refusal is None
    real_session_db = hermes_state.SessionDB
    read_only_modes = []

    def recording_session_db(*args, **kwargs):
        read_only_modes.append(kwargs.get("read_only", False))
        return real_session_db(*args, **kwargs)

    monkeypatch.setattr(hermes_state, "SessionDB", recording_session_db)
    try:
        payload = client.get(f"/api/plugins/kanban/tasks/{task_id}/context").json()
    finally:
        lease.release()

    assert payload == {
        "available": True,
        "context_used": 41_818,
        "context_max": 200_000,
        "estimated": True,
        "source": "provider_usage_plus_estimate",
    }
    assert read_only_modes == [True]


def test_context_endpoint_reads_only_the_anchor_tail(context_client, monkeypatch):
    client, home = context_client
    worker_session = "large-worker-session"
    prefix = [{"role": "user", "content": f"old-{index}"} for index in range(200)]
    anchor = capture_usage_anchor(50_000, 1_000, prefix)

    db = SessionDB(home / "state.db")
    try:
        db.create_session(worker_session, source="kanban", model="worker-model")
        for message in prefix:
            db.append_message(worker_session, message["role"], content=message["content"])
        db.append_message(worker_session, "assistant", content="priced reply")
        db.append_message(worker_session, "user", content="small live tail")
        db.patch_session_model_config(worker_session, {"_usage_anchor": anchor})
    finally:
        db.close()

    task_id, run_id = _running_task("bounded read")

    original_tail_read = SessionDB.get_messages_as_conversation
    observed_offsets = []

    def require_bounded_reconstruction(self, *args, **kwargs):
        observed_offsets.append(kwargs.get("offset"))
        return original_tail_read(self, *args, **kwargs)

    monkeypatch.setattr(SessionDB, "get_messages_as_conversation", require_bounded_reconstruction)
    lease, refusal = _worker_lease(home, worker_session, task_id, run_id)
    assert lease is not None and refusal is None
    try:
        payload = client.get(f"/api/plugins/kanban/tasks/{task_id}/context").json()
    finally:
        lease.release()

    assert payload["available"] is True
    assert payload["context_used"] > 51_000
    assert observed_offsets == [len(prefix) - 1]


def test_context_endpoint_rejects_a_different_kanban_run(context_client):
    client, home = context_client
    db = SessionDB(home / "state.db")
    try:
        db.create_session("other-run", source="kanban", model="worker-model")
        db.append_message("other-run", "user", content="prompt")
        anchor = capture_usage_anchor(1_000, 100, [{"role": "user", "content": "prompt"}])
        db.patch_session_model_config("other-run", {"_usage_anchor": anchor})
    finally:
        db.close()

    task_id, run_id = _running_task("reassigned")
    lease, refusal = _worker_lease(home, "other-run", task_id, run_id + 1)
    assert lease is not None and refusal is None
    try:
        assert client.get(f"/api/plugins/kanban/tasks/{task_id}/context").json() == {"available": False}
    finally:
        lease.release()


def test_context_endpoint_rechecks_task_identity_after_the_worker_read(context_client, monkeypatch):
    client, _home = context_client
    task_id, _run_id = _running_task("racing reassignment")
    mod = sys.modules["hermes_kanban_plugin_context_test"]

    def reassign_during_read(_task, *, board):
        with kbc.connect_closing(board=board) as conn:
            conn.execute("UPDATE tasks SET worker_pid = worker_pid + 1 WHERE id = ?", (task_id,))
            conn.commit()
        return {
            "available": True,
            "context_used": 10,
            "context_max": 100,
            "estimated": False,
            "source": "provider_usage",
        }

    monkeypatch.setattr(mod, "_worker_context_payload", reassign_during_read)

    assert client.get(f"/api/plugins/kanban/tasks/{task_id}/context").json() == {"available": False}


def test_context_endpoint_reports_unavailable_without_exact_live_worker_identity(context_client):
    client, _home = context_client
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="not running", assignee="default", initial_status="blocked")

    assert client.get(f"/api/plugins/kanban/tasks/{task_id}/context").json() == {"available": False}


def test_cli_worker_lease_publishes_run_identity_and_effective_context(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "17")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")

    cli = object.__new__(HermesCLI)
    cli.session_id = "worker-session"
    cli.config = {}
    cli._active_session_lease = None
    cli.agent = None

    try:
        assert cli._claim_active_session("cli") is True
        initial = active_session_registry_snapshot(home, strict=True)[0]
        assert "context_max" not in initial["metadata"]
        cli.agent = type("Agent", (), {
            "context_compressor": type("Compressor", (), {"context_length": 272_000})()
        })()
        cli._refresh_active_session_metadata()
        entry = active_session_registry_snapshot(home, strict=True)[0]
        assert entry["metadata"] == {
            "live_session_id": "worker-session",
            "kanban_task_id": "t_worker",
            "kanban_run_id": "17",
            "kanban_board": "board-a",
            "context_max": 272_000,
        }
    finally:
        cli._release_active_session()


def test_real_agent_persists_kanban_source_and_usage_anchor(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    db = SessionDB(home / "state.db")
    agent = AIAgent(
        model="fixture-model",
        provider="openai-compat",
        api_key="fixture",
        base_url="http://127.0.0.1:1/v1",
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        session_id="real-worker-session",
        session_db=db,
    )
    try:
        agent._ensure_db_session()
        messages = [{"role": "user", "content": "work kanban task t_worker"}]
        db.append_message(agent.session_id, "user", content=messages[0]["content"])
        anchor = capture_usage_anchor(12_345, 321, messages)
        set_usage_anchor(agent, anchor)

        row = db.get_session(agent.session_id)
        assert row["source"] == "kanban"
        assert db.get_session_model_config_value(agent.session_id, "_usage_anchor") == anchor
    finally:
        agent.close()
