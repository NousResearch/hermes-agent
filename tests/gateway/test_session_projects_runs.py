"""/v1/runs executes a project session's turn inside the project folder (``sessions.project_id``).

The run's terminal starts in the folder, context discovery (AGENTS.md / CLAUDE.md / project skills)
resolves there, and the agent is built with that cwd.
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import projects_db as pdb
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def adapter(session_db, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def project_folder(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "AGENTS.md").write_text("project rules\n", encoding="utf-8")
    with pdb.connect_closing() as conn:
        pdb.create_project(conn, name="Proj", slug="proj", folders=[str(folder)], board_slug="proj")
    return str(folder)


def _app(adapter):
    app = web.Application()
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _pwd_agent(seen: dict):
    """An agent whose turn runs the REAL terminal tool (``pwd``) and records where context
    discovery resolves, then answers with the terminal output."""
    agent = MagicMock()

    def run_conversation(user_message=None, conversation_history=None, task_id=None, **_kw):
        import json
        from agent.runtime_cwd import resolve_context_cwd
        from tools.terminal_tool import terminal_tool
        seen["context_cwd"] = str(resolve_context_cwd() or "")
        out = json.loads(terminal_tool("pwd", task_id=task_id))
        seen["pwd"] = (out.get("output") or "").strip()
        return {"final_response": seen["pwd"]}

    agent.run_conversation.side_effect = run_conversation
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    return agent


async def _wait_completed(cli, run_id: str) -> dict:
    for _ in range(200):
        status = await (await cli.get(f"/v1/runs/{run_id}")).json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish: {status}")


@pytest.mark.asyncio
async def test_run_terminal_pwd_is_the_project_folder(adapter, project_folder):
    seen: dict = {}
    created: dict = {}
    real_create = _pwd_agent(seen)

    def _create_agent(**kwargs):
        created.update(kwargs)
        return real_create

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions", json={"id": "s-run", "project": "proj"})
        assert resp.status == 201, await resp.text()
        with patch.object(adapter, "_create_agent", side_effect=_create_agent):
            started = await (await cli.post("/v1/runs", json={"input": "pwd?", "session_id": "s-run"})).json()
            status = await _wait_completed(cli, started["run_id"])

    assert status["status"] == "completed", status
    real = os.path.realpath(project_folder)
    assert os.path.realpath(seen["pwd"]) == real
    assert os.path.realpath(status["output"]) == real
    # Context discovery (AGENTS.md / CLAUDE.md / project skills) resolves in the project folder.
    assert os.path.realpath(seen["context_cwd"]) == real
    assert created["session_cwd"] == project_folder


@pytest.mark.asyncio
async def test_run_without_a_stored_session_keeps_the_default_cwd(adapter):
    created: dict = {}

    def _create_agent(**kwargs):
        created.update(kwargs)
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "ok"}
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        return agent

    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", side_effect=_create_agent):
            started = await (await cli.post("/v1/runs", json={"input": "hi"})).json()
            status = await _wait_completed(cli, started["run_id"])
    assert status["status"] == "completed"
    assert created["session_cwd"] == ""


def test_pin_run_workspace_keeps_an_agent_cd_until_the_folder_changes(tmp_path):
    from gateway.platforms.api_server_session_projects import pin_run_workspace
    from tools import terminal_tool as tt
    first, second, sub = tmp_path / "a", tmp_path / "b", tmp_path / "a" / "sub"
    sub.mkdir(parents=True)
    second.mkdir()
    task = "pin-task"
    try:
        pin_run_workspace(task, str(first))
        assert tt.get_session_cwd(task) == str(first)
        tt.record_session_cwd(task, str(sub))  # the agent cd'd during the turn
        pin_run_workspace(task, str(first))  # next turn, same folder: the cd survives
        assert tt.get_session_cwd(task) == str(sub)
        pin_run_workspace(task, str(second))  # the session moved: re-anchor
        assert tt.get_session_cwd(task) == str(second)
        pin_run_workspace(task, str(tmp_path / "missing"))  # a folder gone on this host is ignored
        assert tt.get_session_cwd(task) == str(second)
    finally:
        tt.clear_task_env_overrides(task)
