"""Tests for the /v1/runs ``workspace`` field: admission-time containment
validation against the gateway's OWN working root, task-scoped tool-cwd
binding during the run, and cleanup on the terminal path.

Covers:
- ``_workspace_contained_in_root`` — realpath containment matrix (sibling
  prefix, ``..`` traversal, symlink escape, non-absolute, exact root)
- ``_request_run_workspace`` — body extraction, fail-closed root resolution,
  nonexistent-directory rejection
- POST /v1/runs end-to-end — a validated workspace reaches the run's
  task-scoped terminal/filesystem cwd (``resolve_task_overrides`` /
  ``get_session_cwd``) and the logical cwd (``resolve_agent_cwd``), and is
  cleared when the run completes; an out-of-root workspace is ignored while
  the run is still admitted at the default cwd.
"""

import asyncio
import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from gateway.platforms.api_server_runs import (
    _request_run_workspace,
    _workspace_contained_in_root,
)


# ---------------------------------------------------------------------------
# Harness (mirrors tests/gateway/test_api_server_runs.py)
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _capturing_agent(captured: dict) -> MagicMock:
    """Mock agent recording the cwd surfaces the terminal/file layers resolve
    DURING run_conversation (executor thread, run context bound)."""
    from agent.runtime_cwd import resolve_agent_cwd
    from tools.terminal_tool import get_session_cwd, resolve_task_overrides

    def _run(user_message=None, conversation_history=None, task_id=None, **kwargs):
        captured["task_id"] = task_id
        captured["agent_cwd"] = str(resolve_agent_cwd())
        captured["task_override_cwd"] = resolve_task_overrides(task_id).get("cwd")
        captured["session_cwd_record"] = get_session_cwd(task_id)
        return {"final_response": "done"}

    mock_agent = MagicMock()
    mock_agent.run_conversation.side_effect = _run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0
    return mock_agent


async def _wait_completed(cli, run_id: str) -> None:
    for _ in range(40):
        status = await (await cli.get(f"/v1/runs/{run_id}")).json()
        if status["status"] == "completed":
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} did not complete")


# ---------------------------------------------------------------------------
# _workspace_contained_in_root — pure containment matrix (real filesystem)
# ---------------------------------------------------------------------------


class TestWorkspaceContainedInRoot:
    def test_exact_root_allowed(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        assert _workspace_contained_in_root(str(root), str(root)) == os.path.realpath(root)

    def test_child_directory_allowed(self, tmp_path):
        root = tmp_path / "workspace"
        child = root / "project"
        child.mkdir(parents=True)
        assert _workspace_contained_in_root(str(child), str(root)) == os.path.realpath(child)

    def test_dotdot_that_stays_inside_normalizes_to_contained(self, tmp_path):
        root = tmp_path / "workspace"
        child = root / "project"
        child.mkdir(parents=True)
        candidate = str(root / "other" / ".." / "project")
        assert _workspace_contained_in_root(candidate, str(root)) == os.path.realpath(child)

    def test_sibling_prefix_rejected(self, tmp_path):
        root = tmp_path / "workspace"
        sibling = tmp_path / "workspace-other"
        sibling.mkdir()
        root.mkdir()
        assert _workspace_contained_in_root(str(sibling), str(root)) is None

    def test_dotdot_escape_rejected(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "secret"
        outside.mkdir()
        candidate = str(root / ".." / "secret")
        assert _workspace_contained_in_root(candidate, str(root)) is None

    def test_symlink_escape_rejected(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = root / "escape"
        try:
            link.symlink_to(outside)
        except OSError as exc:  # Windows without symlink privilege
            pytest.skip(f"cannot create symlink: {exc}")
        assert _workspace_contained_in_root(str(link), str(root)) is None

    def test_symlink_within_root_canonicalized(self, tmp_path):
        root = tmp_path / "workspace"
        real = root / "real"
        real.mkdir(parents=True)
        link = root / "alias"
        try:
            link.symlink_to(real)
        except OSError as exc:
            pytest.skip(f"cannot create symlink: {exc}")
        assert _workspace_contained_in_root(str(link), str(root)) == os.path.realpath(real)

    @pytest.mark.parametrize("candidate", ["", "   ", "relative/path", "./workspace", "project"])
    def test_non_absolute_or_empty_rejected(self, tmp_path, candidate):
        root = tmp_path / "workspace"
        root.mkdir()
        assert _workspace_contained_in_root(candidate, str(root)) is None

    def test_root_resolution_error_fails_closed(self, tmp_path, monkeypatch):
        def _boom(_):
            raise OSError("unresolvable")

        monkeypatch.setattr(os.path, "realpath", _boom)
        assert _workspace_contained_in_root("/workspace/x", "/workspace") is None


# ---------------------------------------------------------------------------
# _request_run_workspace — body extraction + root authority
# ---------------------------------------------------------------------------


class TestRequestRunWorkspace:
    @pytest.mark.parametrize("body", [
        {}, {"workspace": ""}, {"workspace": "   "}, {"workspace": None},
        {"workspace": 42}, {"workspace": ["/workspace"]}, "not-a-dict", None,
    ])
    def test_absent_or_malformed_returns_empty(self, body):
        assert _request_run_workspace(body) == ""

    def test_valid_child_returns_canonical_path(self, tmp_path, monkeypatch):
        root = tmp_path / "workspace"
        child = root / "project"
        child.mkdir(parents=True)
        monkeypatch.setattr(
            "agent.runtime_cwd.resolve_agent_cwd", lambda: root)
        assert _request_run_workspace({"workspace": str(child)}) == os.path.realpath(child)

    def test_out_of_root_rejected(self, tmp_path, monkeypatch):
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "workspace-evil"
        outside.mkdir()
        monkeypatch.setattr("agent.runtime_cwd.resolve_agent_cwd", lambda: root)
        assert _request_run_workspace({"workspace": str(outside)}) == ""

    def test_nonexistent_contained_dir_rejected(self, tmp_path, monkeypatch):
        root = tmp_path / "workspace"
        root.mkdir()
        monkeypatch.setattr("agent.runtime_cwd.resolve_agent_cwd", lambda: root)
        assert _request_run_workspace({"workspace": str(root / "ghost")}) == ""

    def test_root_resolution_error_fails_closed(self, monkeypatch):
        def _boom():
            raise OSError("cwd deleted")

        monkeypatch.setattr("agent.runtime_cwd.resolve_agent_cwd", _boom)
        assert _request_run_workspace({"workspace": "/workspace/x"}) == ""


# ---------------------------------------------------------------------------
# POST /v1/runs end-to-end — binding during the run, cleanup at completion
# ---------------------------------------------------------------------------


class TestRunWorkspaceEndToEnd:
    @pytest.mark.asyncio
    async def test_valid_workspace_binds_tool_cwd_and_clears_on_completion(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "workspace"
        project = root / "project"
        project.mkdir(parents=True)
        # The gateway's own working root (terminal.cwd authority), scope-aware
        # via TERMINAL_CWD; resolve_agent_cwd() stays the REAL resolver.
        monkeypatch.setenv("TERMINAL_CWD", str(root))

        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        captured: dict = {}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=_capturing_agent(captured)):
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "ws-session",
                          "workspace": str(project)},
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await _wait_completed(cli, run_id)

        expected = os.path.realpath(project)
        # Logical cwd (system prompt / context discovery / coding context)…
        assert captured["agent_cwd"] == expected
        # …and the task-scoped terminal/filesystem tool cwd agree.
        assert captured["task_override_cwd"] == expected
        assert captured["session_cwd_record"] == expected
        # Terminal-path cleanup: override AND cwd record dropped after completion.
        from tools.terminal_tool import get_session_cwd, resolve_task_overrides
        assert resolve_task_overrides(captured["task_id"]) == {}
        assert get_session_cwd(captured["task_id"]) is None

    @pytest.mark.asyncio
    async def test_no_workspace_keeps_gateway_default_cwd(self, tmp_path, monkeypatch):
        root = tmp_path / "workspace"
        root.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(root))

        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        captured: dict = {}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=_capturing_agent(captured)):
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                await _wait_completed(cli, (await resp.json())["run_id"])

        assert captured["agent_cwd"] == os.path.realpath(root)
        assert captured["task_override_cwd"] is None
        assert captured["session_cwd_record"] is None

    @pytest.mark.asyncio
    async def test_out_of_root_workspace_ignored_but_run_admitted(self, tmp_path, monkeypatch):
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "workspace-evil"
        outside.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(root))

        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        captured: dict = {}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=_capturing_agent(captured)):
                resp = await cli.post(
                    "/v1/runs", json={"input": "hello", "workspace": str(outside)})
                assert resp.status == 202
                await _wait_completed(cli, (await resp.json())["run_id"])

        # Fail-safe: default gateway cwd, no task override registered.
        assert captured["agent_cwd"] == os.path.realpath(root)
        assert captured["task_override_cwd"] is None
