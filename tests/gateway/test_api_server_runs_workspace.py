"""Tests: POST /v1/runs accepts body cwd and X-Hermes-Workspace.

Covers client-supplied per-run working directories (#29531 / #90439): validation,
binding into session/task cwd machinery, and isolation from a shared default
terminal environment.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware
from tools import terminal_tool


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


class TestValidateWorkspaceCwd:
    def test_accepts_existing_absolute_dir(self, tmp_path: Path):
        adapter = _make_adapter()
        cwd, err = adapter._validate_workspace_cwd(str(tmp_path), source="cwd")
        assert err is None
        assert Path(cwd) == tmp_path.resolve()

    def test_rejects_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter()
        cwd, err = adapter._validate_workspace_cwd("relative-dir", source="cwd")
        assert cwd is None
        assert err is not None
        assert err.status == 400

    def test_rejects_missing_dir(self, tmp_path: Path):
        adapter = _make_adapter()
        missing = tmp_path / "nope"
        cwd, err = adapter._validate_workspace_cwd(str(missing), source="cwd")
        assert cwd is None
        assert err is not None
        assert err.status == 400

    def test_rejects_file_path(self, tmp_path: Path):
        adapter = _make_adapter()
        file_path = tmp_path / "file.txt"
        file_path.write_text("x", encoding="utf-8")
        cwd, err = adapter._validate_workspace_cwd(str(file_path), source="cwd")
        assert cwd is None
        assert err is not None
        assert err.status == 400


class TestResolveRunWorkspace:
    def test_prefers_either_source(self, tmp_path: Path):
        adapter = _make_adapter()
        request = MagicMock()
        request.headers = {"X-Hermes-Workspace": str(tmp_path)}
        cwd, err = adapter._resolve_run_workspace(request, {})
        assert err is None
        assert Path(cwd) == tmp_path.resolve()

        request.headers = {}
        cwd, err = adapter._resolve_run_workspace(request, {"cwd": str(tmp_path)})
        assert err is None
        assert Path(cwd) == tmp_path.resolve()

    def test_conflict_when_header_and_body_disagree(self, tmp_path: Path):
        adapter = _make_adapter()
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        request = MagicMock()
        request.headers = {"X-Hermes-Workspace": str(a)}
        cwd, err = adapter._resolve_run_workspace(request, {"cwd": str(b)})
        assert cwd is None
        assert err is not None
        assert err.status == 400

    def test_agreeing_header_and_body(self, tmp_path: Path):
        adapter = _make_adapter()
        request = MagicMock()
        request.headers = {"X-Hermes-Workspace": str(tmp_path)}
        cwd, err = adapter._resolve_run_workspace(request, {"cwd": str(tmp_path)})
        assert err is None
        assert Path(cwd) == tmp_path.resolve()

    def test_omitted_returns_empty(self):
        adapter = _make_adapter()
        request = MagicMock()
        request.headers = {}
        cwd, err = adapter._resolve_run_workspace(request, {"input": "hi"})
        assert err is None
        assert cwd == ""


class TestBindWorkspaceCwd:
    def test_logical_cwd_and_session_record(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from agent.runtime_cwd import resolve_agent_cwd

        monkeypatch.setattr(terminal_tool, "_session_cwd", {})
        adapter = _make_adapter()
        session_id = "workspace-session"
        tokens = adapter._bind_api_server_session(
            chat_id=session_id,
            session_key=session_id,
            session_id=session_id,
            cwd=str(tmp_path),
        )
        try:
            assert resolve_agent_cwd() == tmp_path.resolve()
            assert terminal_tool.get_session_cwd(session_id) == str(tmp_path)
        finally:
            from gateway.session_context import clear_session_vars

            clear_session_vars(tokens)

    def test_bind_does_not_mutate_shared_default_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """API workspace binding must not move another session's shared env."""
        monkeypatch.setattr(terminal_tool, "_session_cwd", {})
        shared_env = types.SimpleNamespace(cwd=str(tmp_path / "other-session"))
        monkeypatch.setattr(
            terminal_tool,
            "_active_environments",
            {"default": shared_env},
        )
        adapter = _make_adapter()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        tokens = adapter._bind_api_server_session(
            chat_id="workspace-session",
            session_key="workspace-session",
            session_id="workspace-session",
            cwd=str(workspace),
        )
        try:
            assert shared_env.cwd == str(tmp_path / "other-session")
            assert terminal_tool.get_session_cwd("workspace-session") == str(workspace)
        finally:
            from gateway.session_context import clear_session_vars

            clear_session_vars(tokens)


@pytest.mark.asyncio
class TestRunsWorkspaceHTTP:
    async def test_rejects_missing_cwd_directory(self, tmp_path: Path):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        missing = tmp_path / "missing"
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/runs",
                json={"input": "hello", "cwd": str(missing)},
            )
            assert resp.status == 400
            payload = await resp.json()
            assert payload["error"]["code"] == "invalid_workspace"

    async def test_body_cwd_reaches_bind_and_task_override(self, tmp_path: Path):
        adapter = _make_adapter()
        workspace = tmp_path / "project"
        workspace.mkdir()
        captured: dict = {}

        def _capture_create_agent(**kwargs):
            from agent.runtime_cwd import resolve_agent_cwd

            captured["create_cwd"] = resolve_agent_cwd()
            agent = MagicMock()
            agent.run_conversation.return_value = {
                "final_response": "ok",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            return agent

        original_bind = adapter._bind_api_server_session

        def _capture_bind(**kwargs):
            captured["bind_cwd"] = kwargs.get("cwd", "")
            return original_bind(**kwargs)

        with (
            patch.object(adapter, "_create_agent", side_effect=_capture_create_agent),
            patch.object(adapter, "_bind_api_server_session", side_effect=_capture_bind),
            patch.object(terminal_tool, "clear_task_env_overrides"),
        ):
            # Ensure overrides map is clean for assertions after the run.
            terminal_tool.clear_task_env_overrides("ws-sess-1")
            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "session_id": "ws-sess-1",
                        "cwd": str(workspace),
                    },
                )
                assert resp.status == 202, await resp.text()
                body = await resp.json()
                run_id = body["run_id"]

                # Wait until the background run finishes.
                for _ in range(100):
                    status_resp = await client.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status.get("status") in {"completed", "failed", "cancelled"}:
                        break
                    import asyncio

                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("run did not complete")

                assert status.get("status") == "completed"
                assert Path(captured["bind_cwd"]) == workspace.resolve()
                assert captured["create_cwd"] == workspace.resolve()
                assert terminal_tool.get_session_cwd("ws-sess-1") == str(workspace.resolve()) or (
                    terminal_tool.get_session_cwd("ws-sess-1") == str(workspace)
                )

    async def test_header_cwd_accepted(self, tmp_path: Path):
        adapter = _make_adapter()
        workspace = tmp_path / "hdr"
        workspace.mkdir()
        captured: dict = {}

        def _capture_bind(**kwargs):
            captured["bind_cwd"] = kwargs.get("cwd", "")
            return []

        with (
            patch.object(adapter, "_create_agent", return_value=MagicMock(
                run_conversation=MagicMock(return_value={
                    "final_response": "ok", "messages": [], "api_calls": 0, "tools": [],
                }),
                session_prompt_tokens=0,
                session_completion_tokens=0,
                session_total_tokens=0,
            )),
            patch.object(adapter, "_bind_api_server_session", side_effect=_capture_bind),
        ):
            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "hdr-sess"},
                    headers={"X-Hermes-Workspace": str(workspace)},
                )
                assert resp.status == 202, await resp.text()
                body = await resp.json()
                run_id = body["run_id"]
                import asyncio

                for _ in range(100):
                    status_resp = await client.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status.get("status") in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.05)
                assert Path(captured["bind_cwd"]) == workspace.resolve()

    async def test_capabilities_advertise_workspace(self):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["workspace_header"] == "X-Hermes-Workspace"
            assert data["features"]["runs_cwd"] is True
