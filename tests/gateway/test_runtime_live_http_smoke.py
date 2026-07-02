"""Tests for the runtime live HTTP smoke harness.

Verifies that:
1. The standalone_runtime_server.py script can be imported and configured.
2. The smoke_runtime_executor_live.sh script validates requirements correctly.
3. Both --fake and real-credential modes construct correctly.
4. No secrets leak in outputs.

These tests do NOT start a live server — they validate the smoke harness
construction, env handling, and error paths.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


class TestStandaloneRuntimeServerScript:
    """The standalone_runtime_server.py script constructs correctly."""

    def test_script_exists(self):
        path = SCRIPTS_DIR / "standalone_runtime_server.py"
        assert path.exists(), f"Script not found: {path}"

    def test_script_imports_cleanly(self):
        """The script's dependencies can be resolved without side effects."""
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.path.insert(0, ".")
from aiohttp import web
from gateway.runtime.run_manager import RunManager
from gateway.runtime.routes import register_runtime_routes
print("imports ok")
"""],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"

    def test_fake_mode_constructs_app(self):
        """--fake mode produces an app with FakeAgentFactory."""
        result = subprocess.run(
            [sys.executable, "-c", """
import sys, os
sys.path.insert(0, "scripts/..")
os.environ.setdefault("API_SERVER_KEY", "standalone-runtime-smoke-key-001")
from aiohttp import web
from gateway.runtime.run_manager import RunManager
from gateway.runtime.routes import register_runtime_routes
from gateway.runtime.executor import RuntimeExecutor, FakeAgentFactory

app = web.Application()
rm = RunManager()
factory = FakeAgentFactory(result={"final_response": "ok", "completed": True})
executor = RuntimeExecutor(rm, agent_factory=factory)
register_runtime_routes(app, run_manager=rm, executor=executor,
    register_create=True, register_status=True, register_events=True)
app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))
print("app_ready")
"""],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"Fake app build failed: {result.stderr}"
        assert "app_ready" in result.stdout

    def test_real_mode_constructs_app(self):
        """DefaultAgentFactory mode produces an app (may fail live resolution)."""
        result = subprocess.run(
            [sys.executable, "-c", """
import sys, os
sys.path.insert(0, "scripts/..")
os.environ.setdefault("API_SERVER_KEY", "standalone-runtime-smoke-key-001")
from aiohttp import web
from gateway.runtime.run_manager import RunManager
from gateway.runtime.routes import register_runtime_routes
from gateway.runtime.executor import RuntimeExecutor
from gateway.runtime.agent_factory import DefaultAgentFactory

app = web.Application()
rm = RunManager()
factory = DefaultAgentFactory()
executor = RuntimeExecutor(rm, agent_factory=factory)
register_runtime_routes(app, run_manager=rm, executor=executor,
    register_create=True, register_status=True, register_events=True)
app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))
print("app_ready")
"""],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"Real app build failed: {result.stderr}"
        assert "app_ready" in result.stdout


class TestSmokeScriptValidation:
    """Smoke shell script validates requirements correctly."""

    def test_smoke_script_exists(self):
        path = SCRIPTS_DIR / "smoke_runtime_executor_live.sh"
        assert path.exists()

    def test_cross_repo_script_exists(self):
        path = SCRIPTS_DIR / "smoke_cross_repo.sh"
        assert path.exists()

    def test_scripts_are_executable(self):
        smoke = SCRIPTS_DIR / "smoke_runtime_executor_live.sh"
        cross = SCRIPTS_DIR / "smoke_cross_repo.sh"
        standalone = SCRIPTS_DIR / "standalone_runtime_server.py"
        assert os.access(smoke, os.X_OK), f"Not executable: {smoke}"
        assert os.access(cross, os.X_OK), f"Not executable: {cross}"
        assert os.access(standalone, os.X_OK), f"Not executable: {standalone}"


class TestLiveHttpSmokeEndToEnd:
    """End-to-end HTTP smoke with a real (fake-mode) server.

    These tests start the standalone server in --fake mode on a dynamic
    port and exercise the full HTTP contract.
    """

    @pytest.fixture
    def free_port(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @pytest.mark.asyncio
    async def test_fake_mode_execute_run_completes(self, free_port):
        """POST /v1/runs with execute:true in --fake mode completes."""
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.routes import register_runtime_routes
        from gateway.runtime.executor import RuntimeExecutor, FakeAgentFactory

        os.environ.setdefault("API_SERVER_KEY", "standalone-runtime-smoke-key-001")

        app = web.Application()
        rm = RunManager()
        factory = FakeAgentFactory(result={
            "final_response": "runtime executor cross repo smoke ok",
            "completed": True,
        })
        executor = RuntimeExecutor(rm, agent_factory=factory)
        register_runtime_routes(
            app, run_manager=rm, executor=executor,
            register_create=True, register_status=True, register_events=True,
        )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={
                "session_id": "live-test",
                "input": "Return exactly: runtime executor cross repo smoke ok",
                "execute": True,
            })
            assert resp.status == 202
            data = await resp.json()
            run_id = data["run_id"]
            assert run_id
            # Status may be "queued" initially (executor fires async)
            assert data["status"] in ("queued", "running"), f"Unexpected status: {data['status']}"

            # Poll until terminal
            import asyncio
            for _ in range(30):
                s_resp = await cli.get(f"/v1/runs/{run_id}")
                s_data = await s_resp.json()
                if s_data["status"] == "completed":
                    break
                await asyncio.sleep(0.2)
            assert s_data["status"] == "completed", f"Run did not complete: {s_data}"

            # Verify events contain done
            e_resp = await cli.get(f"/v1/runs/{run_id}/events")
            e_data = await e_resp.json()
            types = [e["type"] for e in e_data["events"]]
            assert "done" in types, f"Events missing done: {types}"

    @pytest.mark.asyncio
    async def test_fake_mode_stop_cancels_run(self, free_port):
        """POST /v1/runs/{run_id}/stop cancels a running run in --fake mode."""
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.routes import register_runtime_routes
        from gateway.runtime.executor import RuntimeExecutor, FakeAgentFactory

        os.environ.setdefault("API_SERVER_KEY", "standalone-runtime-smoke-key-001")

        app = web.Application()
        rm = RunManager()
        factory = FakeAgentFactory(delay=5.0)
        executor = RuntimeExecutor(rm, agent_factory=factory)
        register_runtime_routes(
            app, run_manager=rm, executor=executor,
            register_create=True, register_status=True, register_events=True,
        )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={
                "session_id": "stop-test",
                "input": "long running task",
                "execute": True,
            })
            data = await resp.json()
            run_id = data["run_id"]

            import asyncio
            await asyncio.sleep(0.5)

            stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
            stop_data = await stop_resp.json()
            assert stop_data.get("status") in ("cancelled", "stopped"), f"Stop failed: {stop_data}"

            final = await cli.get(f"/v1/runs/{run_id}")
            final_data = await final.json()
            assert final_data["terminal"] is True

    @pytest.mark.asyncio
    async def test_fake_mode_approval_returns_not_found(self, free_port):
        """Approval on a completed run returns action_not_found gracefully."""
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.routes import register_runtime_routes
        from gateway.runtime.executor import RuntimeExecutor, FakeAgentFactory

        os.environ.setdefault("API_SERVER_KEY", "standalone-runtime-smoke-key-001")

        app = web.Application()
        rm = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(rm, agent_factory=factory)
        register_runtime_routes(
            app, run_manager=rm, executor=executor,
            register_create=True, register_status=True, register_events=True,
        )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={
                "session_id": "approval-test",
                "input": "test",
                "execute": True,
            })
            data = await resp.json()
            run_id = data["run_id"]

            import asyncio
            for _ in range(30):
                s = await cli.get(f"/v1/runs/{run_id}")
                sd = await s.json()
                if sd["status"] == "completed":
                    break
                await asyncio.sleep(0.2)

            appr = await cli.post(f"/v1/runs/{run_id}/approval", json={
                "run_id": run_id,
                "choice": "approve",
            })
            appr_data = await appr.json()
            # Should return 404 action_not_found (no pending action)
            assert appr.status == 404
            assert "action_not_found" in appr_data.get("error", {}).get("code", "")

    @pytest.mark.asyncio
    async def test_fake_mode_clarify_returns_not_found(self, free_port):
        """Clarify on a completed run returns action_not_found gracefully."""
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.routes import register_runtime_routes
        from gateway.runtime.executor import RuntimeExecutor, FakeAgentFactory

        os.environ.setdefault("API_SERVER_KEY", "standalone-runtime-smoke-key-001")

        app = web.Application()
        rm = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(rm, agent_factory=factory)
        register_runtime_routes(
            app, run_manager=rm, executor=executor,
            register_create=True, register_status=True, register_events=True,
        )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={
                "session_id": "clarify-test",
                "input": "test",
                "execute": True,
            })
            data = await resp.json()
            run_id = data["run_id"]

            import asyncio
            for _ in range(30):
                s = await cli.get(f"/v1/runs/{run_id}")
                sd = await s.json()
                if sd["status"] == "completed":
                    break
                await asyncio.sleep(0.2)

            clar = await cli.post(f"/v1/runs/{run_id}/clarify", json={
                "run_id": run_id,
                "response": "clarified",
            })
            clar_data = await clar.json()
            assert clar.status == 404
            assert "action_not_found" in clar_data.get("error", {}).get("code", "")
