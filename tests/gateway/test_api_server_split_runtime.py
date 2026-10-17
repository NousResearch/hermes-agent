"""Split-runtime on the API server: client-supplied tools relayed over the Runs API.

Covers flag resolution (adapter extra, env, per-profile config), the capabilities
advertisement, and the HTTP + SSE suspend/resume path end to end.
"""

import asyncio
import json
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


class TestSplitRuntimeConfig:
    def test_split_runtime_defaults_off(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        assert adapter._split_runtime is False
        assert adapter._split_runtime_enabled() is False

    def test_split_runtime_from_extra(self):
        config = PlatformConfig(enabled=True, extra={"split_runtime": True})
        adapter = APIServerAdapter(config)
        assert adapter._split_runtime_enabled() is True

    def test_split_runtime_from_env(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_SPLIT_RUNTIME", "true")
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        assert adapter._split_runtime_enabled() is True

    def test_split_runtime_extra_string_is_coerced(self):
        # Some config loaders hand back strings; "false" must not be truthy.
        off = APIServerAdapter(PlatformConfig(enabled=True, extra={"split_runtime": "false"}))
        on = APIServerAdapter(PlatformConfig(enabled=True, extra={"split_runtime": "true"}))
        assert off._split_runtime_enabled() is False
        assert on._split_runtime_enabled() is True


class TestSplitRuntimeCapabilities:
    @pytest.mark.asyncio
    async def test_capabilities_default_to_server_execution(self):
        app = _create_runs_app(APIServerAdapter(PlatformConfig(enabled=True)))
        async with TestClient(TestServer(app)) as cli:
            data = await (await cli.get("/v1/capabilities")).json()
        assert data["runtime"]["split_runtime"] is False
        assert data["runtime"]["tool_execution"] == "server"
        assert data["features"]["client_tool_execution"] is False

    @pytest.mark.asyncio
    async def test_capabilities_advertise_split_runtime_when_enabled(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"split_runtime": True}),
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["runtime"]["split_runtime"] is True
            assert data["runtime"]["tool_execution"] == "split"
            assert data["features"]["client_tool_execution"] is True
            assert data["endpoints"]["run_tool_result"] == {
                "method": "POST",
                "path": "/v1/runs/{run_id}/tool_result",
            }


# ---------------------------------------------------------------------------
# End-to-end relay over the Runs API (HTTP + SSE)
# ---------------------------------------------------------------------------


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Build an app from the adapter's real route table.

    Uses ``_http_route_table()`` rather than a hand-written route list so the
    test also covers route registration itself (notably the split-runtime
    ``/v1/runs/{run_id}/tool_result`` entry).
    """
    app = web.Application()
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


class _RelayingAgent:
    """Fake agent whose single turn calls one client tool through the relay.

    Stands in for ``_create_agent`` so the HTTP/SSE suspend-resume path can be
    exercised without a model. ``run_conversation`` blocks in
    ``client_tool_gateway.wait_for_result`` exactly as the real agent thread
    does, so the test drives the true suspend → POST → resume sequence.
    """

    def __init__(self, session_key: str, tool_name: str, call_id: str):
        self._client_tool_session_key = session_key
        self._client_tool_names = {tool_name}
        self._tool_name = tool_name
        self._call_id = call_id
        self.session_prompt_tokens = 1
        self.session_completion_tokens = 2
        self.session_total_tokens = 3
        self.relayed_result = None

    def run_conversation(self, user_message=None, conversation_history=None, task_id=None):
        from agent.inline_tool_executors import relay_client_tool

        self.relayed_result = relay_client_tool(
            self, self._tool_name, {"minutes": 5}, self._call_id
        )
        return {"final_response": f"relayed:{self.relayed_result}"}


def _split_runtime_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"split_runtime": True}))


def _client_tool(name: str = "set_timer") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Set a timer on the client device.",
            "parameters": {
                "type": "object",
                "properties": {"minutes": {"type": "integer"}},
            },
        },
    }


async def _await_pending(run_id: str, timeout: float = 5.0):
    """Poll until the run has a pending client-tool entry (or time out)."""
    from tools import client_tool_gateway as ctg

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        entry = ctg.get_pending_for_session(run_id)
        if entry is not None:
            return entry
        await asyncio.sleep(0.05)
    return None


async def _read_sse_event(resp, wanted: str, timeout: float = 5.0):
    """Read the SSE body until an event with ``event == wanted`` arrives.

    Events are framed as ``data: {json}\\n\\n`` with no ``event:`` line — the
    type is a field inside the payload.
    """
    deadline = time.monotonic() + timeout
    buf = b""
    while time.monotonic() < deadline:
        try:
            chunk = await asyncio.wait_for(resp.content.read(512), timeout=1.0)
        except TimeoutError:
            continue
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.splitlines():
                if not line.startswith(b"data: "):
                    continue
                payload = json.loads(line[len(b"data: "):].decode())
                if payload.get("event") == wanted:
                    return payload
    return None


@pytest.fixture(autouse=True)
def _clean_client_tool_registry():
    from tools import client_tool_gateway as ctg

    ctg._entries.clear()
    ctg._session_index.clear()
    ctg._notify_cbs.clear()
    yield
    ctg._entries.clear()
    ctg._session_index.clear()
    ctg._notify_cbs.clear()


class TestSplitRuntimeRelayOverHTTP:
    @pytest.mark.asyncio
    async def test_run_suspends_emits_request_and_resumes_on_tool_result(self):
        """POST /v1/runs → tool_call.request SSE → POST /tool_result → run.completed."""
        adapter = _split_runtime_adapter()
        agents = {}

        def _fake_create_agent(**kwargs):
            agent = _RelayingAgent(
                kwargs.get("client_tool_session_key") or "",
                "set_timer",
                "call_sse_1",
            )
            agents["agent"] = agent
            return agent

        adapter._create_agent = _fake_create_agent
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "set a timer", "tools": [_client_tool()]},
            )
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]

            sse = await cli.get(f"/v1/runs/{run_id}/events")
            assert sse.status == 200

            request_event = await _read_sse_event(sse, "tool_call.request")
            assert request_event is not None, "no tool_call.request event emitted"
            assert request_event["name"] == "set_timer"
            assert request_event["arguments"] == {"minutes": 5}
            call_id = request_event["call_id"]

            # The agent thread is suspended waiting on this exact call.
            assert await _await_pending(run_id) is not None

            done = await cli.post(
                f"/v1/runs/{run_id}/tool_result",
                json={"call_id": call_id, "output": "timer set for 5 minutes"},
            )
            assert done.status == 200

            completed = await _read_sse_event(sse, "run.completed")
            assert completed is not None, "run did not complete after tool_result"
            assert "timer set for 5 minutes" in completed["output"]
            sse.close()

        assert agents["agent"].relayed_result == "timer set for 5 minutes"

    @pytest.mark.asyncio
    async def test_tool_result_from_another_run_is_rejected(self):
        """Cross-run correlation: run B cannot resolve run A's pending call."""
        adapter = _split_runtime_adapter()

        def _fake_create_agent(**kwargs):
            return _RelayingAgent(
                kwargs.get("client_tool_session_key") or "",
                "set_timer",
                "call_shared",
            )

        adapter._create_agent = _fake_create_agent
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            run_ids = []
            for _ in range(2):
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "set a timer", "tools": [_client_tool()]},
                )
                assert resp.status == 202
                run_ids.append((await resp.json())["run_id"])
            run_a, run_b = run_ids

            assert await _await_pending(run_a) is not None
            assert await _await_pending(run_b) is not None

            # Both runs registered the SAME call_id. Posting run A's call_id to
            # run B must not resolve run A (nor run B's own entry with a
            # foreign payload) — it is scoped to (run_id, call_id).
            hijack = await cli.post(
                f"/v1/runs/{run_b}/tool_result",
                json={"call_id": "call_shared", "output": "hijacked"},
            )
            # run_b legitimately owns a call with this id, so this resolves
            # run_b only — run_a must still be suspended.
            assert hijack.status == 200
            assert await _await_pending(run_a) is not None

            # A call_id that belongs to no run at all is a 409, not a
            # cross-run resolution.
            bogus = await cli.post(
                f"/v1/runs/{run_a}/tool_result",
                json={"call_id": "call_does_not_exist", "output": "x"},
            )
            assert bogus.status == 409

            ok = await cli.post(
                f"/v1/runs/{run_a}/tool_result",
                json={"call_id": "call_shared", "output": "for run a"},
            )
            assert ok.status == 200

    @pytest.mark.asyncio
    async def test_stop_releases_a_suspended_client_tool(self):
        """POST /stop must unwind an agent thread blocked on a client tool."""
        from tools import client_tool_gateway as ctg

        adapter = _split_runtime_adapter()

        def _fake_create_agent(**kwargs):
            return _RelayingAgent(
                kwargs.get("client_tool_session_key") or "",
                "set_timer",
                "call_stop_1",
            )

        adapter._create_agent = _fake_create_agent
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "set a timer", "tools": [_client_tool()]},
            )
            run_id = (await resp.json())["run_id"]
            assert await _await_pending(run_id) is not None

            stop = await cli.post(f"/v1/runs/{run_id}/stop")
            assert stop.status in (200, 202)

            # The pending entry is cancelled so the blocked thread unwinds
            # rather than hanging until the relay timeout.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and ctg.has_pending(run_id):
                await asyncio.sleep(0.05)
            assert ctg.has_pending(run_id) is False

    @pytest.mark.asyncio
    async def test_client_tool_named_like_an_agent_tool_is_rejected(self):
        """Names owned by agent-level dispatch cannot be client tools.

        Those handlers run inside the executor and emit their own post hook, so
        a same-named client tool could not be intercepted with consistent
        locality across the sequential and concurrent paths.
        """
        adapter = _split_runtime_adapter()
        adapter._create_agent = lambda **kw: _RelayingAgent("", "noop", "c")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            for reserved in ("memory", "todo_list", "clarify", "delegate_task", "read_terminal"):
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hi", "tools": [_client_tool(reserved)]},
                )
                assert resp.status == 400, f"{reserved} should be rejected"
                body = await resp.json()
                assert "reserved host tool" in json.dumps(body)

    @pytest.mark.asyncio
    async def test_tool_result_on_unknown_run_is_404(self):
        adapter = _split_runtime_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_nonexistent/tool_result",
                json={"call_id": "c1", "output": "x"},
            )
            assert resp.status == 404


class TestSplitRuntimePerProfileResolution:
    """split_runtime resolution must be per-profile under multiplexing."""

    def test_config_fallback_is_cached_per_profile(self, monkeypatch):
        from gateway.platforms import api_server as mod

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        assert adapter._split_runtime is False

        # Simulate two profiles whose top-level api_server blocks disagree.
        per_profile = {"assistant": True, "study": False, "": False}
        seen = []

        def _fake_load_config():
            profile = mod._api_request_profile.get() or ""
            seen.append(profile)
            return {"api_server": {"split_runtime": per_profile[profile]}}

        monkeypatch.setattr("hermes_cli.config.load_config", _fake_load_config)

        for profile, expected in (("assistant", True), ("study", False), ("", False)):
            token = mod._api_request_profile.set(profile or None)
            try:
                assert adapter._split_runtime_enabled() is expected
                # Second call for the same profile is served from the cache.
                assert adapter._split_runtime_enabled() is expected
            finally:
                mod._api_request_profile.reset(token)

        # One config read per distinct profile, not per call.
        assert sorted(seen) == ["", "assistant", "study"]

    def test_adapter_level_flag_short_circuits_config(self, monkeypatch):
        def _boom():
            raise AssertionError("config must not be consulted when extra sets the flag")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"split_runtime": True})
        )
        assert adapter._split_runtime_enabled() is True

