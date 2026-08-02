"""Focused tests for API server session-control endpoints."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


@pytest.mark.asyncio
async def test_capabilities_advertises_session_control_surface(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    features = data["features"]
    assert features["session_resources"] is True
    assert features["session_chat"] is True
    assert features["session_chat_streaming"] is True
    assert features["session_fork"] is True
    assert features["admin_config_rw"] is False
    assert features["memory_write_api"] is False
    assert features["skills_api"] is True
    assert features["realtime_voice"] is False
    assert data["endpoints"]["sessions"] == {"method": "GET", "path": "/api/sessions"}
    assert data["endpoints"]["session_chat_stream"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/chat/stream",
    }


@pytest.mark.asyncio
async def test_run_agent_binds_api_session_context_for_tool_env(adapter, monkeypatch):
    """API-server request sessions should reach tools and terminal subprocess env."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-session")
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.session_context import get_session_env
            from tools.environments.local import _make_run_env

            observed["task_id"] = task_id
            observed["context_session_id"] = get_session_env("HERMES_SESSION_ID")
            observed["context_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            observed["context_session_key"] = get_session_env("HERMES_SESSION_KEY")
            observed["child_session_id"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        gateway_session_key="request-key",
    )

    assert result["session_id"] == "request-session"
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert "runtime" not in usage
    assert observed == {
        "task_id": "request-session",
        "context_session_id": "request-session",
        "context_platform": "api_server",
        "context_session_key": "request-key",
        "child_session_id": "request-session",
    }


@pytest.mark.asyncio
async def test_session_chat_stream_run_completed_carries_turn_transcript(adapter, session_db):
    """run.completed must include the full interleaved turn transcript so a
    client that lost intermediate (pre-tool-call) assistant text from the live
    delta stream can reconcile without a separate /messages fetch. Refs #34703.
    """
    import json as _json

    session_id = session_db.create_session("transcript-session", "api_server")

    async def fake_run(**kwargs):
        # Stream the intermediate planning text the way a real turn would.
        kwargs["stream_delta_callback"]("Let me search for that:")
        kwargs["stream_delta_callback"]("Here is the summary.")
        result = {
            "final_response": "Here is the summary.",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "search then summarize"},
                {
                    "role": "assistant",
                    "content": "Let me search for that:",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "results", "tool_call_id": "call_1", "tool_name": "web_search"},
                {"role": "assistant", "content": "Here is the summary."},
            ],
        }
        return result, {"total_tokens": 6}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "search then summarize"},
            )
            assert resp.status == 200
            body = await resp.text()

    # Pull the run.completed event payload out of the SSE body.
    run_completed_payload = None
    for block in body.split("\n\n"):
        if "event: run.completed" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    run_completed_payload = _json.loads(line[len("data: "):])
            break
    assert run_completed_payload is not None, body
    messages = run_completed_payload.get("messages")
    assert isinstance(messages, list) and messages, run_completed_payload

    # The colon-ended intermediate text that preceded the tool call must be present.
    contents = [m.get("content") for m in messages]
    assert "Let me search for that:" in contents
    assert "Here is the summary." in contents
    # No prior-turn user message should leak into the per-turn slice.
    assert all(m.get("role") in ("assistant", "tool") for m in messages)
    # The tool call is preserved alongside the intermediate text.
    assert any(m.get("tool_calls") for m in messages)


# ---------------------------------------------------------------------------
# Session-persisted model threading + provider-auth failure surfacing
# (salvaged from PR #57947 by @FvanW and PR #59941 by @kaishi00)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_chat_resolves_stored_model_route_alias(session_db, monkeypatch):
    """A session-persisted model that matches a model_routes alias must go
    through the route path (so route provider/credentials apply) and NOT be
    passed as a raw session_model (idea from PR #59941 by @kaishi00)."""
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"model_routes": {"alias": {"model": "route/model", "provider": "openrouter"}}},
        )
    )
    adapter._session_db = session_db
    session_id = session_db.create_session("route-pinned-session", "api_server", model="alias")

    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": session_id}, {"total_tokens": 1}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hi"},
            )
            assert resp.status == 200

    _, kwargs = mock_run.call_args
    assert kwargs["route"] == {"model": "route/model", "provider": "openrouter"}
    assert kwargs["session_model"] is None


def _register_session_model_route(app, adapter):
    app.router.add_post("/api/sessions/{session_id}/model", adapter._handle_session_model_lock)


def _patch_api_server_runtime(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "sk-global",
            "base_url": "https://openrouter.example/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda model="": {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {
            "provider": provider,
            "api_key": f"sk-{provider}",
            "base_url": f"https://{provider}.example/v1",
            "api_mode": "chat_completions",
        },
    )


@pytest.mark.asyncio
async def test_create_session_respects_browser_source_and_model_lock(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions",
            json={
                "id": "browser-lock-session",
                "source": "hermes_browser",
                "provider": "nous",
                "model": "x-ai/grok-4.5",
                "require_model_lock": True,
                "title": "Browser lock",
                "system_prompt": "browser prompt",
            },
        )
        assert resp.status == 201, await resp.text()
        payload = await resp.json()

    assert payload["session"]["source"] == "hermes_browser"
    assert payload["session"]["model"] == "x-ai/grok-4.5"
    row = session_db.get_session("browser-lock-session")
    assert row["source"] == "hermes_browser"
    assert row["model"] == "x-ai/grok-4.5"
    import json as _json
    model_config = row.get("model_config")
    if isinstance(model_config, str):
        model_config = _json.loads(model_config)
    assert model_config["browser_model_lock"]["provider"] == "nous"
    assert model_config["browser_model_lock"]["model"] == "x-ai/grok-4.5"
    assert model_config["browser_model_lock"]["confirmed"] is True


@pytest.mark.asyncio
async def test_session_model_lock_endpoint_then_chat_reuses_persisted_lock_and_provider_credentials(
    adapter,
    session_db,
    monkeypatch,
):
    session_id = session_db.create_session(
        "endpoint-lock-chat",
        "api_server",
        model="gpt-5.5",
        system_prompt="Conversation started:\nModel: gpt-5.5\nProvider: openai-codex\n",
    )
    captured = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = kwargs["session_id"]
            self.provider = kwargs.get("provider") or ""
            self.model = kwargs.get("model") or ""

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "locked", "session_id": self.session_id}

    _patch_api_server_runtime(monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        adapter,
        "_session_model_override_for",
        lambda *_: {
            "model": "session/override-model",
            "provider": "openai-codex",
            "api_key": "sk-session-override",
            "base_url": "https://override.example/v1",
            "api_mode": "codex_responses",
        },
    )

    app = _create_session_app(adapter)
    _register_session_model_route(app, adapter)
    with patch.object(adapter, "_resolve_route", return_value=None):
        async with TestClient(TestServer(app)) as cli:
            lock_resp = await cli.post(
                f"/api/sessions/{session_id}/model",
                json={
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "require_model_lock": True,
                },
            )
            assert lock_resp.status == 200, await lock_resp.text()

            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "use the stored lock"},
            )
            assert resp.status == 200, await resp.text()
            payload = await resp.json()

    assert captured["provider"] == "nous"
    assert captured["model"] == "x-ai/grok-4.5"
    assert captured["api_key"] == "sk-nous"
    assert captured["base_url"] == "https://nous.example/v1"
    assert payload["runtime"]["provider"] == "nous"
    assert payload["runtime"]["model"] == "x-ai/grok-4.5"
    assert payload["runtime"]["requested"] == {
        "provider": "nous",
        "model": "x-ai/grok-4.5",
    }
    assert payload["runtime"]["route_source"] == "session_model_lock"


@pytest.mark.asyncio
async def test_session_model_lock_endpoint_then_chat_stream_reuses_persisted_lock(
    adapter,
    session_db,
):
    session_id = session_db.create_session("endpoint-lock-stream", "api_server")
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        kwargs["stream_delta_callback"]("hi")
        return (
            {
                "final_response": "hi",
                "session_id": session_id,
                "runtime": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "requested": {"provider": "nous", "model": "x-ai/grok-4.5"},
                    "route_source": "session_model_lock",
                },
            },
            {
                "total_tokens": 1,
                "runtime": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "requested": {"provider": "nous", "model": "x-ai/grok-4.5"},
                    "route_source": "session_model_lock",
                },
            },
        )

    app = _create_session_app(adapter)
    _register_session_model_route(app, adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(
        adapter,
        "_run_agent",
        side_effect=fake_run,
    ):
        async with TestClient(TestServer(app)) as cli:
            lock_resp = await cli.post(
                f"/api/sessions/{session_id}/model",
                json={
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "require_model_lock": True,
                },
            )
            assert lock_resp.status == 200, await lock_resp.text()

            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "stream with stored lock"},
            )
            assert resp.status == 200, await resp.text()
            body = await resp.text()

    assert captured["route"] == {"provider": "nous", "model": "x-ai/grok-4.5"}
    assert captured["requested_runtime"]["provider"] == "nous"
    assert captured["requested_runtime"]["model"] == "x-ai/grok-4.5"
    assert captured["route_source"] == "session_model_lock"
    assert "x-ai/grok-4.5" in body


@pytest.mark.asyncio
async def test_run_agent_reports_actual_agent_runtime_not_requested_metadata(adapter, monkeypatch):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self):
            self.session_id = "runtime-session"
            self.provider = "actual-provider"
            self.model = "actual-model"
            self._hermes_api_runtime = {
                "provider": "requested-provider",
                "model": "requested-model",
                "route_source": "raw_request",
            }

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "ok", "session_id": self.session_id}

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: FakeAgent())

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="runtime-session",
        route={"provider": "requested-provider", "model": "requested-model"},
        requested_runtime={
            "provider": "requested-provider",
            "model": "requested-model",
        },
        route_source="session_model_lock",
    )

    assert result["runtime"]["provider"] == "actual-provider"
    assert result["runtime"]["model"] == "actual-model"
    assert result["runtime"]["requested"] == {
        "provider": "requested-provider",
        "model": "requested-model",
    }
    assert usage["runtime"]["provider"] == "actual-provider"
    assert usage["runtime"]["model"] == "actual-model"


@pytest.mark.asyncio
async def test_confirmed_runtime_lock_rejects_actual_runtime_mismatch(adapter, monkeypatch):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "mismatch-session"
        provider = "fallback-provider"
        model = "fallback-model"

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "wrong runtime", "session_id": self.session_id}

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: FakeAgent())

    with pytest.raises(RuntimeError, match="confirmed model lock runtime mismatch"):
        await adapter._run_agent(
            user_message="hello",
            conversation_history=[],
            session_id="mismatch-session",
            route={"provider": "nous", "model": "x-ai/grok-4.5"},
            requested_runtime={"provider": "nous", "model": "x-ai/grok-4.5"},
            route_source="session_model_lock",
            confirmed_runtime_lock=True,
        )


def test_confirmed_runtime_lock_disables_global_fallback_model(adapter, monkeypatch):
    _patch_api_server_runtime(monkeypatch)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: "openrouter/fallback-model"),
    )
    captured = {}

    class FakeAgent:
        provider = "nous"
        model = "x-ai/grok-4.5"

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    adapter._create_agent(
        session_id="locked-session",
        route={"provider": "nous", "model": "x-ai/grok-4.5"},
        confirmed_runtime_lock=True,
    )

    assert captured["fallback_model"] is None


@pytest.mark.asyncio
async def test_unconfirmed_request_does_not_replace_confirmed_session_lock(adapter, session_db):
    session_id = session_db.create_session("one-off-override", "api_server")
    session_db.update_session_runtime_lock(
        session_id,
        provider="nous",
        model="x-ai/grok-4.5",
        route_source="raw_request",
        confirmed=True,
    )
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": "ok",
                "session_id": session_id,
                "runtime": {"provider": "openrouter", "model": "anthropic/claude-sonnet"},
            },
            {"total_tokens": 1},
        )
    )
    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(
        adapter,
        "_run_agent",
        mock_run,
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "one turn only",
                    "provider": "openrouter",
                    "model": "anthropic/claude-sonnet",
                },
            )
            assert resp.status == 200, await resp.text()

    import json as _json

    row = session_db.get_session(session_id)
    config = row["model_config"]
    if isinstance(config, str):
        config = _json.loads(config)
    assert config["browser_model_lock"]["provider"] == "nous"
    assert config["browser_model_lock"]["model"] == "x-ai/grok-4.5"
    assert config["browser_model_lock"]["confirmed"] is True


@pytest.mark.asyncio
async def test_require_model_lock_hard_fails_when_global_default_would_be_used(adapter, session_db, monkeypatch):
    session_id = session_db.create_session("lock-fail-session", "api_server")
    monkeypatch.setattr(adapter, "_model_name", "gpt-5.5")
    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
        async with TestClient(TestServer(app)) as cli:
            # empty model + require_model_lock must not silently fall through
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "hello",
                    "provider": "nous",
                    "model": "",
                    "require_model_lock": True,
                },
            )
            assert resp.status in (400, 409), await resp.text()
            body = await resp.json()
            assert body["error"]["code"] in {"model_lock_unavailable", "invalid_model_lock", "missing_model"}
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_list_sessions_backfills_out_of_page_pinned_row(adapter, session_db):
    """A pinned conversation aging past the LIMIT page must still surface
    (list_sessions_rich include_pinned back-fill), or the desktop Pinned
    section silently empties as new sessions push it off the page."""
    t0 = time.time()
    for i in range(5):
        session_db.create_session(f"page-{i}", "api_server")
        session_db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t0 - (5 - i) * 1000, f"page-{i}"),
        )
    session_db._conn.commit()
    # Pin the oldest row (page-0), which a limit=2 page ordered by last
    # active would otherwise exclude entirely.
    session_db.set_session_pinned("page-0", True)

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/sessions?limit=2")
        assert resp.status == 200, await resp.text()
        data = await resp.json()
    ids = [s["id"] for s in data["data"]]
    assert len(ids) >= 3, ids
    assert "page-0" in ids
    pinned = next(s for s in data["data"] if s["id"] == "page-0")
    assert pinned["pinned"] is True


@pytest.mark.asyncio
async def test_patch_pinned_archived_round_trips_through_real_db(adapter, session_db):
    """PATCH true then false against a real SessionDB: the SQLite row stores
    0/1, so the response must coerce back to booleans or the desktop
    `typeof row.pinned === 'boolean'` guard rejects the server state."""
    session_db.create_session("roundtrip", "api_server")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.patch(
            "/api/sessions/roundtrip",
            json={"pinned": True, "archived": True},
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()
        assert body["session"]["pinned"] is True
        assert body["session"]["archived"] is True

        resp = await cli.patch(
            "/api/sessions/roundtrip",
            json={"pinned": False, "archived": False},
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()
        assert body["session"]["pinned"] is False
        assert body["session"]["archived"] is False

    row = session_db.get_session("roundtrip")
    assert row["pinned"] == 0
    assert row["archived"] == 0


@pytest.mark.asyncio
async def test_patch_rejects_non_boolean_against_real_db(adapter, session_db):
    session_db.create_session("badtype", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.patch(
            "/api/sessions/badtype",
            json={"pinned": "false"},
        )
        assert resp.status == 400, await resp.text()
        body = await resp.json()
        assert body["error"]["code"] == "invalid_field_type"
    row = session_db.get_session("badtype")
    assert row["pinned"] == 0


