"""Integration coverage for configured API session-key aliases."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms import api_server as api_server_module
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionSource, build_session_key
from hermes_state import SessionDB


@pytest.fixture
def alias_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        """gateway:
  session_key_aliases:
    mobile-telegram:
      platform: telegram
      chat_id: configured-chat
      chat_type: dm
      thread_id: configured-thread
      user_id: configured-user
    desktop-discord:
      platform: discord
      chat_id: discord-channel
      chat_type: group
      thread_id: discord-thread
    recursive:
      platform: webhook
      chat_id: not-native
    unsupported-profile:
      platform: telegram
      chat_id: configured-chat
      profile: work
    unsupported-scope-a:
      platform: slack
      chat_id: C123
      chat_type: group
      scope_id: T-A
    unsupported-scope-b:
      platform: slack
      chat_id: C123
      chat_type: group
      scope_id: T-B
""",
        encoding="utf-8",
    )
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-one", "api_server")
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "sk-test"})
    )
    adapter._session_db = db
    try:
        yield adapter
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/api/sessions/{session_id}/chat", adapter._handle_session_chat
    )
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream",
        adapter._handle_session_chat_stream,
    )
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


def _expected_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="configured-chat",
        chat_type="dm",
        thread_id="configured-thread",
        user_id="configured-user",
    )


def _patch_agent_runtime(monkeypatch, captured):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "agent-session"

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_conversation(self, **_kwargs):
            from gateway.session_context import get_session_env
            from tools.approval import get_current_session_key

            captured["runtime_session_key"] = get_session_env("HERMES_SESSION_KEY")
            captured["approval_session_key"] = get_current_session_key()
            captured["platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            captured["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID")
            captured["thread_id"] = get_session_env("HERMES_SESSION_THREAD_ID")
            captured["user_id"] = get_session_env("HERMES_SESSION_USER_ID")
            stream_delta_callback = captured.get("stream_delta_callback")
            if stream_delta_callback is not None:
                stream_delta_callback("ok")
            return {"final_response": "ok", "completed": True}

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "«redacted:sk-…»",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "test/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda model="": {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None)
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)


@pytest.mark.asyncio
async def test_configured_alias_reaches_all_run_agent_http_surfaces(alias_adapter):
    expected_source = _expected_source()
    expected_key = build_session_key(expected_source)
    run_agent = AsyncMock(
        return_value=(
            {"final_response": "ok", "completed": True, "session_id": "session-one"},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    )
    alias_adapter._run_agent = run_agent
    headers = {
        "Authorization": "Bearer sk-test",
        "X-Hermes-Session-Key": "mobile-telegram",
    }
    cases = [
        ("/api/sessions/session-one/chat", {"message": "hello"}),
        ("/api/sessions/session-one/chat/stream", {"message": "hello"}),
        (
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "platform": "discord",
                "chat_id": "attacker-selected-chat",
                "profile": "attacker-selected-profile",
            },
        ),
        ("/v1/responses", {"model": "test", "input": "hello"}),
    ]

    async with TestClient(TestServer(_create_app(alias_adapter))) as client:
        for path, payload in cases:
            run_agent.reset_mock()
            response = await client.post(path, json=payload, headers=headers)
            assert response.status == 200
            await response.read()
            run_agent.assert_awaited_once()
            assert run_agent.await_args is not None
            kwargs = run_agent.await_args.kwargs
            assert kwargs["gateway_session_key"] == expected_key
            assert kwargs["session_source"] == expected_source


@pytest.mark.asyncio
async def test_chat_idempotency_is_partitioned_by_resolved_alias_identity(
    alias_adapter, monkeypatch
):
    monkeypatch.setattr(
        api_server_module,
        "_idem_cache",
        api_server_module._IdempotencyCache(),
    )
    calls = []

    async def run_agent(**kwargs):
        source = kwargs["session_source"]
        calls.append(kwargs["gateway_session_key"])
        return (
            {
                "final_response": source.platform.value,
                "completed": True,
                "session_id": "session-one",
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    alias_adapter._run_agent = AsyncMock(side_effect=run_agent)
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
    }
    base_headers = {
        "Authorization": "Bearer sk-test",
        "Idempotency-Key": "shared-idempotency-key",
    }

    async with TestClient(TestServer(_create_app(alias_adapter))) as client:
        mobile = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={**base_headers, "X-Hermes-Session-Key": "mobile-telegram"},
        )
        desktop = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={**base_headers, "X-Hermes-Session-Key": "desktop-discord"},
        )
        mobile_replay = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={**base_headers, "X-Hermes-Session-Key": "mobile-telegram"},
        )

        assert mobile.status == desktop.status == mobile_replay.status == 200
        mobile_data = await mobile.json()
        desktop_data = await desktop.json()
        replay_data = await mobile_replay.json()

    assert mobile_data["choices"][0]["message"]["content"] == "telegram"
    assert desktop_data["choices"][0]["message"]["content"] == "discord"
    assert replay_data["choices"][0]["message"]["content"] == "telegram"
    assert calls == [
        build_session_key(_expected_source()),
        build_session_key(
            SessionSource(
                platform=Platform.DISCORD,
                chat_id="discord-channel",
                chat_type="group",
                thread_id="discord-thread",
            )
        ),
    ]


@pytest.mark.asyncio
async def test_responses_idempotency_is_partitioned_by_resolved_alias_identity(
    alias_adapter, monkeypatch
):
    monkeypatch.setattr(
        api_server_module,
        "_idem_cache",
        api_server_module._IdempotencyCache(),
    )
    calls = []

    async def run_agent(**kwargs):
        source = kwargs["session_source"]
        calls.append(kwargs["gateway_session_key"])
        text = source.platform.value
        return (
            {
                "final_response": text,
                "completed": True,
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": text},
                ],
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    alias_adapter._run_agent = AsyncMock(side_effect=run_agent)
    payload = {"model": "test", "input": "hello"}
    base_headers = {
        "Authorization": "Bearer sk-test",
        "Idempotency-Key": "shared-idempotency-key",
    }

    async with TestClient(TestServer(_create_app(alias_adapter))) as client:
        mobile = await client.post(
            "/v1/responses",
            json=payload,
            headers={**base_headers, "X-Hermes-Session-Key": "mobile-telegram"},
        )
        desktop = await client.post(
            "/v1/responses",
            json=payload,
            headers={**base_headers, "X-Hermes-Session-Key": "desktop-discord"},
        )
        mobile_replay = await client.post(
            "/v1/responses",
            json=payload,
            headers={**base_headers, "X-Hermes-Session-Key": "mobile-telegram"},
        )

        assert mobile.status == desktop.status == mobile_replay.status == 200
        mobile_data = await mobile.json()
        desktop_data = await desktop.json()
        replay_data = await mobile_replay.json()

    def output_text(data):
        return data["output"][0]["content"][0]["text"]

    assert output_text(mobile_data) == "telegram"
    assert output_text(desktop_data) == "discord"
    assert output_text(replay_data) == "telegram"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_configured_alias_reaches_runs_agent_construction(alias_adapter):
    expected_source = _expected_source()
    expected_key = build_session_key(expected_source)

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "run-session"

        def run_conversation(self, **_kwargs):
            return {"final_response": "ok", "completed": True}

    create_agent = MagicMock(return_value=FakeAgent())
    alias_adapter._create_agent = create_agent
    headers = {
        "Authorization": "Bearer sk-test",
        "X-Hermes-Session-Key": "mobile-telegram",
    }

    async with TestClient(TestServer(_create_app(alias_adapter))) as client:
        response = await client.post(
            "/v1/runs", json={"input": "hello"}, headers=headers
        )
        assert response.status == 202
        await response.read()
        tasks = list(alias_adapter._background_tasks)
        if tasks:
            await asyncio.gather(*tasks)

    create_agent.assert_called_once()
    kwargs = create_agent.call_args.kwargs
    assert kwargs["gateway_session_key"] == expected_key
    assert kwargs["session_source"] == expected_source


@pytest.mark.asyncio
async def test_runs_alias_injects_native_context_and_preserves_instructions(
    alias_adapter, monkeypatch
):
    captured = {}
    _patch_agent_runtime(monkeypatch, captured)
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools", lambda _config, _platform: set()
    )
    headers = {
        "Authorization": "Bearer sk-test",
        "X-Hermes-Session-Key": "mobile-telegram",
    }

    async with TestClient(TestServer(_create_app(alias_adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "instructions": "Keep this caller instruction."},
            headers=headers,
        )
        assert response.status == 202
        await response.read()
        tasks = list(alias_adapter._background_tasks)
        if tasks:
            await asyncio.gather(*tasks)

    prompt = captured["ephemeral_system_prompt"]
    assert "## Current Session Context" in prompt
    assert "Telegram" in prompt
    assert "configured-chat" in prompt
    assert "configured-thread" in prompt
    assert prompt.endswith("Keep this caller instruction.")
    canonical_key = build_session_key(_expected_source())
    assert captured["runtime_session_key"] == canonical_key
    assert captured["approval_session_key"]
    assert captured["approval_session_key"] != canonical_key


@pytest.mark.asyncio
async def test_named_profile_alias_uses_scoped_config_and_native_key_policy(
    alias_adapter, monkeypatch
):
    default_config = GatewayConfig(
        multiplex_profiles=True,
        session_key_aliases={
            "shared-alias": {
                "platform": "telegram",
                "chat_id": "default-chat",
                "chat_type": "dm",
            }
        },
    )
    coder_config = GatewayConfig(
        multiplex_profiles=True,
        group_sessions_per_user=False,
        session_key_aliases={
            "shared-alias": {
                "platform": "telegram",
                "chat_id": "coder-group",
                "chat_type": "group",
                "user_id": "coder-user",
            }
        },
    )
    monkeypatch.setattr(
        api_server_module,
        "load_gateway_config",
        lambda: (
            coder_config
            if api_server_module._api_request_profile.get() == "coder"
            else default_config
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", None), ("coder", None)],
    )
    monkeypatch.setattr(
        alias_adapter, "_profile_scope", lambda _profile: nullcontext()
    )
    alias_adapter.gateway_runner = SimpleNamespace(config=default_config)
    alias_adapter._run_agent = AsyncMock(
        return_value=({"final_response": "ok", "completed": True}, {})
    )

    app = web.Application(
        middlewares=[alias_adapter._make_profile_prefix_middleware()]
    )
    app.router.add_post(
        "/p/{profile}/v1/responses", alias_adapter._handle_responses
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/p/coder/v1/responses",
            headers={
                "Authorization": "Bearer sk-test",
                "X-Hermes-Session-Key": "shared-alias",
            },
            json={"model": "hermes", "input": "hello"},
        )

    assert response.status == 200
    assert alias_adapter._run_agent.await_args is not None
    kwargs = alias_adapter._run_agent.await_args.kwargs
    expected_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="coder-group",
        chat_type="group",
        user_id="coder-user",
        profile="coder",
    )
    assert kwargs["session_source"] == expected_source
    assert kwargs["gateway_session_key"] == build_session_key(
        expected_source,
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
        profile="coder",
    )


@pytest.mark.asyncio
async def test_idempotency_is_partitioned_by_active_request_profile(
    alias_adapter, monkeypatch
):
    default_config = GatewayConfig(
        multiplex_profiles=True,
        session_key_aliases={
            "shared-alias": {
                "platform": "telegram",
                "chat_id": "default-chat",
                "chat_type": "dm",
            }
        },
    )
    coder_config = GatewayConfig(
        multiplex_profiles=True,
        session_key_aliases={
            "shared-alias": {
                "platform": "telegram",
                "chat_id": "coder-chat",
                "chat_type": "dm",
            }
        },
    )
    monkeypatch.setattr(
        api_server_module,
        "load_gateway_config",
        lambda: (
            coder_config
            if api_server_module._api_request_profile.get() == "coder"
            else default_config
        ),
    )
    monkeypatch.setattr(
        api_server_module,
        "_idem_cache",
        api_server_module._IdempotencyCache(),
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", None), ("coder", None)],
    )
    monkeypatch.setattr(alias_adapter, "_profile_scope", lambda _profile: nullcontext())
    alias_adapter.gateway_runner = SimpleNamespace(config=default_config)
    calls = []

    async def run_agent(**kwargs):
        calls.append(kwargs["gateway_session_key"])
        return (
            {"final_response": kwargs["session_source"].chat_id, "completed": True},
            {},
        )

    alias_adapter._run_agent = AsyncMock(side_effect=run_agent)
    app = web.Application(middlewares=[alias_adapter._make_profile_prefix_middleware()])
    app.router.add_post("/p/{profile}/v1/responses", alias_adapter._handle_responses)
    headers = {
        "Authorization": "Bearer sk-test",
        "X-Hermes-Session-Key": "shared-alias",
        "Idempotency-Key": "same-profile-idempotency-key",
    }
    payload = {"model": "hermes", "input": "hello"}

    async with TestClient(TestServer(app)) as client:
        default = await client.post(
            "/p/default/v1/responses", headers=headers, json=payload
        )
        coder = await client.post(
            "/p/coder/v1/responses", headers=headers, json=payload
        )
        default_replay = await client.post(
            "/p/default/v1/responses", headers=headers, json=payload
        )

        assert default.status == coder.status == default_replay.status == 200

    assert len(calls) == 2
    assert calls[0] != calls[1]


@pytest.mark.asyncio
async def test_discord_alias_uses_canonical_source(alias_adapter):
    run_agent = AsyncMock(
        return_value=(
            {"final_response": "ok", "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )
    alias_adapter._run_agent = run_agent
    discord_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="discord-channel",
        chat_type="group",
        thread_id="discord-thread",
    )

    async with TestClient(TestServer(_create_app(alias_adapter))) as client:
        response = await client.post(
            "/v1/responses",
            json={"input": "hello"},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Hermes-Session-Key": "desktop-discord",
            },
        )
        assert response.status == 200
        await response.read()

    assert run_agent.await_args is not None
    kwargs = run_agent.await_args.kwargs
    assert kwargs["session_source"] == discord_source
    assert kwargs["gateway_session_key"] == build_session_key(discord_source)


@pytest.mark.asyncio
async def test_profile_and_scope_aliases_fail_closed(alias_adapter):
    alias_adapter._run_agent = AsyncMock()
    headers = {"Authorization": "Bearer sk-test"}

    async with TestClient(TestServer(_create_app(alias_adapter))) as client:
        for alias in ("unsupported-profile", "unsupported-scope-a", "unsupported-scope-b"):
            response = await client.post(
                "/v1/responses",
                json={"input": "hello"},
                headers={**headers, "X-Hermes-Session-Key": alias},
            )
            assert response.status == 400
            payload = await response.json()
            assert payload["error"]["code"] == "invalid_session_alias"

    alias_adapter._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_key_stays_api_server_and_invalid_alias_fails_closed(alias_adapter):
    run_agent = AsyncMock(
        return_value=(
            {"final_response": "ok", "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )
    alias_adapter._run_agent = run_agent
    app = _create_app(alias_adapter)

    async with TestClient(TestServer(app)) as client:
        unknown = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Hermes-Session-Key": "unconfigured-client-key",
            },
        )
        assert unknown.status == 200
        assert run_agent.await_args is not None
        kwargs = run_agent.await_args.kwargs
        assert kwargs["gateway_session_key"] == "unconfigured-client-key"
        assert kwargs["session_source"] is None

        invalid = await client.post(
            "/v1/responses",
            json={"input": "hello"},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Hermes-Session-Key": "recursive",
            },
        )
        assert invalid.status == 400
        payload = await invalid.json()
        assert payload["error"]["code"] == "invalid_session_alias"


@pytest.mark.asyncio
async def test_configured_alias_is_rejected_when_api_auth_is_disabled(
    alias_adapter,
):
    unauthenticated_adapter = APIServerAdapter(PlatformConfig(enabled=True))
    unauthenticated_adapter._session_db = alias_adapter._session_db
    run_agent = AsyncMock()
    setattr(unauthenticated_adapter, "_run_agent", run_agent)

    async with TestClient(
        TestServer(_create_app(unauthenticated_adapter))
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"X-Hermes-Session-Key": "mobile-telegram"},
        )

    assert response.status == 403
    run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_routes_bind_native_identity_and_clear_it_afterward(
    alias_adapter, monkeypatch
):
    headers = {
        "Authorization": "Bearer sk-test",
        "X-Hermes-Session-Key": "mobile-telegram",
    }
    expected_key = build_session_key(_expected_source())
    chat_capture = {}
    responses_capture = {}
    unconfigured_capture = {}

    async with TestClient(TestServer(_create_app(alias_adapter))) as client:
        _patch_agent_runtime(monkeypatch, chat_capture)
        chat = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers=headers,
        )
        assert chat.status == 200
        assert "[DONE]" in await chat.text()

        _patch_agent_runtime(monkeypatch, responses_capture)
        responses = await client.post(
            "/v1/responses",
            json={"model": "test", "input": "hello", "stream": True},
            headers=headers,
        )
        assert responses.status == 200
        assert "response.completed" in await responses.text()

        _patch_agent_runtime(monkeypatch, unconfigured_capture)
        unconfigured = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers={
                "Authorization": "Bearer sk-test",
                "X-Hermes-Session-Key": "unconfigured-client-key",
            },
        )
        assert unconfigured.status == 200
        assert "[DONE]" in await unconfigured.text()

    assert chat_capture["runtime_session_key"] == expected_key
    assert responses_capture["runtime_session_key"] == expected_key
    assert chat_capture["platform"] == responses_capture["platform"] == "telegram"
    assert unconfigured_capture["runtime_session_key"] == "unconfigured-client-key"
    assert unconfigured_capture["platform"] == "api_server"
    assert unconfigured_capture["chat_id"].startswith("api-")
    assert unconfigured_capture["chat_id"] != "configured-chat"
    assert unconfigured_capture["thread_id"] == ""
    assert unconfigured_capture["user_id"] == ""


@pytest.mark.asyncio
async def test_native_alias_binds_platform_toolset_and_context(alias_adapter, monkeypatch):
    expected_source = _expected_source()
    expected_key = build_session_key(expected_source)
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "agent-session"

        def run_conversation(self, **_kwargs):
            from gateway.session_context import get_session_env

            observed["platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            observed["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID")
            observed["thread_id"] = get_session_env("HERMES_SESSION_THREAD_ID")
            observed["user_id"] = get_session_env("HERMES_SESSION_USER_ID")
            observed["profile"] = get_session_env("HERMES_SESSION_PROFILE")
            observed["session_key"] = get_session_env("HERMES_SESSION_KEY")
            return {"final_response": "ok", "completed": True}

    def fake_create_agent(**kwargs):
        observed["create_kwargs"] = kwargs
        return FakeAgent()

    monkeypatch.setattr(alias_adapter, "_create_agent", fake_create_agent)
    result, _usage = await alias_adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="agent-session",
        gateway_session_key=expected_key,
        session_source=expected_source,
    )

    assert result["final_response"] == "ok"
    assert observed["platform"] == "telegram"
    assert observed["chat_id"] == "configured-chat"
    assert observed["thread_id"] == "configured-thread"
    assert observed["user_id"] == "configured-user"
    assert observed["profile"] == ""
    assert observed["session_key"] == expected_key
    assert observed["create_kwargs"]["session_source"] == expected_source


def test_native_alias_selects_native_agent_platform_toolset_and_prompt(
    alias_adapter, monkeypatch
):
    captured = {}
    _patch_agent_runtime(monkeypatch, captured)
    selected_platforms = []

    def fake_platform_tools(_config, platform):
        selected_platforms.append(platform)
        return {f"toolset-{platform}"}

    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools", fake_platform_tools
    )
    source = _expected_source()
    canonical_key = build_session_key(source, profile=source.profile)

    alias_adapter._create_agent(
        ephemeral_system_prompt="Keep this caller instruction.",
        gateway_session_key=canonical_key,
        session_source=source,
    )

    assert selected_platforms == ["telegram"]
    assert captured["enabled_toolsets"] == ["toolset-telegram"]
    assert captured["platform"] == "telegram"
    assert captured["gateway_session_key"] == canonical_key
    prompt = captured["ephemeral_system_prompt"]
    assert "## Current Session Context" in prompt
    assert "configured-chat" in prompt
    assert "configured-thread" in prompt
    assert prompt.endswith("Keep this caller instruction.")
