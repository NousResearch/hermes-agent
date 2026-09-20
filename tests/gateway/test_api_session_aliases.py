"""HTTP regression coverage for configured API session-key aliases."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.session import SessionSource, build_session_key
from hermes_state import SessionDB


def _app(adapter):
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


def _adapter_with_native_history(tmp_path, history, key=None):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "session-key-test"}))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("native-transcript", source="discord", session_key=key)
    db.create_session("unrelated-transcript", source="api_server")
    for message in history:
        db.append_message("native-transcript", **message)
    db.append_message("unrelated-transcript", role="user", content="unrelated history")
    adapter._session_db = db
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("group_sessions_per_user", [True, False])
@pytest.mark.parametrize("thread_sessions_per_user", [True, False])
@pytest.mark.parametrize("thread_id", [None, "thread-456"])
@pytest.mark.parametrize("stream", [False, True])
async def test_configured_alias_executes_with_native_thread_identity_and_history(
    tmp_path, monkeypatch, group_sessions_per_user, thread_sessions_per_user, thread_id, stream
):
    """The HTTP consumer must receive the exact native key and persisted transcript."""
    native = SessionSource(
        platform=Platform.DISCORD, chat_id="channel-123", chat_type="group",
        thread_id=thread_id, user_id="member-789", profile="travel",
    )
    expected_key = build_session_key(
        native,
        group_sessions_per_user=group_sessions_per_user,
        thread_sessions_per_user=thread_sessions_per_user,
        profile="travel",
    )
    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        f"  group_sessions_per_user: {str(group_sessions_per_user).lower()}\n"
        f"  thread_sessions_per_user: {str(thread_sessions_per_user).lower()}\n"
        "  session_key_aliases:\n"
        "    phone-thread:\n"
        "      platform: discord\n"
        "      chat_id: channel-123\n"
        "      chat_type: group\n"
        f"      thread_id: {thread_id or 'null'}\n"
        "      user_id: member-789\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    native_history = [
        {"role": "user", "content": "native thread question"},
        {"role": "assistant", "content": "native thread answer"},
    ]
    config = load_gateway_config()
    assert GatewayConfig.from_dict(config.to_dict()).session_key_aliases == config.session_key_aliases
    adapter = _adapter_with_native_history(tmp_path, native_history, expected_key)
    assert adapter._session_db is not None
    db = adapter._session_db
    persisted_history = db.get_messages_as_conversation("native-transcript")
    # The test scopes a non-default profile without constructing that profile's secret vault;
    # auth is orthogonal to alias resolution and remains exercised by the HTTP decorator.
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "session-key-test")
    adapter._run_agent = AsyncMock(return_value=(
        {"final_response": "continued native thread", "messages": []},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    ))

    profile_token = _api_request_profile.set("travel")
    try:
        key, source, error = adapter._resolve_api_session_identity("phone-thread")
        assert error is None and key == expected_key
        assert source == native
        async with TestClient(TestServer(_app(adapter))) as client:
            suffix = "/stream" if stream else ""
            headers = {
                "Authorization": "Bearer session-key-test",
                "X-Hermes-Session-Key": "phone-thread",
            }
            response = await client.post(
                f"/api/sessions/unrelated-transcript/chat{suffix}",
                headers=headers, json={"message": "must not mix identities"},
            )
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "session_alias_mismatch"
            adapter._run_agent.assert_not_awaited()
            response = await client.post(
                f"/api/sessions/native-transcript/chat{suffix}",
                headers=headers,
                json={"message": "continue this conversation"},
            )
            body = await response.text()
    finally:
        _api_request_profile.reset(profile_token)
        adapter._response_store.close()
        db.close()

    assert response.status == 200, body
    assert response.headers["X-Hermes-Session-Key"] == expected_key
    call = adapter._run_agent.await_args
    assert call is not None
    run_kwargs = call.kwargs
    assert run_kwargs["gateway_session_key"] == expected_key
    assert run_kwargs["conversation_history"] == persisted_history
    assert run_kwargs["session_id"] == "native-transcript"


@pytest.mark.asyncio
async def test_invalid_configured_alias_returns_typed_http_error(tmp_path, monkeypatch):
    """A bad configured alias must fail closed at the endpoint, not fall back to its alias text."""
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  session_key_aliases:\n    broken: not-a-session-source\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter_with_native_history(tmp_path, [])
    assert adapter._session_db is not None
    db = adapter._session_db
    adapter._run_agent = AsyncMock()
    try:
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/api/sessions/native-transcript/chat",
                headers={
                    "Authorization": "Bearer session-key-test",
                    "X-Hermes-Session-Key": "broken",
                },
                json={"message": "this must not execute"},
            )
            body = await response.json()
            assert response.status == 400
            assert body["error"]["code"] == "invalid_session_alias"
            adapter._run_agent.assert_not_awaited()

            def unexpected_config_load():
                raise AssertionError("No header must bypass alias configuration loading")

            monkeypatch.setattr("gateway.platforms.api_server.load_gateway_config", unexpected_config_load)
            adapter._run_agent.return_value = ({"final_response": "ordinary API turn"}, {})
            response = await client.post(
                "/api/sessions/unrelated-transcript/chat",
                headers={"Authorization": "Bearer session-key-test"},
                json={"message": "no alias"},
            )
            assert response.status == 200, await response.text()
            assert adapter._run_agent.await_args.kwargs["gateway_session_key"] is None
    finally:
        adapter._response_store.close()
        db.close()
