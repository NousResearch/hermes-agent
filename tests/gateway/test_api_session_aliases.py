"""HTTP regression coverage for configured API session-key aliases."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.session import SessionSource, build_session_key


def _app(adapter):
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    return app


def _adapter_with_native_history(history):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "session-key-test"}))
    db = MagicMock()
    db.get_session.return_value = {"id": "native-transcript", "source": "api_server"}
    db.get_messages_as_conversation.return_value = history
    adapter._session_db = db
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("group_sessions_per_user", "thread_sessions_per_user"),
    [(True, True), (False, False)],
    ids=["per-user-group-and-thread", "shared-group-and-thread"],
)
async def test_configured_alias_executes_with_native_thread_identity_and_history(
    tmp_path, monkeypatch, group_sessions_per_user, thread_sessions_per_user
):
    """The HTTP consumer must receive the exact native key and persisted transcript."""
    native = SessionSource(
        platform=Platform.DISCORD, chat_id="channel-123", chat_type="group",
        thread_id="thread-456", user_id="member-789", profile="travel",
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
        "      thread_id: thread-456\n"
        "      user_id: member-789\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    native_history = [
        {"role": "user", "content": "native thread question"},
        {"role": "assistant", "content": "native thread answer"},
    ]
    adapter = _adapter_with_native_history(native_history)
    # The test scopes a non-default profile without constructing that profile's secret vault;
    # auth is orthogonal to alias resolution and remains exercised by the HTTP decorator.
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "session-key-test")
    adapter._run_agent = AsyncMock(return_value=(
        {"final_response": "continued native thread", "messages": []},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    ))

    profile_token = _api_request_profile.set("travel")
    try:
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/api/sessions/native-transcript/chat",
                headers={
                    "Authorization": "Bearer session-key-test",
                    "X-Hermes-Session-Key": "phone-thread",
                },
                json={"message": "continue this conversation"},
            )
            body = await response.json()
    finally:
        _api_request_profile.reset(profile_token)
        adapter._response_store.close()

    assert response.status == 200, body
    assert response.headers["X-Hermes-Session-Key"] == expected_key
    call = adapter._run_agent.await_args
    assert call is not None
    run_kwargs = call.kwargs
    assert run_kwargs["gateway_session_key"] == expected_key
    assert run_kwargs["conversation_history"] == native_history
    assert run_kwargs["session_id"] == "native-transcript"


@pytest.mark.asyncio
async def test_invalid_configured_alias_returns_typed_http_error(tmp_path, monkeypatch):
    """A bad configured alias must fail closed at the endpoint, not fall back to its alias text."""
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  session_key_aliases:\n    broken: not-a-session-source\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter_with_native_history([])
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
    finally:
        adapter._response_store.close()

    assert response.status == 400
    assert body["error"]["code"] == "invalid_session_alias"
    adapter._run_agent.assert_not_awaited()
