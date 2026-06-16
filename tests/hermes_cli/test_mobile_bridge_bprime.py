"""B-prime mobile bridge TDD tests.

These tests define the smaller current-baseline bridge contract before any
production code is added:

* mobile/native chat uses dashboard auth under ``/api/*`` instead of reviving
  the old public ``/mobile-native`` namespace;
* browser/mobile clients never receive the raw assistant body or API-server
  bearer credentials;
* the per-turn agent cleanup boundary is soft: ``release_clients()`` only,
  never ``agent.close()`` or ``shutdown_memory_provider()``.
"""
from __future__ import annotations

import pytest


def _dashboard_client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client, web_server


def _load_mobile_turn_helper():
    try:
        from hermes_cli.mobile_bridge import run_mobile_dashboard_chat_turn
    except ImportError as exc:  # Expected RED until the bridge module exists.
        pytest.fail(f"B-prime mobile bridge helper is missing: {exc}")
    return run_mobile_dashboard_chat_turn


def test_mobile_bridge_routes_stay_behind_dashboard_auth_allowlist():
    from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS

    assert "/api/mobile/sessions" not in PUBLIC_API_PATHS
    assert "/api/mobile/sessions/{session_id}/chat" not in PUBLIC_API_PATHS
    assert not any(path.startswith("/api/mobile") for path in PUBLIC_API_PATHS)


def test_dashboard_mobile_chat_route_uses_existing_auth_and_redacts_response(
    _isolate_hermes_home,
    monkeypatch,
):
    client, web_server = _dashboard_client()
    calls: dict[str, object] = {}

    async def fake_turn(**kwargs):
        calls.update(kwargs)
        return {
            "object": "hermes.mobile.chat.completion",
            "session_id": kwargs["session_id"],
            "message": {
                "role": "assistant",
                "content_redacted": True,
                "content_length": len("assistant secret body"),
            },
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }

    # The future route should call this helper. ``raising=False`` keeps this
    # test RED as a route-missing 404 instead of failing during monkeypatch.
    monkeypatch.setattr(web_server, "_run_mobile_dashboard_chat_turn", fake_turn, raising=False)

    session_id = "mobile-bprime-session"
    response = client.post(
        f"/api/mobile/sessions/{session_id}/chat",
        json={"message": "hello from ipad", "system_message": "do not leak"},
        headers={"X-Hermes-Session-Key": session_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "hermes.mobile.chat.completion"
    assert body["session_id"] == session_id
    assert body["message"] == {
        "role": "assistant",
        "content_redacted": True,
        "content_length": len("assistant secret body"),
    }
    assert "assistant secret body" not in response.text
    assert "API_SERVER_KEY" not in response.text
    assert "Authorization" not in response.text
    assert "Bearer " not in response.text
    assert calls["session_id"] == session_id
    assert calls["user_message"] == "hello from ipad"
    assert calls["system_message"] == "do not leak"
    assert calls["gateway_session_key"] == session_id


def test_dashboard_mobile_chat_route_validates_gateway_session_key(
    _isolate_hermes_home,
    monkeypatch,
):
    client, web_server = _dashboard_client()
    calls: list[dict[str, object]] = []

    async def fake_turn(**kwargs):
        calls.append(kwargs)
        return {
            "object": "hermes.mobile.chat.completion",
            "session_id": kwargs["session_id"],
            "message": {"role": "assistant", "content_redacted": True, "content_length": 0},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    monkeypatch.setattr(web_server, "_run_mobile_dashboard_chat_turn", fake_turn, raising=False)

    session_id = "mobile-bprime-session"
    too_long = "x" * 257
    rejected = client.post(
        f"/api/mobile/sessions/{session_id}/chat",
        json={"message": "hello"},
        headers={"X-Hermes-Session-Key": too_long},
    )
    assert rejected.status_code == 400
    assert calls == []

    accepted = client.post(
        f"/api/mobile/sessions/{session_id}/chat",
        json={"message": "hello"},
        headers={"X-Hermes-Session-Key": "  ipad-session-key  "},
    )
    assert accepted.status_code == 200
    assert calls[0]["gateway_session_key"] == "ipad-session-key"


@pytest.mark.asyncio
async def test_mobile_turn_helper_soft_releases_agent_and_disconnects_owned_adapter():
    run_mobile_dashboard_chat_turn = _load_mobile_turn_helper()

    class FakeAgent:
        def __init__(self):
            self.release_calls = 0

        async def release_clients(self):
            self.release_calls += 1

        def close(self):  # pragma: no cover - must never be called
            raise AssertionError("mobile per-turn cleanup must not call agent.close()")

        def shutdown_memory_provider(self, *args, **kwargs):  # pragma: no cover - must never be called
            raise AssertionError("mobile per-turn cleanup must not shutdown memory provider")

    class FakeAdapter:
        def __init__(self):
            self.agent = FakeAgent()
            self.run_kwargs = None
            self.disconnect_calls = 0

        async def _run_agent(self, **kwargs):
            self.run_kwargs = kwargs
            kwargs["agent_ref"][0] = self.agent
            return (
                {"final_response": "assistant secret body", "session_id": kwargs["session_id"]},
                {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
            )

        async def disconnect(self):
            self.disconnect_calls += 1

    adapter = FakeAdapter()
    conversation_history = [{"role": "user", "content": "previous turn"}]

    result = await run_mobile_dashboard_chat_turn(
        session_id="mobile-bprime-session",
        user_message="continue on ipad",
        conversation_history=conversation_history,
        system_message="short system prompt",
        gateway_session_key="mobile-bprime-session",
        adapter_factory=lambda: adapter,
    )

    assert adapter.run_kwargs["user_message"] == "continue on ipad"
    assert adapter.run_kwargs["conversation_history"] == conversation_history
    assert adapter.run_kwargs["ephemeral_system_prompt"] == "short system prompt"
    assert adapter.run_kwargs["session_id"] == "mobile-bprime-session"
    assert adapter.run_kwargs["gateway_session_key"] == "mobile-bprime-session"
    assert adapter.agent.release_calls == 1
    assert adapter.disconnect_calls == 1
    assert result == {
        "object": "hermes.mobile.chat.completion",
        "session_id": "mobile-bprime-session",
        "message": {
            "role": "assistant",
            "content_redacted": True,
            "content_length": len("assistant secret body"),
        },
        "usage": {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
    }
    assert "assistant secret body" not in str(result)


@pytest.mark.asyncio
async def test_mobile_turn_helper_cleanup_survives_release_failure_and_closes_session_db():
    run_mobile_dashboard_chat_turn = _load_mobile_turn_helper()

    class FailingReleaseAgent:
        def __init__(self):
            self.release_calls = 0

        async def release_clients(self):
            self.release_calls += 1
            raise RuntimeError("release failed")

        def close(self):  # pragma: no cover - must never be called
            raise AssertionError("mobile per-turn cleanup must not call agent.close()")

        def shutdown_memory_provider(self, *args, **kwargs):  # pragma: no cover - must never be called
            raise AssertionError("mobile per-turn cleanup must not shutdown memory provider")

    class FakeSessionDB:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class FakeAdapter:
        def __init__(self):
            self.agent = FailingReleaseAgent()
            self._session_db = FakeSessionDB()
            self.disconnect_calls = 0

        async def _run_agent(self, **kwargs):
            kwargs["agent_ref"][0] = self.agent
            return (
                {"final_response": "assistant body", "session_id": kwargs["session_id"]},
                {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            )

        async def disconnect(self):
            self.disconnect_calls += 1

    adapter = FakeAdapter()

    result = await run_mobile_dashboard_chat_turn(
        session_id="mobile-bprime-session",
        user_message="continue on ipad",
        gateway_session_key="mobile-bprime-session",
        adapter_factory=lambda: adapter,
    )

    assert result["message"] == {
        "role": "assistant",
        "content_redacted": True,
        "content_length": len("assistant body"),
    }
    assert adapter.agent.release_calls == 1
    assert adapter.disconnect_calls == 1
    assert adapter._session_db.close_calls == 1
    assert "assistant body" not in str(result)
