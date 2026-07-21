"""Mattermost-native approval and feedback interaction tests."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.mattermost.adapter import MattermostAdapter


def _adapter(monkeypatch, *, extra=None):
    monkeypatch.setenv("MATTERMOST_INTERACTION_SECRET", "test-secret-not-sent")
    config = PlatformConfig(
        enabled=True,
        token="bot-token",
        extra={
            "url": "https://mm.example.com",
            "interaction_url": "https://hermes.example.com/mattermost/actions",
            "interaction_allowed_cidrs": ["127.0.0.1/32"],
            **(extra or {}),
        },
    )
    adapter = MattermostAdapter(config)
    adapter._post_preserving_thread = AsyncMock(return_value={"id": "post-1"})
    adapter._thread_root_for_send = AsyncMock(
        side_effect=lambda _reply, metadata: (metadata or {}).get("thread_id")
    )
    return adapter


def _callback_from_action(action, *, user_id="user-1", post_id="post-1"):
    return {
        "user_id": user_id,
        "user_name": "Nawaf",
        "channel_id": "channel-1",
        "team_id": "team-1",
        "post_id": post_id,
        "context": action["integration"]["context"],
    }


def _authorize(adapter, allowed=True):
    class Runner:
        def __init__(self):
            self._is_user_authorized = MagicMock(return_value=allowed)

        def handler(self, _message):
            return None

    runner = Runner()
    adapter._message_handler = runner.handler
    adapter.set_authorization_check(runner._is_user_authorized)
    return runner


def test_interactive_authorization_uses_profile_bound_adapter_callback(monkeypatch):
    adapter = _adapter(monkeypatch)
    runner = _authorize(adapter, allowed=True)
    profile_bound_check = MagicMock(return_value=False)
    adapter.set_authorization_check(profile_bound_check)

    assert (
        adapter._is_interactive_user_authorized(
            "user-1", channel_id="channel-1", team_id="team-1"
        )
        is False
    )
    profile_bound_check.assert_called_once_with("user-1", "group", "channel-1")
    runner._is_user_authorized.assert_not_called()


def test_interaction_secret_does_not_fall_back_outside_profile_scope(monkeypatch):
    monkeypatch.setenv("MATTERMOST_INTERACTION_SECRET", "wrong-profile-secret")
    with patch("agent.secret_scope.get_secret", return_value=None):
        assert MattermostAdapter._secret_value("MATTERMOST_INTERACTION_SECRET") == ""


def test_interactions_require_valid_url_secret_and_trusted_source(monkeypatch):
    adapter = _adapter(monkeypatch)
    assert adapter._interactions_available() is True

    monkeypatch.delenv("MATTERMOST_INTERACTION_SECRET")
    assert adapter._interactions_available() is False
    assert (
        _adapter(monkeypatch, extra={"interaction_url": ""})._interactions_available()
        is False
    )
    assert (
        _adapter(
            monkeypatch, extra={"interaction_url": "https://"}
        )._interactions_available()
        is False
    )
    assert (
        _adapter(
            monkeypatch, extra={"interaction_url": "http://hermes.internal/actions"}
        )._interactions_available()
        is False
    )
    assert (
        _adapter(
            monkeypatch, extra={"interaction_url": "http://127.0.0.1:8789/actions"}
        )._interactions_available()
        is True
    )
    assert (
        _adapter(
            monkeypatch, extra={"interaction_allowed_cidrs": []}
        )._interactions_available()
        is False
    )


def test_feedback_state_never_evicts_live_approvals(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr("plugins.platforms.mattermost.adapter.time.time", lambda: 1000)
    for index in range(256):
        assert adapter._remember_interaction(
            f"approval-{index}",
            {"kind": "approval", "expires_at": 2000},
        )

    remembered = adapter._remember_interaction(
        "feedback-new",
        {"kind": "feedback", "expires_at": 2000},
    )

    assert remembered is False
    assert "feedback-new" not in adapter._pending_interactions
    assert len(adapter._pending_interactions) == 256
    assert all(
        state["kind"] == "approval" for state in adapter._pending_interactions.values()
    )


def test_remember_interaction_purges_expired_state(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr("plugins.platforms.mattermost.adapter.time.time", lambda: 1000)
    adapter._pending_interactions["expired"] = {
        "kind": "approval",
        "expires_at": 999,
    }

    assert adapter._remember_interaction(
        "fresh", {"kind": "feedback", "expires_at": 2000}
    )
    assert set(adapter._pending_interactions) == {"fresh"}


@pytest.mark.asyncio
async def test_send_exec_approval_uses_signed_native_actions(monkeypatch):
    adapter = _adapter(monkeypatch)

    result = await adapter.send_exec_approval(
        chat_id="channel-1",
        command="deploy production",
        session_key="session-1",
        approval_id="approval-1",
        description="Production deployment",
        metadata={"thread_id": "root-1"},
        allow_permanent=True,
        allow_session=True,
    )

    assert result.success is True
    assert result.message_id == "post-1"
    payload = adapter._post_preserving_thread.await_args.args[1]
    assert payload["channel_id"] == "channel-1"
    assert payload["root_id"] == "root-1"
    attachment = payload["props"]["attachments"][0]
    assert attachment["fallback"].startswith("Command approval required")
    assert "deploy production" in attachment["text"]
    actions = attachment["actions"]
    assert [action["name"] for action in actions] == [
        "Allow Once",
        "Allow Session",
        "Always Allow",
        "Deny",
    ]
    assert all(action["type"] == "button" for action in actions)
    assert all(
        action["integration"]["url"].startswith("https://") for action in actions
    )
    serialized = repr(payload)
    assert "test-secret-not-sent" not in serialized
    assert "session-1" not in serialized
    assert "approval-1" not in serialized


@pytest.mark.asyncio
async def test_smart_denied_approval_only_offers_once_and_deny(monkeypatch):
    adapter = _adapter(monkeypatch)

    result = await adapter.send_exec_approval(
        "channel-1",
        "rm -rf build",
        "session-1",
        approval_id="approval-1",
        smart_denied=True,
    )

    assert result.success is True
    payload = adapter._post_preserving_thread.await_args.args[1]
    actions = payload["props"]["attachments"][0]["actions"]
    assert [action["name"] for action in actions] == ["Allow Once", "Deny"]
    assert "Smart DENY" in payload["props"]["attachments"][0]["text"]


@pytest.mark.asyncio
async def test_permanent_approval_is_independent_of_session_option(monkeypatch):
    adapter = _adapter(monkeypatch)

    await adapter.send_exec_approval(
        "channel-1",
        "deploy production",
        "session-1",
        approval_id="approval-1",
        allow_session=False,
        allow_permanent=True,
    )

    payload = adapter._post_preserving_thread.await_args.args[1]
    actions = payload["props"]["attachments"][0]["actions"]
    assert [action["name"] for action in actions] == [
        "Allow Once",
        "Always Allow",
        "Deny",
    ]


@pytest.mark.asyncio
async def test_send_exec_approval_falls_back_when_callbacks_unavailable(monkeypatch):
    adapter = _adapter(monkeypatch, extra={"interaction_url": ""})

    result = await adapter.send_exec_approval("channel-1", "date", "session-1")

    assert result.success is False
    assert "interaction" in (result.error or "").lower()
    adapter._post_preserving_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_approval_callback_authorizes_resolves_and_removes_buttons(
    monkeypatch,
):
    adapter = _adapter(monkeypatch)
    runner = _authorize(adapter, allowed=True)
    adapter.resume_typing_for_chat = MagicMock()
    await adapter.send_exec_approval(
        "channel-1",
        "deploy production",
        "session-1",
        approval_id="approval-1",
    )
    payload = adapter._post_preserving_thread.await_args.args[1]
    action = payload["props"]["attachments"][0]["actions"][0]

    with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
        body, status = await adapter._dispatch_interaction(
            _callback_from_action(action)
        )

    assert status == 200
    resolve.assert_called_once_with("session-1", "once", approval_id="approval-1")
    adapter.resume_typing_for_chat.assert_called_once_with("channel-1")
    runner._is_user_authorized.assert_called_once()
    assert "Approved once by Nawaf" in body["update"]["message"]
    assert "actions" not in body["update"]["props"]["attachments"][0]


@pytest.mark.asyncio
async def test_approval_callback_rejects_unauthorized_user(monkeypatch):
    adapter = _adapter(monkeypatch)
    _authorize(adapter, allowed=False)
    await adapter.send_exec_approval(
        "channel-1", "deploy production", "session-1", approval_id="approval-1"
    )
    payload = adapter._post_preserving_thread.await_args.args[1]
    action = payload["props"]["attachments"][0]["actions"][0]

    with patch("tools.approval.resolve_gateway_approval") as resolve:
        body, status = await adapter._dispatch_interaction(
            _callback_from_action(action, user_id="intruder")
        )

    assert status == 403
    resolve.assert_not_called()
    assert "authorized" in body["ephemeral_text"].lower()


@pytest.mark.asyncio
async def test_tampered_and_replayed_callbacks_do_not_resolve(monkeypatch):
    adapter = _adapter(monkeypatch)
    _authorize(adapter, allowed=True)
    await adapter.send_exec_approval(
        "channel-1", "deploy production", "session-1", approval_id="approval-1"
    )
    payload = adapter._post_preserving_thread.await_args.args[1]
    action = payload["props"]["attachments"][0]["actions"][0]
    tampered = _callback_from_action(action)
    tampered["context"] = dict(tampered["context"], choice="always")

    with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
        body, status = await adapter._dispatch_interaction(tampered)
        assert status == 401
        resolve.assert_not_called()

        valid = _callback_from_action(action)
        _, first_status = await adapter._dispatch_interaction(valid)
        replay_body, replay_status = await adapter._dispatch_interaction(valid)

    assert first_status == 200
    assert replay_status == 200
    assert "expired" in replay_body["ephemeral_text"].lower()
    resolve.assert_called_once()


@pytest.mark.asyncio
async def test_expired_callback_is_rejected_and_consumed(monkeypatch):
    adapter = _adapter(monkeypatch, extra={"interaction_timeout_seconds": 60})
    _authorize(adapter, allowed=True)
    await adapter.send_exec_approval(
        "channel-1", "deploy production", "session-1", approval_id="approval-1"
    )
    payload = adapter._post_preserving_thread.await_args.args[1]
    action = payload["props"]["attachments"][0]["actions"][0]
    callback = _callback_from_action(action)
    expires_at = action["integration"]["context"]["expires_at"]

    with (
        patch(
            "plugins.platforms.mattermost.adapter.time.time",
            return_value=expires_at + 1,
        ),
        patch("tools.approval.resolve_gateway_approval") as resolve,
    ):
        body, status = await adapter._dispatch_interaction(callback)
        replay_body, replay_status = await adapter._dispatch_interaction(callback)

    assert status == 200 and replay_status == 200
    assert "expired" in body["ephemeral_text"].lower()
    assert "expired" in replay_body["ephemeral_text"].lower()
    resolve.assert_not_called()


@pytest.mark.asyncio
async def test_callback_must_match_original_channel_and_post(monkeypatch):
    adapter = _adapter(monkeypatch)
    _authorize(adapter, allowed=True)
    await adapter.send_exec_approval(
        "channel-1", "deploy production", "session-1", approval_id="approval-1"
    )
    payload = adapter._post_preserving_thread.await_args.args[1]
    action = payload["props"]["attachments"][0]["actions"][0]
    callback = _callback_from_action(action, post_id="different-post")

    with patch("tools.approval.resolve_gateway_approval") as resolve:
        body, status = await adapter._dispatch_interaction(callback)

    assert status == 401
    resolve.assert_not_called()
    assert "invalid" in body["ephemeral_text"].lower()


@pytest.mark.asyncio
async def test_http_callback_handler_returns_dispatch_result(monkeypatch):
    adapter = _adapter(monkeypatch)
    adapter._dispatch_interaction = AsyncMock(
        return_value=({"ephemeral_text": "Thanks"}, 200)
    )
    request = SimpleNamespace(
        remote="127.0.0.1",
        content_length=123,
        json=AsyncMock(return_value={"x": 1}),
    )

    response = await adapter._handle_interaction_request(request)

    assert response.status == 200
    assert json.loads(response.text) == {"ephemeral_text": "Thanks"}
    adapter._dispatch_interaction.assert_awaited_once_with({"x": 1})


@pytest.mark.asyncio
async def test_http_callback_handler_rejects_oversized_or_invalid_json(monkeypatch):
    adapter = _adapter(monkeypatch)
    oversized = SimpleNamespace(
        remote="127.0.0.1", content_length=65_537, json=AsyncMock()
    )
    invalid = SimpleNamespace(
        remote="127.0.0.1",
        content_length=10,
        json=AsyncMock(side_effect=UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")),
    )

    oversized_response = await adapter._handle_interaction_request(oversized)
    invalid_response = await adapter._handle_interaction_request(invalid)

    assert oversized_response.status == 413
    assert invalid_response.status == 400


@pytest.mark.asyncio
async def test_http_callback_handler_rejects_forged_user_from_untrusted_source(
    monkeypatch,
):
    adapter = _adapter(monkeypatch)
    _authorize(adapter, allowed=True)
    await adapter.send_exec_approval(
        "channel-1", "deploy production", "session-1", approval_id="approval-1"
    )
    payload = adapter._post_preserving_thread.await_args.args[1]
    action = payload["props"]["attachments"][0]["actions"][0]
    request = SimpleNamespace(
        remote="203.0.113.9",
        content_length=100,
        json=AsyncMock(
            return_value=_callback_from_action(action, user_id="allowed-user")
        ),
    )

    with patch("tools.approval.resolve_gateway_approval") as resolve:
        response = await adapter._handle_interaction_request(request)

    assert response.status == 403
    request.json.assert_not_awaited()
    resolve.assert_not_called()


@pytest.mark.asyncio
async def test_interaction_server_starts_and_stops_with_config(monkeypatch):
    adapter = _adapter(
        monkeypatch,
        extra={"interaction_host": "127.0.0.1", "interaction_port": 9876},
    )
    fake_runner = SimpleNamespace(setup=AsyncMock(), cleanup=AsyncMock())
    fake_site = SimpleNamespace(start=AsyncMock())

    with (
        patch("aiohttp.web.AppRunner", return_value=fake_runner) as runner_cls,
        patch("aiohttp.web.TCPSite", return_value=fake_site) as site_cls,
    ):
        await adapter._start_interaction_server()
        await adapter._stop_interaction_server()

    fake_runner.setup.assert_awaited_once()
    fake_site.start.assert_awaited_once()
    site_cls.assert_called_once_with(fake_runner, "127.0.0.1", 9876)
    fake_runner.cleanup.assert_awaited_once()
    app = runner_cls.call_args.args[0]
    routes = {(route.method, route.resource.canonical) for route in app.router.routes()}
    assert ("POST", "/mattermost/actions") in routes
    assert ("GET", "/mattermost/actions/health") in routes
