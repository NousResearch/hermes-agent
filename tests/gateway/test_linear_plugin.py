"""Tests for the native Linear AgentSession platform plugin."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome, SendResult
from gateway.platform_registry import PlatformEntry, PlatformRegistry
from plugins.platforms.linear import adapter as linear_adapter
from plugins.platforms.linear.adapter import LinearAgentSessionAdapter
from plugins.platforms.linear.linear_client import LinearClient


def _signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _adapter(secret="linear-secret", token="linear-token"):
    return LinearAgentSessionAdapter(PlatformConfig(enabled=True, extra={"webhook_secret": secret, "token": token}))


@pytest.mark.asyncio
async def test_agent_session_signature_session_mapping_and_created_thought():
    payload = {
        "type": "AgentSessionEvent",
        "action": "created",
        "agentSession": {"id": "as_123", "promptContext": "Investigate LIN-1"},
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    adapter = _adapter()
    adapter.client.create_agent_activity = AsyncMock(return_value={"id": "act_thought"})
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture

    status, data = adapter._handle_agent_session_body(
        body=body,
        headers={"Linear-Signature": _signature(body, "linear-secret"), "Linear-Delivery": "delivery-1"},
        content_length=len(body),
    )
    assert status == 200
    assert data["status"] == "accepted"

    await adapter.wait_background_tasks()
    adapter.client.create_agent_activity.assert_awaited_once()
    _, kwargs = adapter.client.create_agent_activity.call_args
    assert kwargs["agent_session_id"] == "as_123"
    assert kwargs["content"]["type"] == "thought"

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "Investigate LIN-1"
    assert event.source.chat_type == "dm"
    assert event.source.chat_id == "agentSession:as_123"
    assert event.source.user_id == "linear-agent-session"
    assert event.source.user_name == "Linear Agent Session"
    assert event.message_id == "delivery-1"
    assert event.raw_message == payload


@pytest.mark.asyncio
async def test_agent_session_duplicate_delivery_is_acked_without_agent_run():
    payload = {"action": "prompted", "agentSession": {"id": "as_123"}, "agentActivity": {"id": "act_1", "body": "hello"}}
    body = json.dumps(payload).encode()
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    headers = {"Linear-Signature": _signature(body, "linear-secret"), "Linear-Delivery": "same-delivery"}
    status1, _data1 = adapter._handle_agent_session_body(body=body, headers=headers, content_length=len(body))
    status2, data2 = adapter._handle_agent_session_body(body=body, headers=headers, content_length=len(body))
    assert status1 == 200
    assert status2 == 200
    assert data2["status"] == "duplicate"

    await adapter.wait_background_tasks()
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_agent_session_duplicate_fallback_prefers_webhook_id():
    payload = {
        "webhookId": "webhook-delivery-1",
        "id": "payload-id-should-not-be-used",
        "action": "prompted",
        "agentSession": {"id": "as_123"},
        "agentActivity": {"id": "act_1", "body": "hello"},
    }
    body = json.dumps(payload).encode()
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    headers = {"Linear-Signature": _signature(body, "linear-secret")}
    status1, data1 = adapter._handle_agent_session_body(body=body, headers=headers, content_length=len(body))
    status2, data2 = adapter._handle_agent_session_body(body=body, headers=headers, content_length=len(body))
    assert status1 == 200
    assert data1["delivery_id"] == "webhook-delivery-1"
    assert status2 == 200
    assert data2["status"] == "duplicate"
    assert data2["delivery_id"] == "webhook-delivery-1"

    await adapter.wait_background_tasks()
    assert adapter.handle_message.await_count == 1


def test_requirements_only_check_dependencies_not_credentials(monkeypatch):
    monkeypatch.setattr(linear_adapter, "AIOHTTP_AVAILABLE", True)
    monkeypatch.delenv("LINEAR_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    assert linear_adapter.check_requirements() is True

    monkeypatch.setattr(linear_adapter, "AIOHTTP_AVAILABLE", False)
    assert linear_adapter.check_requirements() is False


def test_registry_can_create_adapter_from_config_credentials(monkeypatch):
    monkeypatch.setattr(linear_adapter, "AIOHTTP_AVAILABLE", True)
    monkeypatch.delenv("LINEAR_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    registry = PlatformRegistry()
    registry.register(PlatformEntry(
        name="linear",
        label="Linear",
        adapter_factory=lambda cfg: LinearAgentSessionAdapter(cfg),
        check_fn=linear_adapter.check_requirements,
        validate_config=linear_adapter.validate_config,
    ))

    cfg = PlatformConfig(enabled=True, extra={"webhook_secret": "secret", "token": "config-token"})
    assert isinstance(registry.create_adapter("linear", cfg), LinearAgentSessionAdapter)


def test_config_validation_accepts_config_token_as_outbound_token(monkeypatch):
    monkeypatch.delenv("LINEAR_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    missing_token = PlatformConfig(enabled=True, extra={"webhook_secret": "secret"})
    with_token = PlatformConfig(enabled=True, extra={"webhook_secret": "secret", "token": "config-token"})
    with_access_token = PlatformConfig(enabled=True, extra={"webhook_secret": "secret", "access_token": "config-access-token"})
    with_api_key = PlatformConfig(enabled=True, extra={"webhook_secret": "secret", "api_key": "config-api-key"})
    assert linear_adapter.validate_config(missing_token) is False
    assert linear_adapter.is_connected(missing_token) is False
    assert linear_adapter.validate_config(with_token) is True
    assert linear_adapter.is_connected(with_token) is True
    assert LinearAgentSessionAdapter(with_token)._token == "config-token"
    assert linear_adapter.validate_config(with_access_token) is True
    assert LinearAgentSessionAdapter(with_access_token)._token == "config-access-token"
    assert linear_adapter.validate_config(with_api_key) is True
    assert LinearAgentSessionAdapter(with_api_key)._token == "config-api-key"


def test_env_enablement_requires_outbound_token(monkeypatch):
    monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "secret")
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    assert linear_adapter._env_enablement() is None

    monkeypatch.setenv("LINEAR_API_KEY", "api-key")
    seed = linear_adapter._env_enablement()
    assert seed["webhook_secret"] == "secret"
    assert seed["token"] == "api-key"


def test_register_declares_secret_and_token_requirements():
    class _Ctx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    ctx = _Ctx()
    linear_adapter.register(ctx)
    assert ctx.kwargs["required_env"] == ["LINEAR_WEBHOOK_SECRET"]
    assert "LINEAR_ACCESS_TOKEN" in ctx.kwargs["install_hint"]
    assert "LINEAR_API_KEY" in ctx.kwargs["install_hint"]
    assert ctx.kwargs["allowed_users_env"] == "LINEAR_ALLOWED_USERS"
    assert ctx.kwargs["allow_all_env"] == "LINEAR_ALLOW_ALL_USERS"
    assert ctx.kwargs["suppress_home_channel_prompt"] is True


@pytest.mark.asyncio
async def test_connect_refuses_missing_outbound_token(monkeypatch):
    monkeypatch.setattr(linear_adapter, "AIOHTTP_AVAILABLE", True)
    adapter = LinearAgentSessionAdapter(PlatformConfig(enabled=True, extra={"webhook_secret": "secret"}))
    assert await adapter.connect() is False


@pytest.mark.asyncio
async def test_agent_session_prompted_and_stop_signal_mapping():
    adapter = _adapter()
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    prompted = {"action": "prompted", "agentSession": {"id": "as_1", "promptContext": "ctx"}, "agentActivity": {"id": "act_2", "body": "do it"}}
    stop = {"action": "prompted", "agentSession": {"id": "as_1"}, "agentActivity": {"id": "act_3", "body": "ignored", "signal": "stop"}}

    await adapter._process_agent_session_event(prompted, "delivery-prompt")
    await adapter._process_agent_session_event(stop, "delivery-stop")

    assert captured[0].text == "do it\n\nContext:\nctx"
    assert captured[0].message_id == "act_2"
    assert captured[1].text == "/stop"
    assert captured[1].source.chat_id == "agentSession:as_1"


@pytest.mark.asyncio
async def test_prompted_slash_command_does_not_append_prompt_context():
    adapter = _adapter()
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    payload = {
        "action": "prompted",
        "agentSession": {"id": "as_1", "promptContext": "ctx should not become slash args"},
        "agentActivity": {"id": "act_approve", "body": "/approve session"},
    }

    await adapter._process_agent_session_event(payload, "delivery-approve")

    assert captured[0].text == "/approve session"


@pytest.mark.asyncio
async def test_prompted_clarify_reply_does_not_append_prompt_context():
    adapter = _adapter()
    adapter._awaiting_clarify_sessions["as_1"] = linear_adapter.time.time() + 60.0
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    payload = {
        "action": "prompted",
        "agentSession": {"id": "as_1", "promptContext": "ctx should not contaminate choice"},
        "agentActivity": {"id": "act_choice", "body": "Friday morning"},
    }

    await adapter._process_agent_session_event(payload, "delivery-choice")

    assert captured[0].text == "Friday morning"
    assert "as_1" not in adapter._awaiting_clarify_sessions


@pytest.mark.asyncio
async def test_expired_clarify_marker_does_not_suppress_prompt_context():
    adapter = _adapter()
    adapter._awaiting_clarify_sessions["as_1"] = linear_adapter.time.time() - 1.0
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    payload = {
        "action": "prompted",
        "agentSession": {"id": "as_1", "promptContext": "ctx should be kept"},
        "agentActivity": {"id": "act_followup", "body": "normal follow-up"},
    }

    await adapter._process_agent_session_event(payload, "delivery-followup")

    assert captured[0].text == "normal follow-up\n\nContext:\nctx should be kept"
    assert "as_1" not in adapter._awaiting_clarify_sessions


@pytest.mark.asyncio
async def test_stop_signal_clears_pending_clarify_marker():
    adapter = _adapter()
    adapter._awaiting_clarify_sessions["as_1"] = linear_adapter.time.time() + 60.0
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    payload = {
        "action": "prompted",
        "agentSession": {"id": "as_1"},
        "agentActivity": {"id": "act_stop", "body": "ignored", "signal": "stop"},
    }

    await adapter._process_agent_session_event(payload, "delivery-stop")

    assert captured[0].text == "/stop"
    assert adapter._awaiting_clarify_sessions == {}


@pytest.mark.asyncio
async def test_agent_session_auth_identity_prefers_linear_actor():
    adapter = _adapter()
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    payload = {
        "action": "prompted",
        "actor": {"id": "lin_user_123", "name": "Paul"},
        "agentSession": {"id": "as_1"},
        "agentActivity": {"id": "act_2", "content": {"body": "hello"}},
    }

    await adapter._process_agent_session_event(payload, "delivery-prompt")

    assert captured[0].text == "hello"
    assert captured[0].source.chat_id == "agentSession:as_1"
    assert captured[0].source.user_id == "lin_user_123"
    assert captured[0].source.user_name == "Paul"


@pytest.mark.asyncio
async def test_send_creates_agent_activity_response_payload():
    adapter = _adapter()
    adapter.client.create_agent_activity = AsyncMock(return_value={"id": "act_response"})

    result = await adapter.send("agentSession:as_123", "done", metadata={"content_type": "response"})

    assert result == SendResult(success=True, message_id="act_response")
    adapter.client.create_agent_activity.assert_awaited_once_with(
        agent_session_id="as_123",
        content={"type": "response", "body": "done"},
    )


@pytest.mark.asyncio
async def test_processing_start_emits_action_and_in_progress_plan():
    adapter = _adapter()
    adapter.client.get_agent_session_work_context = AsyncMock(return_value={
        "viewer": {"id": "hio_user", "name": "Hio"},
        "agentSession": {
            "issue": {
                "id": "issue_1",
                "delegate": {"id": "someone", "name": "Someone"},
                "state": {"id": "state_started", "type": "started"},
                "team": {"states": {"nodes": []}},
            }
        },
    })
    adapter.client.update_issue = AsyncMock()
    adapter.client.create_agent_activity = AsyncMock(return_value={"id": "act_action"})
    adapter.client.update_agent_session = AsyncMock(return_value={"id": "as_123", "status": "active"})
    source = adapter.build_source(
        chat_id="agentSession:as_123",
        chat_name="Linear AgentSession as_123",
        chat_type="dm",
        user_id="lin_user_123",
        user_name="Paul",
    )
    event = MessageEvent(
        text="Investigate BEN-1 and schedule product reviews",
        message_type=MessageType.TEXT,
        source=source,
    )

    await adapter.on_processing_start(event)

    adapter.client.get_agent_session_work_context.assert_awaited_once_with(agent_session_id="as_123")
    adapter.client.update_issue.assert_not_called()
    adapter.client.create_agent_activity.assert_awaited_once_with(
        agent_session_id="as_123",
        content={
            "type": "action",
            "action": "Processing",
            "parameter": "Investigate BEN-1 and schedule product reviews",
        },
        ephemeral=True,
    )
    adapter.client.update_agent_session.assert_awaited_once_with(
        agent_session_id="as_123",
        plan=[{"content": "Investigate BEN-1 and schedule product reviews", "status": "inProgress"}],
    )


@pytest.mark.asyncio
async def test_processing_start_prepares_issue_delegate_and_started_state():
    adapter = _adapter()
    adapter.client.get_agent_session_work_context = AsyncMock(return_value={
        "viewer": {"id": "hio_user", "name": "Hio"},
        "agentSession": {
            "issue": {
                "id": "issue_1",
                "delegate": None,
                "state": {"id": "state_triage", "type": "triage"},
                "team": {
                    "states": {
                        "nodes": [
                            {"id": "state_later", "type": "started", "position": 200},
                            {"id": "state_first", "type": "started", "position": 10},
                        ]
                    }
                },
            }
        },
    })
    adapter.client.update_issue = AsyncMock(return_value={
        "id": "issue_1",
        "delegate": {"id": "hio_user", "name": "Hio"},
        "state": {"id": "state_first", "type": "started"},
    })
    adapter.client.create_agent_activity = AsyncMock(return_value={"id": "act_action"})
    adapter.client.update_agent_session = AsyncMock(return_value={"id": "as_123", "status": "active"})
    source = adapter.build_source(
        chat_id="agentSession:as_123",
        chat_name="Linear AgentSession as_123",
        chat_type="dm",
        user_id="lin_user_123",
        user_name="Paul",
    )
    event = MessageEvent(text="Start work", message_type=MessageType.TEXT, source=source)

    await adapter.on_processing_start(event)

    adapter.client.update_issue.assert_awaited_once_with(
        issue_id="issue_1",
        delegate_id="hio_user",
        state_id="state_first",
    )


@pytest.mark.asyncio
async def test_processing_start_does_not_overwrite_existing_delegate():
    adapter = _adapter()
    adapter.client.get_agent_session_work_context = AsyncMock(return_value={
        "viewer": {"id": "hio_user", "name": "Hio"},
        "agentSession": {
            "issue": {
                "id": "issue_1",
                "delegate": {"id": "other_delegate", "name": "Other"},
                "state": {"id": "state_started", "type": "started"},
                "team": {"states": {"nodes": [{"id": "state_started", "position": 1}]}},
            }
        },
    })
    adapter.client.update_issue = AsyncMock()

    await adapter._prepare_issue_for_agent_work("as_123")

    adapter.client.update_issue.assert_not_called()


@pytest.mark.asyncio
async def test_processing_complete_marks_plan_completed():
    adapter = _adapter()
    adapter.client.update_agent_session = AsyncMock(return_value={"id": "as_123", "status": "complete"})
    source = adapter.build_source(
        chat_id="agentSession:as_123",
        chat_name="Linear AgentSession as_123",
        chat_type="dm",
        user_id="lin_user_123",
        user_name="Paul",
    )
    event = MessageEvent(text="done", message_type=MessageType.TEXT, source=source)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter.client.update_agent_session.assert_awaited_once_with(
        agent_session_id="as_123",
        plan=[{"content": "done", "status": "completed"}],
    )


@pytest.mark.asyncio
async def test_send_clarify_creates_elicitation_and_awaiting_plan(monkeypatch):
    adapter = _adapter()
    adapter.client.create_agent_activity = AsyncMock(return_value={"id": "act_elicit"})
    adapter.client.update_agent_session = AsyncMock(return_value={"id": "as_123", "status": "awaitingInput"})
    mark_awaiting_text = MagicMock()
    monkeypatch.setattr("tools.clarify_gateway.mark_awaiting_text", mark_awaiting_text)

    result = await adapter.send_clarify(
        chat_id="agentSession:as_123",
        question="Which slot should I use?",
        choices=["Thursday afternoon", "Friday morning"],
        clarify_id="clarify-1",
        session_key="session-1",
    )

    assert result == SendResult(success=True, message_id="act_elicit")
    mark_awaiting_text.assert_called_once_with("clarify-1")
    adapter.client.create_agent_activity.assert_awaited_once_with(
        agent_session_id="as_123",
        content={"type": "elicitation", "body": "Which slot should I use?"},
        signal="select",
        signal_metadata={
            "options": [
                {"label": "Thursday afternoon", "value": "Thursday afternoon"},
                {"label": "Friday morning", "value": "Friday morning"},
            ]
        },
    )
    adapter.client.update_agent_session.assert_awaited_once_with(
        agent_session_id="as_123",
        plan=[
            {"content": "Which slot should I use?", "status": "inProgress"},
            {"content": "Continue after user response", "status": "pending"},
        ],
    )
    assert set(adapter._awaiting_clarify_sessions) == {"as_123"}
    assert adapter._awaiting_clarify_sessions["as_123"] > linear_adapter.time.time()


@pytest.mark.asyncio
async def test_send_clarify_failure_returns_failed_result(monkeypatch):
    adapter = _adapter()
    adapter.client.create_agent_activity = AsyncMock(side_effect=RuntimeError("Linear rejected activity"))
    adapter.client.update_agent_session = AsyncMock()
    mark_awaiting_text = MagicMock()
    monkeypatch.setattr("tools.clarify_gateway.mark_awaiting_text", mark_awaiting_text)

    result = await adapter.send_clarify(
        chat_id="agentSession:as_123",
        question="Which slot should I use?",
        choices=["Thursday afternoon"],
        clarify_id="clarify-1",
        session_key="session-1",
    )

    assert result.success is False
    assert result.error == "Linear elicitation failed"
    mark_awaiting_text.assert_called_once_with("clarify-1")
    adapter.client.update_agent_session.assert_not_called()
    assert adapter._awaiting_clarify_sessions == {}


@pytest.mark.asyncio
async def test_send_exec_approval_creates_elicitation_select_and_approval_plan():
    adapter = _adapter()
    adapter.client.create_agent_activity = AsyncMock(return_value={"id": "act_approval"})
    adapter.client.update_agent_session = AsyncMock(return_value={"id": "as_123", "status": "awaitingInput"})

    result = await adapter.send_exec_approval(
        chat_id="agentSession:as_123",
        command="rm -rf /tmp/example",
        session_key="session-1",
        description="recursive delete",
    )

    assert result == SendResult(success=True, message_id="act_approval")
    adapter.client.create_agent_activity.assert_awaited_once_with(
        agent_session_id="as_123",
        content={
            "type": "elicitation",
            "body": (
                "Dangerous command requires approval:\n\n"
                "```\nrm -rf /tmp/example\n```\n"
                "Reason: recursive delete"
            ),
        },
        signal="select",
        signal_metadata={
            "options": [
                {"label": "Allow once", "value": "/approve"},
                {"label": "Allow for session", "value": "/approve session"},
                {"label": "Always allow", "value": "/approve always"},
                {"label": "Deny", "value": "/deny"},
            ]
        },
    )
    adapter.client.update_agent_session.assert_awaited_once_with(
        agent_session_id="as_123",
        plan=[
            {"content": "Await command approval", "status": "inProgress"},
            {"content": "recursive delete", "status": "pending"},
        ],
    )


@pytest.mark.asyncio
async def test_linear_client_create_activity_includes_signal_metadata():
    client = LinearClient(token="linear-token")
    client.graphql = AsyncMock(return_value={
        "data": {
            "agentActivityCreate": {
                "success": True,
                "agentActivity": {"id": "act_1"},
            }
        }
    })

    result = await client.create_agent_activity(
        agent_session_id="as_123",
        content={"type": "elicitation", "body": "Pick one"},
        signal="select",
        signal_metadata={"options": [{"label": "A", "value": "a"}]},
    )

    assert result == {"id": "act_1"}
    _, variables = client.graphql.call_args.args
    assert variables["input"]["signal"] == "select"
    assert variables["input"]["signalMetadata"] == {
        "options": [{"label": "A", "value": "a"}]
    }


@pytest.mark.asyncio
async def test_linear_client_update_issue_sets_delegate_and_state():
    client = LinearClient(token="linear-token")
    client.graphql = AsyncMock(return_value={
        "data": {
            "issueUpdate": {
                "success": True,
                "issue": {"id": "issue_1"},
            }
        }
    })

    result = await client.update_issue(
        issue_id="issue_1",
        delegate_id="hio_user",
        state_id="state_started",
    )

    assert result == {"id": "issue_1"}
    _query, variables = client.graphql.call_args.args
    assert variables == {
        "id": "issue_1",
        "input": {"delegateId": "hio_user", "stateId": "state_started"},
    }


@pytest.mark.asyncio
async def test_linear_client_update_issue_skips_empty_update():
    client = LinearClient(token="linear-token")
    client.graphql = AsyncMock()

    result = await client.update_issue(issue_id="issue_1")

    assert result == {}
    client.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_signature_rejected():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    body = json.dumps({"agentSession": {"id": "as_123"}}).encode()
    status, data = adapter._handle_agent_session_body(
        body=body,
        headers={"Linear-Signature": "bad", "Linear-Delivery": "delivery-1"},
        content_length=len(body),
    )
    assert status == 401
    assert data["error"] == "Invalid signature"
    adapter.handle_message.assert_not_called()


def test_invalid_json_rejected_without_agent_run():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    body = b"{not-json"

    status, data = adapter._handle_agent_session_body(
        body=body,
        headers={"Linear-Signature": _signature(body, "linear-secret")},
        content_length=len(body),
    )

    assert status == 400
    assert data["error"] == "Cannot parse body"
    adapter.handle_message.assert_not_called()


def test_non_object_json_rejected_without_agent_run():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    body = b"[]"

    status, data = adapter._handle_agent_session_body(
        body=body,
        headers={"Linear-Signature": _signature(body, "linear-secret")},
        content_length=len(body),
    )

    assert status == 400
    assert data["error"] == "Expected JSON object"
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_nested_session_or_activity_does_not_raise(caplog):
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    payload = {
        "action": "prompted",
        "agentSession": "not-an-object",
        "agentSessionId": "as_123",
        "agentActivity": "not-an-object",
    }

    with caplog.at_level("WARNING"):
        await adapter._process_agent_session_event(payload, "delivery-malformed")

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "agentSession:as_123"
    assert event.text == "Linear AgentSession prompt."


def test_payload_too_large_rejected_before_signature():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    status, data = adapter._handle_agent_session_body(
        body=b"{}",
        headers={},
        content_length=1_048_577,
    )

    assert status == 413
    assert data["error"] == "Payload too large"
    adapter.handle_message.assert_not_called()


def test_payload_too_large_rejected_when_content_length_missing():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    body = b"x" * (linear_adapter.MAX_WEBHOOK_BODY_BYTES + 1)

    status, data = adapter._handle_agent_session_body(
        body=body,
        headers={},
        content_length=None,
    )

    assert status == 413
    assert data["error"] == "Payload too large"
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_rejects_oversized_content_length_before_read():
    adapter = _adapter()

    class _Request:
        content_length = linear_adapter.MAX_WEBHOOK_BODY_BYTES + 1
        headers = {}

        async def read(self):
            raise AssertionError("oversized request body should not be read")

    response = await adapter._handle_agent_session_webhook(_Request())

    assert response.status == 413


@pytest.mark.asyncio
async def test_background_task_failure_is_logged_and_consumed(caplog):
    payload = {"action": "prompted", "agentSession": {"id": "as_123"}, "agentActivity": {"body": "hello"}}
    body = json.dumps(payload).encode()
    adapter = _adapter()

    async def _boom(event):
        raise RuntimeError("handler exploded")

    adapter.handle_message = _boom
    headers = {"Linear-Signature": _signature(body, "linear-secret"), "Linear-Delivery": "delivery-fail"}

    with caplog.at_level("WARNING"):
        status, data = adapter._handle_agent_session_body(body=body, headers=headers, content_length=len(body))
        await adapter.wait_background_tasks()

    assert status == 200
    assert data["status"] == "accepted"
    assert "AgentSession background task failed delivery=delivery-fail" in caplog.text
    assert "handler exploded" in caplog.text


@pytest.mark.asyncio
async def test_missing_delivery_id_falls_back_to_stable_body_hash():
    payload = {"action": "prompted", "agentSession": {"id": "as_123"}, "agentActivity": {"body": "hello"}}
    body = json.dumps(payload, sort_keys=True).encode()
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    headers = {"Linear-Signature": _signature(body, "linear-secret")}

    status1, data1 = adapter._handle_agent_session_body(body=body, headers=headers, content_length=None)
    status2, data2 = adapter._handle_agent_session_body(body=body, headers=headers, content_length=None)

    expected_delivery_id = f"body-sha256:{hashlib.sha256(body).hexdigest()}"
    assert status1 == 200
    assert data1 == {"status": "accepted", "delivery_id": expected_delivery_id}
    assert status2 == 200
    assert data2 == {"status": "duplicate", "delivery_id": expected_delivery_id}
    await adapter.wait_background_tasks()
    adapter.handle_message.assert_awaited_once()
