"""Tests for the AAMP gateway adapter."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.aamp import (
    AampAdapter,
    AampIdentity,
    load_aamp_sender_policies,
    match_aamp_sender_policy,
    send_aamp_direct,
)


def _config(**extra):
    return PlatformConfig(
        enabled=True,
        extra={
            "base_url": "https://meshmail.ai",
            "slug": "hermes",
            **extra,
        },
    )


class _FakeSdkClient:
    def __init__(self, *, send_task_result=("task-new", "msg-new")):
        self.handlers = {}
        self.connect_called = False
        self.disconnect_called = False
        self.reconcile_calls = []
        self.send_task_calls = []
        self.send_result_calls = []
        self.send_task_result = send_task_result

    def on(self, event_name, handler):
        self.handlers[event_name] = handler

    def connect(self):
        self.connect_called = True

    def disconnect(self):
        self.disconnect_called = True

    def reconcile_recent_emails(self, limit):
        self.reconcile_calls.append(limit)
        return 0

    def is_using_polling_fallback(self):
        return False

    def send_task(self, **kwargs):
        self.send_task_calls.append(kwargs)
        return self.send_task_result

    def send_result(self, **kwargs):
        self.send_result_calls.append(kwargs)


class TestAampPlatformEnum:
    def test_aamp_enum_exists(self):
        assert Platform.AAMP.value == "aamp"

    def test_aamp_capabilities_disable_streaming_behaviors(self):
        assert AampAdapter.SUPPORTS_MESSAGE_EDITING is False
        assert AampAdapter.SUPPORTS_STREAMING is False
        assert AampAdapter.SUPPORTS_INTERIM_MESSAGES is False
        assert AampAdapter.SUPPORTS_TOOL_PROGRESS is False


class TestAampConfigLoading:
    def test_apply_env_overrides_aamp(self, monkeypatch):
        monkeypatch.setenv("AAMP_BASE_URL", "meshmail.ai")
        monkeypatch.setenv("AAMP_POLL_INTERVAL", "7")
        monkeypatch.setenv("AAMP_REJECT_UNAUTHORIZED", "false")
        monkeypatch.setenv(
            "AAMP_SENDER_POLICIES",
            '[{"sender":"dispatch@example.com","dispatchContextRules":{"tenant":["acme"]}}]',
        )

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.AAMP in config.platforms
        aamp = config.platforms[Platform.AAMP]
        assert aamp.enabled is True
        assert aamp.extra["base_url"] == "http://meshmail.ai"
        assert aamp.extra["slug"] == "hermes"
        assert aamp.extra["poll_interval"] == 7
        assert aamp.extra["reject_unauthorized"] is False
        assert aamp.extra["sender_policies"] == '[{"sender":"dispatch@example.com","dispatchContextRules":{"tenant":["acme"]}}]'

    def test_home_channel_set_from_env(self, monkeypatch):
        monkeypatch.setenv("AAMP_BASE_URL", "https://meshmail.ai")
        monkeypatch.setenv("AAMP_HOME_CHANNEL", "peer@example.com")

        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.platforms[Platform.AAMP].home_channel
        assert home is not None
        assert home.chat_id == "peer@example.com"

    def test_connected_platforms_include_aamp(self, monkeypatch):
        monkeypatch.setenv("AAMP_BASE_URL", "https://meshmail.ai")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.AAMP in config.get_connected_platforms()

    def test_default_base_url_used_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("AAMP_BASE_URL", raising=False)
        monkeypatch.delenv("AAMP_HOST", raising=False)
        monkeypatch.setenv("AAMP_EMAIL", "hermes@example.com")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert config.platforms[Platform.AAMP].extra["base_url"] == "https://meshmail.ai"
        assert config.platforms[Platform.AAMP].extra["slug"] == "hermes"

    def test_cached_identity_enables_aamp_without_base_url_env(self, monkeypatch, tmp_path):
        credentials_file = tmp_path / "mailbox_identity.json"
        credentials_file.write_text(
            json.dumps(
                {
                    "base_url": "http://cached.example",
                    "email": "hermes@example.com",
                    "mailbox_token": "token-123",
                    "smtp_password": "secret",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.delenv("AAMP_BASE_URL", raising=False)
        monkeypatch.delenv("AAMP_HOST", raising=False)
        monkeypatch.delenv("AAMP_SLUG", raising=False)
        monkeypatch.setenv("AAMP_CREDENTIALS_FILE", str(credentials_file))

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.AAMP in config.platforms
        assert config.platforms[Platform.AAMP].extra["base_url"] == "http://cached.example"
        assert Platform.AAMP in config.get_connected_platforms()

    def test_invalid_poll_interval_does_not_crash(self, monkeypatch):
        monkeypatch.setenv("AAMP_BASE_URL", "http://localhost:3000")
        monkeypatch.setenv("AAMP_POLL_INTERVAL", "0.5s")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert config.platforms[Platform.AAMP].extra["poll_interval"] == 10


class TestAampEventConstruction:
    def test_build_event_uses_sender_and_task_id(self):
        adapter = AampAdapter(_config())

        event = adapter._build_event(
            {
                "taskId": "task-123",
                "from": "peer@example.com",
                "messageId": "msg-123",
                "title": "Review this diff",
                "bodyText": "Please review before lunch.",
                "contextLinks": ["https://example.com/pr/1"],
            }
        )

        assert event is not None
        assert event.source.chat_id == "peer@example.com"
        assert event.source.user_id == "peer@example.com"
        assert event.source.thread_id == "task-123"
        assert event.message_id == "task-123"
        assert "Review this diff" in event.text
        assert "Please review before lunch." in event.text
        assert "https://example.com/pr/1" in event.text

    def test_build_event_accepts_attachment_only_tasks(self):
        adapter = AampAdapter(_config())

        event = adapter._build_event(
            {
                "taskId": "task-attachments",
                "from": "peer@example.com",
                "attachments": [{"filename": "spec.pdf"}],
            }
        )

        assert event is not None
        assert event.source.thread_id == "task-attachments"
        assert "Attachments:\nspec.pdf" in event.text

    def test_build_event_accepts_context_only_tasks(self):
        adapter = AampAdapter(_config())

        event = adapter._build_event(
            {
                "taskId": "task-context",
                "from": "peer@example.com",
                "contextLinks": ["https://example.com/spec"],
            }
        )

        assert event is not None
        assert event.source.thread_id == "task-context"
        assert "Context:\nhttps://example.com/spec" in event.text


class TestAampSenderPolicies:
    def test_load_sender_policies_from_config(self):
        policies = load_aamp_sender_policies(
            _config(
                sender_policies=[
                    {
                        "sender": "Dispatch@Example.com",
                        "dispatchContextRules": {
                            "tenant": ["acme"],
                            "workflow": ["prod", "staging"],
                        },
                    }
                ]
            )
        )

        assert len(policies) == 1
        assert policies[0].sender == "dispatch@example.com"
        assert policies[0].dispatch_context_rules == {
            "tenant": ["acme"],
            "workflow": ["prod", "staging"],
        }

    def test_match_sender_policy_allows_matching_dispatch_context(self):
        policies = load_aamp_sender_policies(
            _config(
                sender_policies=[
                    {
                        "sender": "dispatch@example.com",
                        "dispatchContextRules": {"tenant": ["acme"], "workflow": ["prod"]},
                    }
                ]
            )
        )

        result = match_aamp_sender_policy(
            {
                "from": "dispatch@example.com",
                "dispatchContext": {"tenant": "acme", "workflow": "prod"},
            },
            policies,
        )

        assert result is not None
        assert result.allowed is True
        assert result.reason is None

    def test_match_sender_policy_rejects_missing_sender(self):
        policies = load_aamp_sender_policies(
            _config(sender_policies=[{"sender": "dispatch@example.com"}])
        )

        result = match_aamp_sender_policy(
            {"from": "other@example.com"},
            policies,
        )

        assert result is not None
        assert result.allowed is False
        assert "other@example.com" in (result.reason or "")

    def test_match_sender_policy_rejects_missing_dispatch_context_key(self):
        policies = load_aamp_sender_policies(
            _config(
                sender_policies=[
                    {
                        "sender": "dispatch@example.com",
                        "dispatchContextRules": {"tenant": ["acme"]},
                    }
                ]
            )
        )

        result = match_aamp_sender_policy(
            {"from": "dispatch@example.com", "dispatchContext": {}},
            policies,
        )

        assert result is not None
        assert result.allowed is False
        assert result.reason == 'dispatchContext missing required key "tenant"'

    def test_match_sender_policy_rejects_disallowed_dispatch_context_value(self):
        policies = load_aamp_sender_policies(
            _config(
                sender_policies=[
                    {
                        "sender": "dispatch@example.com",
                        "dispatchContextRules": {"tenant": ["acme"]},
                    }
                ]
            )
        )

        result = match_aamp_sender_policy(
            {"from": "dispatch@example.com", "dispatchContext": {"tenant": "globex"}},
            policies,
        )

        assert result is not None
        assert result.allowed is False
        assert result.reason == "dispatchContext tenant=globex is not allowed"


class TestAampConnect:
    @pytest.mark.asyncio
    async def test_connect_uses_sdk_and_dispatches_task_events(self, monkeypatch):
        adapter = AampAdapter(_config())
        fake_client = _FakeSdkClient()
        seen = []

        async def fake_handle(event):
            seen.append(event)

        monkeypatch.setattr(
            "gateway.platforms.aamp.resolve_aamp_identity",
            AsyncMock(
                return_value=(
                    AampIdentity(
                        base_url="https://meshmail.ai",
                        email="hermes@example.com",
                        mailbox_token="token-123",
                        smtp_password="secret",
                    ),
                    adapter._credentials_path,
                )
            ),
        )
        monkeypatch.setattr("gateway.platforms.aamp._build_sdk_client", lambda config, identity: fake_client)
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args, **kwargs: True)
        monkeypatch.setattr(adapter, "handle_message", fake_handle)

        connected = await adapter.connect()
        assert connected is True
        assert fake_client.connect_called is True
        assert fake_client.reconcile_calls == [10]

        fake_client.handlers["task.dispatch"](
            {
                "taskId": "task-123",
                "from": "peer@example.com",
                "messageId": "msg-123",
                "title": "Review this diff",
                "bodyText": "Please review before lunch.",
            }
        )
        await asyncio.sleep(0.05)

        assert len(seen) == 1
        assert seen[0].source.thread_id == "task-123"
        assert "Review this diff" in seen[0].text

        await adapter.disconnect()
        assert fake_client.disconnect_called is True


class TestAampSend:
    @pytest.mark.asyncio
    async def test_send_delegates_to_direct_sender(self, monkeypatch):
        adapter = AampAdapter(_config())

        async def fake_send(config, chat_id, content, *, reply_to=None, metadata=None):
            assert chat_id == "peer@example.com"
            assert content == "Done"
            assert reply_to == "task-123"
            assert metadata == {"thread_id": "task-123"}
            return {
                "success": True,
                "platform": "aamp",
                "chat_id": chat_id,
                "message_id": "msg-1",
            }

        monkeypatch.setattr("gateway.platforms.aamp.send_aamp_direct", fake_send)

        result = await adapter.send(
            "peer@example.com",
            "Done",
            reply_to="task-123",
            metadata={"thread_id": "task-123"},
        )

        assert result.success is True
        assert result.message_id == "msg-1"


class TestAampDirectSend:
    @pytest.mark.asyncio
    async def test_direct_send_uses_task_result_for_threaded_replies(self, monkeypatch):
        fake_client = _FakeSdkClient()

        monkeypatch.setattr(
            "gateway.platforms.aamp.resolve_aamp_identity",
            AsyncMock(
                return_value=(
                    AampIdentity(
                        base_url="https://meshmail.ai",
                        email="hermes@example.com",
                        mailbox_token="token-123",
                        smtp_password="secret",
                    ),
                    adapter_path := AampAdapter(_config())._credentials_path,
                )
            ),
        )
        monkeypatch.setattr("gateway.platforms.aamp._build_sdk_client", lambda config, identity: fake_client)

        result = await send_aamp_direct(
            _config(),
            "peer@example.com",
            "**Done**",
            metadata={"thread_id": "task-123"},
        )

        assert result["success"] is True
        assert result["intent"] == "task.result"
        assert result["task_id"] == "task-123"
        assert fake_client.send_result_calls == [
            {
                "to": "peer@example.com",
                "task_id": "task-123",
                "status": "completed",
                "output": "Done",
                "error_msg": None,
                "structured_result": None,
                "in_reply_to": None,
            }
        ]
        assert adapter_path.name == "mailbox_identity.json"

    @pytest.mark.asyncio
    async def test_direct_send_supports_rejected_task_results(self, monkeypatch):
        fake_client = _FakeSdkClient()

        monkeypatch.setattr(
            "gateway.platforms.aamp.resolve_aamp_identity",
            AsyncMock(
                return_value=(
                    AampIdentity(
                        base_url="https://meshmail.ai",
                        email="hermes@example.com",
                        mailbox_token="token-123",
                        smtp_password="secret",
                    ),
                    AampAdapter(_config())._credentials_path,
                )
            ),
        )
        monkeypatch.setattr("gateway.platforms.aamp._build_sdk_client", lambda config, identity: fake_client)

        result = await send_aamp_direct(
            _config(),
            "peer@example.com",
            "Request rejected: sender dispatch@example.com is not allowed",
            metadata={
                "thread_id": "task-123",
                "aamp_message_id": "msg-123",
                "aamp_status": "rejected",
                "aamp_error_msg": "sender dispatch@example.com is not allowed",
            },
        )

        assert result["success"] is True
        assert result["intent"] == "task.result"
        assert result["status"] == "rejected"
        assert fake_client.send_result_calls == [
            {
                "to": "peer@example.com",
                "task_id": "task-123",
                "status": "rejected",
                "output": "Request rejected: sender dispatch@example.com is not allowed",
                "error_msg": "sender dispatch@example.com is not allowed",
                "structured_result": None,
                "in_reply_to": "msg-123",
            }
        ]

    @pytest.mark.asyncio
    async def test_direct_send_without_thread_starts_new_task_dispatch(self, monkeypatch):
        fake_client = _FakeSdkClient(send_task_result=("task-plain", "msg-plain"))

        monkeypatch.setattr(
            "gateway.platforms.aamp.resolve_aamp_identity",
            AsyncMock(
                return_value=(
                    AampIdentity(
                        base_url="https://meshmail.ai",
                        email="hermes@example.com",
                        mailbox_token="token-123",
                        smtp_password="secret",
                    ),
                    AampAdapter(_config())._credentials_path,
                )
            ),
        )
        monkeypatch.setattr("gateway.platforms.aamp._build_sdk_client", lambda config, identity: fake_client)

        result = await send_aamp_direct(_config(), "peer@example.com", "Notification only")

        assert result["success"] is True
        assert result["intent"] == "task.dispatch"
        assert result["task_id"] == "task-plain"
        assert result["message_id"] == "msg-plain"
        assert fake_client.send_task_calls == [
            {
                "to": "peer@example.com",
                "title": "Notification only",
                "body_text": "Notification only",
            }
        ]


class TestAampTooling:
    def test_toolset_exists(self):
        from toolsets import TOOLSETS

        assert "hermes-aamp" in TOOLSETS
        assert "hermes-aamp" in TOOLSETS["hermes-gateway"]["includes"]

    def test_platform_hint_exists(self):
        from agent.prompt_builder import PLATFORM_HINTS

        assert "aamp" in PLATFORM_HINTS
        assert "asynchronous" in PLATFORM_HINTS["aamp"]
