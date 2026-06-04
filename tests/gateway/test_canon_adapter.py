"""Tests for the Canon platform adapter plugin."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_canon = load_plugin_adapter("canon")

CanonAdapter = _canon.CanonAdapter
CanonHttpClient = _canon.CanonHttpClient
CanonStreamFrame = _canon.CanonStreamFrame
DEFAULT_BASE_URL = _canon.DEFAULT_BASE_URL
DEFAULT_STREAM_URL = _canon.DEFAULT_STREAM_URL
TURN_COMPLETE_METADATA = _canon.TURN_COMPLETE_METADATA
_env_enablement = _canon._env_enablement
_parse_sse_frame = _canon._parse_sse_frame
_profile_slug = _canon._profile_slug
_save_canon_profile = _canon._save_canon_profile
_standalone_send = _canon._standalone_send
_resolve_canon_agent = _canon._resolve_canon_agent
_wait_for_registration_approval = _canon._wait_for_registration_approval
_canon_timeout_seconds = _canon._canon_timeout_seconds
check_requirements = _canon.check_requirements
register = _canon.register
validate_config = _canon.validate_config


def _config(**kwargs):
    from gateway.config import PlatformConfig

    return PlatformConfig(enabled=True, **kwargs)


async def _wait_for(predicate, *, interval: float = 0.01):
    while not predicate():
        await asyncio.sleep(interval)


class TestCanonConfig:
    def test_init_from_config_extra(self, monkeypatch):
        monkeypatch.delenv("CANON_API_KEY", raising=False)
        cfg = _config(
            extra={
                "api_key": "config-key",
                "base_url": "https://api.example.test",
                "stream_url": "https://stream.example.test",
                "history_limit": "25",
            }
        )

        adapter = CanonAdapter(cfg)

        assert adapter.api_key == "config-key"
        assert adapter.base_url == "https://api.example.test"
        assert adapter.stream_url == "https://stream.example.test"
        assert adapter.history_limit == 25

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("CANON_API_KEY", "env-key")
        monkeypatch.setenv("CANON_BASE_URL", "https://api.env.test")

        adapter = CanonAdapter(
            _config(api_key="config-key", extra={"base_url": "https://api.config.test"})
        )

        assert adapter.api_key == "env-key"
        assert adapter.base_url == "https://api.env.test"

    def test_init_from_canon_agent_profile_bootstrap(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CANON_API_KEY", raising=False)
        monkeypatch.setenv("CANON_HOME", str(tmp_path))
        monkeypatch.setenv("CANON_AGENT", "leonardo-2")
        monkeypatch.setenv(
            "CANON_AGENTS_JSON_BOOTSTRAP",
            json.dumps(
                {
                    "leonardo-2": {
                        "apiKey": "profile-key",
                        "agentId": "agent-leonardo",
                        "agentName": "Leonardo 2",
                        "clientType": "hermes",
                        "baseUrl": "https://api.profile.test",
                        "streamUrl": "https://stream.profile.test",
                    }
                }
            ),
        )

        adapter = CanonAdapter(_config())

        assert adapter.api_key == "profile-key"
        assert adapter.profile_name == "leonardo-2"
        assert adapter.profile_agent_id == "agent-leonardo"
        assert adapter.base_url == "https://api.profile.test"
        assert adapter.stream_url == "https://stream.profile.test"
        assert (tmp_path / "agents.json").exists()

    def test_direct_key_precedes_malformed_profile_bootstrap(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CANON_HOME", str(tmp_path))
        monkeypatch.setenv("CANON_API_KEY", "env-key")
        monkeypatch.setenv("CANON_AGENT", "leonardo-2")
        monkeypatch.setenv("CANON_AGENTS_JSON_BOOTSTRAP", "not-json")

        resolved = _resolve_canon_agent(_config())

        assert resolved.api_key == "env-key"
        assert resolved.profile is None

    def test_validate_config_accepts_env_config_or_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CANON_HOME", str(tmp_path))
        monkeypatch.delenv("CANON_API_KEY", raising=False)
        monkeypatch.delenv("CANON_AGENT", raising=False)
        monkeypatch.delenv("CANON_AGENTS_JSON_BOOTSTRAP", raising=False)
        assert not validate_config(_config())
        assert validate_config(_config(api_key="config-key"))

        monkeypatch.setenv("CANON_API_KEY", "env-key")
        assert validate_config(_config())

        monkeypatch.delenv("CANON_API_KEY", raising=False)
        monkeypatch.setenv("CANON_AGENT", "leonardo-2")
        monkeypatch.setenv(
            "CANON_AGENTS_JSON_BOOTSTRAP",
            json.dumps(
                {
                    "leonardo-2": {
                        "apiKey": "profile-key",
                        "agentId": "agent-leonardo",
                        "agentName": "Leonardo 2",
                        "clientType": "hermes",
                    }
                }
            ),
        )
        assert validate_config(_config())

    def test_env_enablement_seeds_extra_and_home_channel(self, monkeypatch):
        monkeypatch.setenv("CANON_API_KEY", "env-key")
        monkeypatch.setenv("CANON_BASE_URL", "https://api.env.test")
        monkeypatch.setenv("CANON_STREAM_URL", "https://stream.env.test")
        monkeypatch.setenv("CANON_HOME_CHANNEL", "convo-home")
        monkeypatch.setenv("CANON_HISTORY_LIMIT", "75")

        seed = _env_enablement()

        assert seed["api_key"] == "env-key"
        assert seed["base_url"] == "https://api.env.test"
        assert seed["stream_url"] == "https://stream.env.test"
        assert seed["history_limit"] == "75"
        assert seed["home_channel"]["chat_id"] == "convo-home"

    def test_requirements_follow_canon_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CANON_HOME", str(tmp_path))
        monkeypatch.delenv("CANON_API_KEY", raising=False)
        monkeypatch.delenv("CANON_AGENT", raising=False)
        monkeypatch.delenv("CANON_AGENTS_JSON_BOOTSTRAP", raising=False)
        assert check_requirements() is False

        monkeypatch.setenv("CANON_API_KEY", "env-key")
        assert check_requirements() is True

    def test_runtime_hitl_timeout_matches_canon_api_limit(self, monkeypatch):
        monkeypatch.setenv("HERMES_CLARIFY_TIMEOUT", "3600")

        assert _canon_timeout_seconds("HERMES_CLARIFY_TIMEOUT", 300) == 1800

    def test_profile_slug_normalizes_display_name(self):
        assert _profile_slug("My Hermes Agent!") == "my-hermes-agent"
        assert _profile_slug("!!!") == "hermes"

    def test_save_canon_profile_writes_resolvable_agents_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CANON_HOME", str(tmp_path))
        monkeypatch.delenv("CANON_API_KEY", raising=False)
        monkeypatch.setenv("CANON_AGENT", "my-hermes")

        _save_canon_profile(
            "my-hermes",
            api_key="profile-key",
            agent_id="agent-1",
            agent_name="My Hermes",
            base_url="https://api.example.test",
        )

        resolved = _resolve_canon_agent(_config())

        assert resolved.api_key == "profile-key"
        assert resolved.profile == "my-hermes"
        assert resolved.agent_id == "agent-1"
        assert resolved.base_url == "https://api.example.test"

    def test_wait_for_registration_approval_returns_approved(self, monkeypatch):
        statuses = [
            {"status": "pending"},
            {
                "status": "approved",
                "apiKey": "agk_live_test",
                "agentId": "agent-1",
            },
        ]
        seen = []

        def fake_status(**_kwargs):
            return statuses.pop(0)

        monkeypatch.setattr(_canon, "_get_registration_status", fake_status)
        monkeypatch.setattr(_canon.time, "sleep", lambda _seconds: None)

        result = _wait_for_registration_approval(
            base_url=DEFAULT_BASE_URL,
            request_id="req-1",
            poll_token="poll-1",
            on_poll=seen.append,
            timeout_seconds=10,
            poll_interval_seconds=0.01,
        )

        assert result["status"] == "approved"
        assert result["apiKey"] == "agk_live_test"
        assert [item["status"] for item in seen] == ["pending", "approved"]


class TestCanonHttpClient:
    @pytest.mark.asyncio
    async def test_download_media_blocks_unsafe_url(self, monkeypatch):
        client = CanonHttpClient("key")
        client._client.get = AsyncMock()
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _url: False)

        try:
            with pytest.raises(ValueError, match="SSRF"):
                await client.download_media("http://127.0.0.1/private")

            client._client.get.assert_not_called()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_download_media_uses_redirect_guard(self, monkeypatch):
        client = CanonHttpClient("key")
        monkeypatch.setattr(
            "tools.url_safety.is_safe_url",
            lambda url: url == "https://media.example/voice.m4a",
        )
        redirect_response = SimpleNamespace(
            is_redirect=True,
            next_request=SimpleNamespace(
                url="http://169.254.169.254/latest/meta-data"
            ),
        )

        try:
            hooks = client._client.event_hooks["response"]
            assert hooks
            with pytest.raises(ValueError, match="Blocked redirect"):
                await hooks[0](redirect_response)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_download_media_returns_bytes_for_safe_url(self, monkeypatch):
        response = SimpleNamespace(
            status_code=200,
            content=b"audio-bytes",
            text="",
            headers={"content-type": "audio/mp4"},
        )
        client = CanonHttpClient("key")
        client._client.get = AsyncMock(return_value=response)
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _url: True)

        try:
            data, content_type = await client.download_media(
                "https://media.example/voice.m4a"
            )

            assert data == b"audio-bytes"
            assert content_type == "audio/mp4"
            client._client.get.assert_awaited_once_with(
                "https://media.example/voice.m4a",
                headers={"User-Agent": "HermesAgent/CanonPlatform"},
                follow_redirects=True,
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_runtime_card_and_status_endpoints_use_agent_api(self):
        client = CanonHttpClient("key")
        calls = []

        async def fake_request(method, path, *, params=None, json_body=None):
            calls.append((method, path, params, json_body))
            if path == "/runtime-input/request":
                return {"success": True, "messageId": "input-card"}
            if path == "/runtime-card/request":
                return {"success": True, "messageId": "runtime-card", "cardId": "card-1"}
            if path == "/runtime-approval/request":
                return {"success": True, "messageId": "approval-card"}
            if path == "/runtime-input/consume":
                return {"status": "submitted", "inputId": "input-1", "answers": {"field": {"answers": ["supplier"]}}}
            if path == "/runtime-card/consume":
                return {"status": "submitted", "cardId": "card-1", "actionId": "approve"}
            if path == "/runtime-approval/consume":
                return {"status": "pending", "approvalId": "approval-1"}
            if path == "/runtime/signal/consume":
                return {"status": "none"}
            return {"ok": True}

        client._request_json = fake_request

        try:
            await client.update_runtime_status(
                runtime="hermes",
                runtime_descriptor={"coreControls": []},
            )
            await client.update_runtime_turn(
                "convo-1",
                state="thinking",
                turn_id="turn-1",
                queue_depth=1,
                capabilities={"supportsQueue": True},
            )
            await client.update_message_disposition(
                "convo-1",
                "msg-1",
                "accepted_now",
            )
            await client.mark_as_read("convo-1")
            await client.create_runtime_input_request(
                "convo-1",
                input_id="input-1",
                kind="clarify",
                expires_at=1234,
                prompt="Pick one",
                choices=[{"label": "A", "value": "a"}],
                questions=[{"id": "field", "question": "Which field?"}],
                response_user_id="owner-1",
                turn_id="turn-1",
            )
            await client.consume_runtime_input_response("convo-1", "input-1")
            await client.create_runtime_card_request(
                "convo-1",
                card={
                    "schema": "canon.card.v1",
                    "title": "Review",
                    "fallbackText": "Review",
                    "blocks": [{"kind": "actions", "actions": [{"id": "approve", "label": "Approve"}]}],
                },
                card_id="card-1",
                expires_at=1234,
                response_user_id="owner-1",
                runtime_id="hermes",
                turn_id="turn-1",
            )
            await client.consume_runtime_card_response("convo-1", "card-1")
            await client.create_runtime_approval_request(
                "convo-1",
                approval_id="approval-1",
                tool_name="Command",
                tool_summary="Run command",
                expires_at=1234,
            )
            await client.consume_runtime_approval_response("convo-1", "approval-1")
            await client.consume_runtime_signal("convo-1")
        finally:
            await client.close()

        assert [call[1] for call in calls] == [
            "/runtime/status",
            "/runtime/turn",
            "/conversations/convo-1/messages/msg-1/disposition",
            "/conversations/convo-1/read",
            "/runtime-input/request",
            "/runtime-input/consume",
            "/runtime-card/request",
            "/runtime-card/consume",
            "/runtime-approval/request",
            "/runtime-approval/consume",
            "/runtime/signal/consume",
        ]
        assert calls[1][3] == {
            "conversationId": "convo-1",
            "state": "thinking",
            "queueDepth": 1,
            "turnId": "turn-1",
            "capabilities": {"supportsQueue": True},
        }
        assert calls[2][3] == {"inboundDisposition": "accepted_now"}
        assert calls[3][3] is None
        assert calls[4][3] == {
            "conversationId": "convo-1",
            "inputId": "input-1",
            "kind": "clarify",
            "expiresAt": 1234,
            "prompt": "Pick one",
            "choices": [{"label": "A", "value": "a"}],
            "questions": [{"id": "field", "question": "Which field?"}],
            "responseUserId": "owner-1",
            "turnId": "turn-1",
        }
        assert calls[6][3] == {
            "conversationId": "convo-1",
            "cardId": "card-1",
            "card": {
                "schema": "canon.card.v1",
                "title": "Review",
                "fallbackText": "Review",
                "blocks": [{"kind": "actions", "actions": [{"id": "approve", "label": "Approve"}]}],
            },
            "expiresAt": 1234,
            "responseUserId": "owner-1",
            "runtimeId": "hermes",
            "turnId": "turn-1",
        }


class TestCanonInbound:
    def test_message_text_includes_rich_contact_card_identity(self):
        text = _canon._message_text(
            {
                "contentType": "contact_card",
                "text": "Please meet Moty.",
                "contactCard": {
                    "userId": "agent-moty",
                    "displayName": "Moty",
                    "userType": "ai_agent",
                    "ownerName": "Matan",
                    "about": "Priority invoice review agent",
                },
            }
        )

        assert text == (
            '[Contact card] "Moty" - ai_agent; userId: agent-moty; '
            "owner: Matan; about: Priority invoice review agent\n"
            "Please meet Moty."
        )

    class FakeClient:
        def __init__(self, *, fail_read: bool = False):
            self.dispositions = []
            self.reads = []
            self.fail_read = fail_read

        async def update_message_disposition(
            self,
            conversation_id,
            message_id,
            inbound_disposition,
        ):
            self.dispositions.append((conversation_id, message_id, inbound_disposition))

        async def mark_as_read(self, conversation_id):
            if self.fail_read:
                raise RuntimeError("read failed")
            self.reads.append(conversation_id)

    @staticmethod
    def _message_event(*, message_id="msg-1", metadata=None):
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.session import SessionSource

        return MessageEvent(
            text="hello",
            source=SessionSource(
                platform=Platform("canon"),
                chat_id="convo-1",
                user_id="human-1",
            ),
            message_id=message_id,
            raw_message={"message": {"metadata": metadata or {}}},
        )

    @staticmethod
    def _gateway_runner(adapter, *, authorized=True):
        from gateway.config import GatewayConfig, PlatformConfig
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={adapter.platform: PlatformConfig(enabled=True)}
        )
        runner.adapters = {adapter.platform: adapter}
        runner.session_store = None
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        runner.pairing_store._is_rate_limited.return_value = False
        runner._running_agents = {}
        runner._running_agents_ts = {}
        runner._pending_messages = {}
        runner._queued_events = {}
        runner._busy_ack_ts = {}
        runner._update_prompt_pending = {}
        runner._session_run_generation = {}
        runner._draining = False
        runner._is_user_authorized = MagicMock(return_value=authorized)
        runner._get_unauthorized_dm_behavior = MagicMock(return_value="ignore")
        runner._handle_message_with_agent = AsyncMock(
            return_value={"final_response": "", "messages": []}
        )
        return runner

    @pytest.mark.asyncio
    async def test_message_created_normalizes_to_message_event(self):
        from gateway.platforms.base import MessageType

        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._agent_id = "agent-1"
        adapter._conversation_cache["convo-1"] = {
            "id": "convo-1",
            "type": "group",
            "name": "Research",
        }
        adapter.handle_message = AsyncMock()

        await adapter._handle_message_payload({
            "conversationId": "convo-1",
            "message": {
                "id": "msg-1",
                "senderId": "human-1",
                "senderName": "Ada",
                "text": "/status please",
                "contentType": "text",
                "replyTo": "msg-parent",
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/status please"
        assert event.message_type == MessageType.COMMAND
        assert event.message_id == "msg-1"
        assert event.reply_to_message_id == "msg-parent"
        assert event.source.platform.value == "canon"
        assert event.source.chat_id == "convo-1"
        assert event.source.chat_type == "group"
        assert event.source.chat_name == "Research"
        assert event.source.user_id == "human-1"
        assert event.source.user_name == "Ada"
        assert "did not structurally mention" in event.channel_prompt

    @pytest.mark.asyncio
    async def test_group_mentions_surface_in_channel_prompt(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._agent_id = "agent-1"
        adapter._conversation_cache["convo-1"] = {
            "id": "convo-1",
            "type": "group",
            "name": "Research",
        }
        adapter.handle_message = AsyncMock()

        await adapter._handle_message_payload({
            "conversationId": "convo-1",
            "message": {
                "id": "msg-mention",
                "senderId": "human-1",
                "senderName": "Ada",
                "mentions": ["agent-1"],
                "text": "@Leonardo can you check this?",
            },
        })

        event = adapter.handle_message.await_args.args[0]
        assert "explicitly mentioned this agent" in event.channel_prompt

    @pytest.mark.asyncio
    async def test_ignores_self_messages_and_duplicates(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._agent_id = "agent-1"
        fake = self.FakeClient()
        adapter._client = fake
        adapter.handle_message = AsyncMock()

        payload = {
            "conversationId": "convo-1",
            "message": {
                "id": "msg-1",
                "senderId": "agent-1",
                "text": "own echo",
                "contentType": "text",
            },
        }
        await adapter._handle_message_payload(payload)
        adapter.handle_message.assert_not_awaited()

        payload["message"]["senderId"] = "human-1"
        await adapter._handle_message_payload(payload)
        await adapter._handle_message_payload(payload)
        adapter.handle_message.assert_awaited_once()
        assert fake.reads == []

    @pytest.mark.asyncio
    async def test_attachment_only_message_gets_text_placeholder(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter.handle_message = AsyncMock()

        await adapter._handle_message_payload({
            "conversationId": "convo-1",
            "message": {
                "id": "msg-2",
                "senderId": "human-1",
                "contentType": "image",
                "attachments": [{"kind": "image", "fileName": "plot.png"}],
            },
        })

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "[image attachment: plot.png]"
        assert event.source.chat_type == "dm"

    @pytest.mark.asyncio
    async def test_runtime_control_messages_do_not_dispatch_as_user_prompts(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        adapter.handle_message = AsyncMock()

        await adapter._handle_message_payload({
            "conversationId": "convo-1",
            "message": {
                "id": "msg-control",
                "senderId": "owner-1",
                "text": "Approved.",
                "metadata": {
                    "type": "approval_reply",
                    "approvalId": "approval-1",
                    "decision": "allow",
                },
            },
        })

        adapter.handle_message.assert_not_awaited()
        assert fake.reads == []

    @pytest.mark.asyncio
    async def test_runtime_signal_dispatches_hermes_session_command(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._agent_id = "agent-1"
        adapter._conversation_cache["convo-1"] = {
            "id": "convo-1",
            "type": "dm",
            "displayName": "Matan",
        }
        adapter.handle_message = AsyncMock()

        await adapter._handle_runtime_signal("convo-1", "new_session")

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/new"
        assert event.source.chat_id == "convo-1"
        assert event.source.user_id == "agent-1"

    @pytest.mark.asyncio
    async def test_runtime_signal_uses_native_control_handler_when_available(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._agent_id = "agent-1"
        adapter._conversation_cache["convo-1"] = {"id": "convo-1", "type": "dm"}
        adapter.handle_message = AsyncMock()
        adapter._publish_runtime_turn = AsyncMock()
        handler = AsyncMock(return_value=True)
        adapter.set_runtime_control_handler(handler)

        await adapter._handle_runtime_signal("convo-1", "stop_and_drop")

        handler.assert_awaited_once()
        event, session_key, signal = handler.await_args.args
        assert event.text == ""
        assert event.source.chat_id == "convo-1"
        assert session_key
        assert signal == "stop_and_drop"
        adapter.handle_message.assert_not_awaited()
        adapter._publish_runtime_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_runtime_status_publishes_host_mode(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            async def update_runtime_status(self, **kwargs):
                self.calls.append(kwargs)

        client = FakeClient()
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._client = client

        await adapter._publish_runtime_status()

        assert client.calls[0]["host_mode"] is True
        descriptor = client.calls[0]["runtime_descriptor"]
        assert descriptor["supportsInputInterrupt"] is True
        assert any(action["id"] == "stop-and-clear-queue" for action in descriptor["actions"])

    @pytest.mark.asyncio
    async def test_runtime_signal_preserves_canon_actor_for_group_sessions(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._agent_id = "agent-1"
        adapter._conversation_cache["convo-1"] = {
            "id": "convo-1",
            "type": "group",
            "name": "Research",
        }
        adapter.handle_message = AsyncMock()

        await adapter._handle_runtime_signal(
            "convo-1",
            "interrupt",
            updated_by="owner-1",
        )

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/stop"
        assert event.source.chat_type == "group"
        assert event.source.user_id == "owner-1"

    @pytest.mark.asyncio
    async def test_runtime_signal_poll_passes_updated_by(self):
        class FakeClient:
            async def consume_runtime_signal(self, conversation_id):
                assert conversation_id == "convo-1"
                return {
                    "status": "signal",
                    "signal": "new_session",
                    "updatedBy": "owner-1",
                }

        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._client = FakeClient()
        adapter._conversation_cache["convo-1"] = {"id": "convo-1", "type": "group"}
        adapter._handle_runtime_signal = AsyncMock()

        await adapter._poll_runtime_signals_once()

        adapter._handle_runtime_signal.assert_awaited_once_with(
            "convo-1",
            "new_session",
            updated_by="owner-1",
        )

    @pytest.mark.asyncio
    async def test_audio_and_video_attachments_are_materialized(self, monkeypatch):
        from gateway.platforms.base import MessageType

        class FakeClient:
            async def download_media(self, url):
                if url.endswith("voice.m4a"):
                    return b"audio-bytes", "audio/mp4"
                return b"video-bytes", "video/mp4"

        monkeypatch.setattr(
            _canon,
            "cache_audio_from_bytes",
            lambda data, ext=".ogg": f"/tmp/canon-audio{ext}",
        )
        monkeypatch.setattr(
            _canon,
            "cache_video_from_bytes",
            lambda data, ext=".mp4": f"/tmp/canon-video{ext}",
        )

        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._client = FakeClient()
        adapter.handle_message = AsyncMock()

        await adapter._handle_message_payload({
            "conversationId": "convo-1",
            "message": {
                "id": "msg-media",
                "senderId": "human-1",
                "text": "review these",
                "contentType": "file",
                "attachments": [
                    {"kind": "audio", "url": "https://media.example/voice.m4a"},
                    {"kind": "file", "url": "https://media.example/demo.mp4"},
                ],
            },
        })

        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.VOICE
        assert event.media_urls == ["/tmp/canon-audio.m4a", "/tmp/canon-video.mp4"]
        assert event.media_types == ["audio/mp4", "video/mp4"]

    @pytest.mark.asyncio
    async def test_gateway_message_accepted_marks_accepted_message_read(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake

        await adapter.on_gateway_message_accepted(
            self._message_event(metadata={"inboundDisposition": "queued"}),
            "session-1",
        )

        assert fake.dispositions == [("convo-1", "msg-1", "accepted_now")]
        assert fake.reads == ["convo-1"]

    @pytest.mark.asyncio
    async def test_runtime_turn_start_only_publishes_runtime_turn(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        adapter._publish_runtime_turn = AsyncMock()

        await adapter._on_runtime_turn_start(
            self._message_event(metadata={"inboundDisposition": "queued"}),
            "session-1",
        )

        assert fake.dispositions == []
        assert fake.reads == []
        adapter._publish_runtime_turn.assert_awaited_once_with(
            "convo-1",
            "thinking",
            session_key="session-1",
            active_message_ids=["msg-1"],
        )

    @pytest.mark.asyncio
    async def test_gateway_message_accepted_read_failure_is_non_fatal(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient(fail_read=True)
        adapter._client = fake

        await adapter.on_gateway_message_accepted(self._message_event(), "session-1")

        assert fake.dispositions == []
        assert fake.reads == []

    @pytest.mark.asyncio
    async def test_unauthorized_gateway_message_does_not_mark_read(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: [])
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        runner = self._gateway_runner(adapter, authorized=False)

        result = await runner._handle_message(self._message_event())

        assert result is None
        assert fake.dispositions == []
        assert fake.reads == []
        runner._handle_message_with_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pre_gateway_dispatch_skip_does_not_mark_read(self, monkeypatch):
        def fake_hook(name, **kwargs):
            if name == "pre_gateway_dispatch":
                return [{"action": "skip", "reason": "plugin-handled"}]
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_hook)
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        runner = self._gateway_runner(adapter, authorized=True)

        result = await runner._handle_message(self._message_event())

        assert result is None
        assert fake.dispositions == []
        assert fake.reads == []
        runner._is_user_authorized.assert_not_called()
        runner._handle_message_with_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_authorized_gateway_message_marks_read(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: [])
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        runner = self._gateway_runner(adapter, authorized=True)

        result = await runner._handle_message(self._message_event())

        assert result == {"final_response": "", "messages": []}
        assert fake.dispositions == []
        assert fake.reads == ["convo-1"]
        runner._handle_message_with_agent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_queued_busy_message_waits_until_active_turn_to_mark_read(self, monkeypatch):
        from gateway.session import build_session_key
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        runner = self._gateway_runner(adapter, authorized=True)
        runner._busy_input_mode = "interrupt"
        event = self._message_event(
            metadata={"deliveryIntent": "queue", "inboundDisposition": "queued"}
        )
        session_key = build_session_key(event.source)
        runner._running_agents[session_key] = MagicMock()

        result = await GatewayRunner._handle_active_session_busy_message(
            runner,
            event,
            session_key,
        )

        assert result is True
        assert adapter._pending_messages[session_key] is event
        assert fake.dispositions == []
        assert fake.reads == []

        await adapter.on_gateway_message_accepted(
            event,
            session_key,
            phase="queued_turn",
        )

        assert fake.dispositions == [("convo-1", "msg-1", "accepted_now")]
        assert fake.reads == ["convo-1"]


class TestCanonOutbound:
    class FakeClient:
        def __init__(self):
            self.sent = []
            self.streaming = []
            self.streaming_cleared = []
            self.runtime_turns = []
            self.typing = []
            self.uploads = []
            self.runtime_inputs = []
            self.runtime_input_responses = []
            self.runtime_approvals = []
            self.runtime_approval_responses = []
            self.closed = False

        async def send_message(
            self, conversation_id, text, *, reply_to=None, metadata=None, options=None
        ):
            self.sent.append({
                "conversation_id": conversation_id,
                "text": text,
                "reply_to": reply_to,
                "metadata": metadata,
                "options": options,
            })
            return {"messageId": "msg-out"}

        async def set_streaming(self, conversation_id, *, text, status="streaming", turn_id=None):
            self.streaming.append((conversation_id, text, status, turn_id))

        async def clear_streaming(self, conversation_id):
            self.streaming_cleared.append(conversation_id)

        async def update_runtime_turn(self, conversation_id, **kwargs):
            self.runtime_turns.append((conversation_id, kwargs))

        async def set_typing(self, conversation_id, typing, status=None):
            self.typing.append((conversation_id, typing, status))

        async def upload_media(
            self, conversation_id, data, mime_type, *, file_name=None
        ):
            self.uploads.append((conversation_id, data, mime_type, file_name))
            kind = "audio" if mime_type.startswith("audio/") else "file"
            return {
                "url": f"https://media.example/{file_name}",
                "attachment": {
                    "kind": kind,
                    "url": f"https://media.example/{file_name}",
                    "mimeType": mime_type,
                    "fileName": file_name,
                },
            }

        async def create_runtime_input_request(self, conversation_id, **kwargs):
            self.runtime_inputs.append((conversation_id, kwargs))
            return {"messageId": "input-card"}

        async def consume_runtime_input_response(self, conversation_id, input_id, *, cancel=False):
            self.runtime_input_responses.append((conversation_id, input_id, cancel))
            return {
                "status": "submitted",
                "inputId": input_id,
                "kind": "clarify",
                "value": "Use option B",
            }

        async def create_runtime_approval_request(self, conversation_id, **kwargs):
            self.runtime_approvals.append((conversation_id, kwargs))
            return {"messageId": "approval-card"}

        async def consume_runtime_approval_response(self, conversation_id, approval_id, *, cancel=False):
            self.runtime_approval_responses.append((conversation_id, approval_id, cancel))
            return {
                "status": "allow",
                "approvalId": approval_id,
                "sessionRule": {"type": "approve-tool", "toolPattern": "Command"},
            }

        async def close(self):
            self.closed = True

    @pytest.mark.asyncio
    async def test_send_posts_turn_complete_metadata(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake

        result = await adapter.send(
            "convo-1",
            "hello",
            reply_to="msg-parent",
            metadata={
                "canon_metadata": {"source": "test"},
                "canon_options": {"mentions": ["u1"]},
            },
        )

        assert result.success is True
        assert result.message_id == "msg-out"
        assert fake.sent[0]["conversation_id"] == "convo-1"
        assert fake.sent[0]["text"] == "hello"
        assert fake.sent[0]["reply_to"] == "msg-parent"
        assert fake.sent[0]["metadata"] == {
            "source": "test",
            **TURN_COMPLETE_METADATA,
        }
        assert fake.sent[0]["options"] == {"mentions": ["u1"]}

    @pytest.mark.asyncio
    async def test_streaming_preview_uses_live_snapshot_until_final_edit(self, monkeypatch):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        monkeypatch.setattr(_canon.asyncio, "sleep", AsyncMock())

        preview = await adapter.send(
            "convo-1",
            "Hel",
            metadata={"canon_streaming_preview": True},
        )
        final = await adapter.edit_message(
            "convo-1",
            preview.message_id or "",
            "Hello",
            finalize=True,
            metadata={"canon_streaming_preview": True},
        )

        assert preview.success is True
        assert fake.streaming == [
            ("convo-1", "Hel", "streaming", preview.message_id),
        ]
        assert final.success is True
        assert fake.sent == [
            {
                "conversation_id": "convo-1",
                "text": "Hello",
                "reply_to": None,
                "metadata": {**TURN_COMPLETE_METADATA, "turnId": preview.message_id},
                "options": {},
            }
        ]
        assert fake.streaming_cleared == ["convo-1"]

    @pytest.mark.asyncio
    async def test_stream_consumer_same_text_finalizes_streaming_preview(self, monkeypatch):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        monkeypatch.setattr(_canon.asyncio, "sleep", AsyncMock())

        consumer = GatewayStreamConsumer(
            adapter,
            "convo-1",
            config=StreamConsumerConfig(cursor=""),
            metadata={"canon_streaming_preview": True},
        )

        assert adapter.REQUIRES_EDIT_FINALIZE is True
        assert await consumer._send_or_edit("Hello") is True
        assert await consumer._send_or_edit("Hello", finalize=True) is True

        turn_id = fake.streaming[0][3]
        assert fake.streaming == [
            ("convo-1", "Hello", "streaming", turn_id),
        ]
        assert fake.sent == [
            {
                "conversation_id": "convo-1",
                "text": "Hello",
                "reply_to": None,
                "metadata": {**TURN_COMPLETE_METADATA, "turnId": turn_id},
                "options": {},
            }
        ]
        assert fake.streaming_cleared == ["convo-1"]

    @pytest.mark.asyncio
    async def test_typing_is_best_effort(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake

        await adapter.send_typing("convo-1")
        await adapter.stop_typing("convo-1")

        assert fake.typing == [
            ("convo-1", True, "thinking"),
            ("convo-1", False, None),
        ]

    @pytest.mark.asyncio
    async def test_send_voice_uploads_audio_attachment(self, tmp_path):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        audio_path = tmp_path / "reply.mp3"
        audio_path.write_bytes(b"audio-bytes")

        result = await adapter.send_voice(
            "convo-1", str(audio_path), caption="voice reply"
        )

        assert result.success is True
        assert fake.uploads[0][2] == "audio/mpeg"
        sent = fake.sent[0]
        assert sent["text"] == "voice reply"
        assert sent["options"]["contentType"] == "audio"
        assert sent["options"]["attachments"][0]["kind"] == "audio"
        assert sent["metadata"] == TURN_COMPLETE_METADATA

    @pytest.mark.asyncio
    async def test_send_video_uploads_file_attachment_with_video_mime(self, tmp_path):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake
        video_path = tmp_path / "demo.mp4"
        video_path.write_bytes(b"video-bytes")

        result = await adapter.send_video("convo-1", str(video_path))

        assert result.success is True
        assert fake.uploads[0][2] == "video/mp4"
        sent = fake.sent[0]
        assert sent["options"]["contentType"] == "file"
        assert sent["options"]["attachments"][0]["mimeType"] == "video/mp4"

    @pytest.mark.asyncio
    async def test_send_clarify_creates_runtime_input_card_and_resolves(self, monkeypatch):
        resolved = []
        monkeypatch.setattr(
            "tools.clarify_gateway.resolve_gateway_clarify",
            lambda input_id, value: resolved.append((input_id, value)) or True,
        )

        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake

        result = await adapter.send_clarify(
            "convo-1",
            "Which branch?",
            ["main", "backup"],
            "clarify-1",
            "session-1",
        )

        assert result.success is True
        assert result.message_id == "input-card"
        assert fake.runtime_inputs[0][0] == "convo-1"
        assert fake.runtime_inputs[0][1]["kind"] == "clarify"
        assert fake.runtime_inputs[0][1]["prompt"] == "Which branch?"
        assert fake.runtime_inputs[0][1]["choices"] == [
            {"label": "main", "value": "main"},
            {"label": "backup", "value": "backup"},
        ]
        await asyncio.wait_for(_wait_for(lambda: bool(resolved)), timeout=1)
        assert resolved == [("clarify-1", "Use option B")]

    @pytest.mark.asyncio
    async def test_send_clarify_resolves_structured_answers(self, monkeypatch):
        resolved = []
        monkeypatch.setattr(
            "tools.clarify_gateway.resolve_gateway_clarify",
            lambda input_id, value: resolved.append((input_id, value)) or True,
        )

        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()

        async def structured_response(conversation_id, input_id, *, cancel=False):
            fake.runtime_input_responses.append((conversation_id, input_id, cancel))
            return {
                "status": "submitted",
                "inputId": input_id,
                "kind": "clarify",
                "value": "",
                "answers": {
                    "field": {"answers": ["pre_vat_total"]},
                    "new_value": {"answers": ["900.00"]},
                },
            }

        fake.consume_runtime_input_response = structured_response
        adapter._client = fake

        result = await adapter.send_clarify(
            "convo-1",
            "Which field?",
            [],
            "clarify-structured",
            "session-1",
        )

        assert result.success is True
        await asyncio.wait_for(_wait_for(lambda: bool(resolved)), timeout=1)
        assert resolved == [("clarify-structured", "field: pre_vat_total\nnew_value: 900.00")]

    @pytest.mark.asyncio
    async def test_send_clarify_preserves_blank_legacy_value(self, monkeypatch):
        resolved = []
        monkeypatch.setattr(
            "tools.clarify_gateway.resolve_gateway_clarify",
            lambda input_id, value: resolved.append((input_id, value)) or True,
        )

        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()

        async def blank_response(conversation_id, input_id, *, cancel=False):
            fake.runtime_input_responses.append((conversation_id, input_id, cancel))
            return {
                "status": "submitted",
                "inputId": input_id,
                "kind": "clarify",
                "value": "",
                "choice": {"value": "fallback-choice", "label": "Fallback choice"},
            }

        fake.consume_runtime_input_response = blank_response
        adapter._client = fake

        result = await adapter.send_clarify(
            "convo-1",
            "Optional note?",
            [],
            "clarify-blank",
            "session-1",
        )

        assert result.success is True
        await asyncio.wait_for(_wait_for(lambda: bool(resolved)), timeout=1)
        assert resolved == [("clarify-blank", "")]

    @pytest.mark.asyncio
    async def test_send_exec_approval_creates_approval_card_and_resolves_session(self, monkeypatch):
        resolved = []
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda session_key, choice, resolve_all=False: resolved.append((session_key, choice)) or 1,
        )

        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        adapter._client = fake

        result = await adapter.send_exec_approval(
            "convo-1",
            "rm -rf tmp",
            "session-1",
            "delete files",
        )

        assert result.success is True
        assert result.message_id == "approval-card"
        assert fake.runtime_approvals[0][0] == "convo-1"
        request = fake.runtime_approvals[0][1]
        assert request["tool_name"] == "Command"
        assert request["tool_summary"] == "delete files"
        assert request["details"][0] == {
            "label": "Command",
            "value": "rm -rf tmp",
            "monospace": True,
        }
        await asyncio.wait_for(_wait_for(lambda: bool(resolved)), timeout=1)
        assert resolved == [("session-1", "session")]


class TestCanonLifecycle:
    class FakeClient:
        def __init__(self):
            self.closed = False

        async def get_me(self):
            return {"agentId": "agent-1", "displayName": "Hermes"}

        async def get_conversations(self):
            return [{"id": "convo-1", "type": "direct", "name": "Ada"}]

        async def close(self):
            self.closed = True

    @pytest.mark.asyncio
    async def test_connect_hydrates_identity_and_conversations(self, monkeypatch):
        monkeypatch.delenv("HERMES_CANON_AGENT_NAME", raising=False)
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        fake = self.FakeClient()
        block = asyncio.Event()

        async def fake_stream_loop():
            await block.wait()

        adapter._make_client = lambda: fake
        adapter._stream_loop = fake_stream_loop

        assert await adapter.connect() is True
        assert adapter.is_connected is True
        assert adapter._agent_id == "agent-1"
        assert adapter.profile_agent_name == "Hermes"
        assert os.environ["HERMES_CANON_AGENT_NAME"] == "Hermes"
        assert adapter._conversation_cache["convo-1"]["name"] == "Ada"

        block.set()
        await adapter.disconnect()
        assert fake.closed is True
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_stream_frame_dispatches_message_created(self):
        adapter = CanonAdapter(_config(extra={"api_key": "key"}))
        adapter._handle_message_payload = AsyncMock()

        frame = CanonStreamFrame(
            event="message.created",
            data={"conversationId": "convo-1", "message": {"id": "m1"}},
            event_id="evt-1",
        )
        await adapter._handle_stream_frame(frame)

        adapter._handle_message_payload.assert_awaited_once_with(frame.data)


class TestCanonSSE:
    def test_parse_sse_frame(self):
        frame = _parse_sse_frame(
            'id: evt-1\nevent: message.created\ndata: {"conversationId":"convo-1"}'
        )

        assert frame.event == "message.created"
        assert frame.event_id == "evt-1"
        assert frame.data == {"conversationId": "convo-1"}

    def test_parse_sse_frame_ignores_comments(self):
        frame = _parse_sse_frame(": keepalive\nevent: heartbeat\ndata: ok")

        assert frame.event == "heartbeat"
        assert frame.data == "ok"


class TestCanonStandalone:
    class FakeClient:
        instances = []

        def __init__(
            self, api_key, *, base_url=DEFAULT_BASE_URL, stream_url=DEFAULT_STREAM_URL
        ):
            self.api_key = api_key
            self.base_url = base_url
            self.stream_url = stream_url
            self.sent = []
            self.closed = False
            self.instances.append(self)

        async def send_message(
            self, conversation_id, text, *, reply_to=None, metadata=None, options=None
        ):
            self.sent.append((conversation_id, text, reply_to, metadata, options))
            return {"messageId": "standalone-1"}

        async def close(self):
            self.closed = True

    @pytest.mark.asyncio
    async def test_standalone_send_uses_home_channel_and_closes(self, monkeypatch):
        self.FakeClient.instances = []
        monkeypatch.setattr(_canon, "CanonHttpClient", self.FakeClient)
        monkeypatch.setenv("CANON_API_KEY", "env-key")
        monkeypatch.setenv("CANON_HOME_CHANNEL", "convo-home")

        result = await _standalone_send(_config(), "", "hello cron")

        client = self.FakeClient.instances[0]
        assert result == {"success": True, "message_id": "standalone-1"}
        assert client.api_key == "env-key"
        assert client.sent == [
            ("convo-home", "hello cron", None, TURN_COMPLETE_METADATA, None)
        ]
        assert client.closed is True

    @pytest.mark.asyncio
    async def test_standalone_send_errors_without_api_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CANON_HOME", str(tmp_path))
        monkeypatch.delenv("CANON_API_KEY", raising=False)
        monkeypatch.delenv("CANON_AGENT", raising=False)
        monkeypatch.delenv("CANON_AGENTS_JSON_BOOTSTRAP", raising=False)

        result = await _standalone_send(_config(), "convo-1", "hello")

        assert "error" in result
        assert "credentials" in result["error"]


class TestCanonRegister:
    def test_register_metadata(self):
        recorded = {}

        class Context:
            def register_platform(self, **kwargs):
                recorded.update(kwargs)

        register(Context())

        assert recorded["name"] == "canon"
        assert recorded["label"] == "Canon"
        assert recorded["required_env"] == []
        assert callable(recorded["setup_fn"])
        assert recorded["cron_deliver_env_var"] == "CANON_HOME_CHANNEL"
        assert recorded["standalone_sender_fn"] is _standalone_send
        assert recorded["allowed_users_env"] == "CANON_ALLOWED_USERS"
        assert recorded["allow_all_env"] == "CANON_ALLOW_ALL_USERS"
        assert recorded["group_allowed_users_env"] == "CANON_GROUP_ALLOWED_USERS"
        assert recorded["group_allowed_chats_env"] == "CANON_GROUP_ALLOWED_CONVERSATIONS"
        assert recorded["max_message_length"] == 8000
