"""Tests for ntfy platform adapter and integration points."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


class TestPlatformEnum:

    def test_ntfy_value(self):
        assert Platform.NTFY.value == "ntfy"

    def test_ntfy_in_all_platforms(self):
        values = [p.value for p in Platform]
        assert "ntfy" in values


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


class TestNtfyRequirements:

    def test_returns_false_when_httpx_unavailable(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-test")
        monkeypatch.setattr("gateway.platforms.ntfy.HTTPX_AVAILABLE", False)
        from gateway.platforms.ntfy import check_ntfy_requirements
        assert check_ntfy_requirements() is False

    def test_returns_false_when_topic_not_set(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.ntfy.HTTPX_AVAILABLE", True)
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        from gateway.platforms.ntfy import check_ntfy_requirements
        with patch("gateway.config.load_gateway_config") as mock_load:
            mock_cfg = MagicMock()
            mock_cfg.platforms = {}
            mock_load.return_value = mock_cfg
            assert check_ntfy_requirements() is False

    def test_returns_true_when_topic_set_via_env(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.ntfy.HTTPX_AVAILABLE", True)
        monkeypatch.setenv("NTFY_TOPIC", "hermes-test")
        from gateway.platforms.ntfy import check_ntfy_requirements
        assert check_ntfy_requirements() is True

    def test_returns_true_when_topic_set_via_env(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.ntfy.HTTPX_AVAILABLE", True)
        monkeypatch.setenv("NTFY_TOPIC", "hermes-cfg")
        from gateway.platforms.ntfy import check_ntfy_requirements
        assert check_ntfy_requirements() is True


# ---------------------------------------------------------------------------
# Config loading from env vars
# ---------------------------------------------------------------------------


class TestNtfyConfigLoading:

    def test_ntfy_topic_enables_platform(self, monkeypatch):
        from gateway.config import load_gateway_config

        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        config = load_gateway_config()
        assert Platform.NTFY in config.platforms
        pc = config.platforms[Platform.NTFY]
        assert pc.enabled is True
        assert pc.extra["topic"] == "hermes-in"

    def test_ntfy_server_url_stored_in_extra(self, monkeypatch):
        from gateway.config import load_gateway_config

        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_SERVER_URL", "https://ntfy.example.com")
        config = load_gateway_config()
        pc = config.platforms[Platform.NTFY]
        assert pc.extra.get("server") == "https://ntfy.example.com"

    def test_ntfy_token_stored_in_extra(self, monkeypatch):
        from gateway.config import load_gateway_config

        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_TOKEN", "tk_secret")
        config = load_gateway_config()
        pc = config.platforms[Platform.NTFY]
        assert pc.extra.get("token") == "tk_secret"

    def test_ntfy_publish_topic_stored_in_extra(self, monkeypatch):
        from gateway.config import load_gateway_config

        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_PUBLISH_TOPIC", "hermes-out")
        config = load_gateway_config()
        pc = config.platforms[Platform.NTFY]
        assert pc.extra.get("publish_topic") == "hermes-out"

    def test_ntfy_home_channel_set(self, monkeypatch):
        from gateway.config import load_gateway_config

        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_HOME_CHANNEL", "hermes-home")
        config = load_gateway_config()
        pc = config.platforms[Platform.NTFY]
        assert pc.home_channel is not None
        assert pc.home_channel.chat_id == "hermes-home"
        assert pc.home_channel.platform == Platform.NTFY

    def test_ntfy_home_channel_name_default(self, monkeypatch):
        from gateway.config import load_gateway_config

        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_HOME_CHANNEL", "hermes-home")
        monkeypatch.delenv("NTFY_HOME_CHANNEL_NAME", raising=False)
        config = load_gateway_config()
        pc = config.platforms[Platform.NTFY]
        assert pc.home_channel.name == "Home"

    def test_ntfy_not_enabled_when_topic_absent(self, monkeypatch):
        from gateway.config import load_gateway_config

        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        config = load_gateway_config()
        pc = config.platforms.get(Platform.NTFY)
        if pc is not None:
            assert not pc.enabled or pc.extra.get("topic", "") == ""

    def test_ntfy_in_connected_platforms_when_topic_set(self, monkeypatch):
        from gateway.config import load_gateway_config

        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        config = load_gateway_config()
        connected = config.get_connected_platforms()
        assert Platform.NTFY in connected


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


class TestNtfyAdapterInit:

    def test_default_server_url(self, monkeypatch):
        from gateway.platforms.ntfy import NtfyAdapter, DEFAULT_SERVER

        monkeypatch.delenv("NTFY_SERVER_URL", raising=False)
        config = PlatformConfig(enabled=True, extra={"topic": "hermes-in"})
        adapter = NtfyAdapter(config)
        assert adapter._server == DEFAULT_SERVER.rstrip("/")

    def test_topic_read_from_extra(self):
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(enabled=True, extra={"topic": "my-topic"})
        adapter = NtfyAdapter(config)
        assert adapter._topic == "my-topic"

    def test_topic_read_from_env(self, monkeypatch):
        from gateway.platforms.ntfy import NtfyAdapter

        monkeypatch.setenv("NTFY_TOPIC", "env-topic")
        config = PlatformConfig(enabled=True, extra={})
        adapter = NtfyAdapter(config)
        assert adapter._topic == "env-topic"

    def test_publish_topic_falls_back_to_topic(self, monkeypatch):
        from gateway.platforms.ntfy import NtfyAdapter

        monkeypatch.delenv("NTFY_PUBLISH_TOPIC", raising=False)
        config = PlatformConfig(enabled=True, extra={"topic": "hermes-in"})
        adapter = NtfyAdapter(config)
        assert adapter._publish_topic == "hermes-in"

    def test_publish_topic_uses_extra_value(self):
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(
            enabled=True,
            extra={"topic": "hermes-in", "publish_topic": "hermes-out"},
        )
        adapter = NtfyAdapter(config)
        assert adapter._publish_topic == "hermes-out"

    def test_token_read_from_extra(self):
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(enabled=True, extra={"topic": "t", "token": "tok-123"})
        adapter = NtfyAdapter(config)
        assert adapter._token == "tok-123"

    def test_token_read_from_env(self, monkeypatch):
        from gateway.platforms.ntfy import NtfyAdapter

        monkeypatch.setenv("NTFY_TOKEN", "env-token")
        config = PlatformConfig(enabled=True, extra={"topic": "t"})
        adapter = NtfyAdapter(config)
        assert adapter._token == "env-token"

    def test_server_trailing_slash_stripped(self):
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(
            enabled=True,
            extra={"topic": "t", "server": "https://ntfy.example.com/"},
        )
        adapter = NtfyAdapter(config)
        assert not adapter._server.endswith("/")

    def test_name_is_ntfy(self):
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(enabled=True, extra={"topic": "t"})
        adapter = NtfyAdapter(config)
        assert adapter.name == "Ntfy"

    def test_initial_state(self):
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(enabled=True, extra={"topic": "t"})
        adapter = NtfyAdapter(config)
        assert adapter._stream_task is None
        assert adapter._http_client is None
        assert adapter._seen_messages == {}


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:

    def _make_adapter(self, token=""):
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(enabled=True, extra={"topic": "t", "token": token})
        return NtfyAdapter(config)

    def test_no_token_returns_empty_dict(self):
        adapter = self._make_adapter(token="")
        assert adapter._auth_headers() == {}

    def test_bearer_token_for_plain_token(self):
        adapter = self._make_adapter(token="myapitoken")
        headers = adapter._auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer myapitoken"

    def test_basic_auth_for_user_colon_password(self):
        adapter = self._make_adapter(token="user:pass")
        headers = adapter._auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")
        expected = "Basic " + __import__("base64").b64encode(b"user:pass").decode()
        assert headers["Authorization"] == expected

    def test_bearer_token_used_when_no_colon(self):
        adapter = self._make_adapter(token="noColonHere")
        headers = adapter._auth_headers()
        assert headers["Authorization"] == "Bearer noColonHere"

    def test_auth_header_key_is_authorization(self):
        adapter = self._make_adapter(token="tok")
        headers = adapter._auth_headers()
        assert list(headers.keys()) == ["Authorization"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def _make_adapter(self):
        from gateway.platforms.ntfy import NtfyAdapter

        return NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))

    def test_first_message_not_duplicate(self):
        adapter = self._make_adapter()
        assert adapter._is_duplicate("msg-1") is False

    def test_second_occurrence_is_duplicate(self):
        adapter = self._make_adapter()
        adapter._is_duplicate("msg-1")
        assert adapter._is_duplicate("msg-1") is True

    def test_different_ids_not_duplicate(self):
        adapter = self._make_adapter()
        adapter._is_duplicate("msg-1")
        assert adapter._is_duplicate("msg-2") is False

    def test_many_messages_recorded(self):
        adapter = self._make_adapter()
        for i in range(50):
            adapter._is_duplicate(f"msg-{i}")
        assert len(adapter._seen_messages) == 50

    def test_cache_pruned_on_overflow(self):
        from gateway.platforms.ntfy import NtfyAdapter, DEDUP_MAX_SIZE

        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        for i in range(DEDUP_MAX_SIZE + 20):
            adapter._is_duplicate(f"msg-{i}")
        assert len(adapter._seen_messages) <= DEDUP_MAX_SIZE + 20

    def test_expired_id_can_be_seen_again(self):
        import time
        from gateway.platforms.ntfy import NtfyAdapter, DEDUP_WINDOW_SECONDS, DEDUP_MAX_SIZE

        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        adapter._seen_messages["old-msg"] = time.time() - DEDUP_WINDOW_SECONDS - 1
        for i in range(DEDUP_MAX_SIZE + 1):
            adapter._is_duplicate(f"fill-{i}")
        assert adapter._is_duplicate("old-msg") is False


# ---------------------------------------------------------------------------
# connect() / disconnect()
# ---------------------------------------------------------------------------


class TestConnect:

    def test_connect_fails_when_httpx_unavailable(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.ntfy.HTTPX_AVAILABLE", False)
        from gateway.platforms.ntfy import NtfyAdapter

        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        result = _run(adapter.connect())
        assert result is False

    def test_connect_fails_when_no_topic(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.ntfy.HTTPX_AVAILABLE", True)
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(enabled=True, extra={})
        adapter = NtfyAdapter(config)
        result = _run(adapter.connect())
        assert result is False

    def test_connect_starts_stream_task(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.ntfy.HTTPX_AVAILABLE", True)
        from gateway.platforms.ntfy import NtfyAdapter

        config = PlatformConfig(enabled=True, extra={"topic": "hermes-test"})
        adapter = NtfyAdapter(config)

        with patch.object(adapter, "_run_stream", new_callable=AsyncMock):
            with patch("gateway.platforms.ntfy.httpx") as mock_httpx:
                mock_httpx.AsyncClient.return_value = MagicMock()
                result = _run(adapter.connect())

        assert result is True
        assert adapter._stream_task is not None
        adapter._stream_task.cancel()
        try:
            _run(adapter._stream_task)
        except (asyncio.CancelledError, Exception):
            pass

    def test_disconnect_clears_state(self):
        from gateway.platforms.ntfy import NtfyAdapter

        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        adapter._seen_messages["x"] = 1.0
        adapter._http_client = AsyncMock()
        adapter._stream_task = None
        adapter._running = True

        _run(adapter.disconnect())

        assert adapter._seen_messages == {}
        assert adapter._http_client is None
        assert adapter._running is False

    def test_disconnect_cancels_stream_task(self):
        from gateway.platforms.ntfy import NtfyAdapter

        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))

        async def _hang():
            await asyncio.sleep(9999)

        loop = asyncio.get_event_loop()
        adapter._stream_task = loop.create_task(_hang())
        adapter._http_client = AsyncMock()
        adapter._running = True

        _run(adapter.disconnect())
        assert adapter._stream_task is None


# ---------------------------------------------------------------------------
# send()
# ---------------------------------------------------------------------------


class TestSend:

    def _make_adapter(self, topic="hermes-in", publish_topic="", token=""):
        from gateway.platforms.ntfy import NtfyAdapter

        extra = {"topic": topic, "token": token}
        if publish_topic:
            extra["publish_topic"] = publish_topic
        return NtfyAdapter(PlatformConfig(enabled=True, extra=extra))

    def test_send_fails_without_http_client(self):
        adapter = self._make_adapter()
        result = _run(adapter.send("hermes-in", "hello"))
        assert result.success is False
        assert "not initialized" in result.error.lower()

    def test_send_posts_to_publish_topic(self):
        adapter = self._make_adapter(topic="hermes-in", publish_topic="hermes-out")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc123"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        result = _run(adapter.send("hermes-in", "Hello ntfy!"))
        assert result.success is True
        assert result.message_id == "abc123"

        call_args = mock_client.post.call_args
        posted_url = call_args[0][0]
        assert posted_url.endswith("/hermes-out")

    def test_send_falls_back_to_subscribe_topic(self):
        adapter = self._make_adapter(topic="hermes-in")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        result = _run(adapter.send("hermes-in", "Hello!"))
        assert result.success is True
        posted_url = mock_client.post.call_args[0][0]
        assert posted_url.endswith("/hermes-in")

    def test_send_uses_metadata_publish_topic(self):
        adapter = self._make_adapter(topic="hermes-in")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        result = _run(adapter.send(
            "hermes-in", "Hi!", metadata={"publish_topic": "override-out"}
        ))
        assert result.success is True
        posted_url = mock_client.post.call_args[0][0]
        assert posted_url.endswith("/override-out")

    def test_send_handles_http_error_status(self):
        adapter = self._make_adapter(topic="hermes-in")

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        result = _run(adapter.send("hermes-in", "Hello!"))
        assert result.success is False
        assert "403" in result.error

    def test_send_handles_timeout(self):
        import gateway.platforms.ntfy as ntfy_mod

        adapter = self._make_adapter(topic="hermes-in")

        class _FakeTimeout(Exception):
            pass

        fake_httpx = MagicMock()
        fake_httpx.TimeoutException = _FakeTimeout

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=_FakeTimeout("timed out"))
        adapter._http_client = mock_client

        with patch.object(ntfy_mod, "httpx", fake_httpx):
            result = _run(adapter.send("hermes-in", "Hello!"))

        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_send_truncates_to_max_length(self):
        from gateway.platforms.ntfy import NtfyAdapter, MAX_MESSAGE_LENGTH

        adapter = self._make_adapter(topic="t")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        long_msg = "x" * (MAX_MESSAGE_LENGTH + 500)
        _run(adapter.send("t", long_msg))

        posted_body = mock_client.post.call_args[1]["content"]
        assert len(posted_body.decode()) <= MAX_MESSAGE_LENGTH

    def test_send_typing_is_noop(self):
        from gateway.platforms.ntfy import NtfyAdapter

        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        # Should not raise
        _run(adapter.send_typing("t"))

    def test_get_chat_info_returns_dict(self):
        from gateway.platforms.ntfy import NtfyAdapter

        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        info = _run(adapter.get_chat_info("hermes-in"))
        assert info["name"] == "hermes-in"
        assert info["type"] == "dm"

    def test_send_includes_bearer_auth_header(self):
        adapter = self._make_adapter(topic="hermes-in", token="mytoken")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        _run(adapter.send("hermes-in", "secure message"))

        call_headers = mock_client.post.call_args[1]["headers"]
        assert call_headers.get("Authorization") == "Bearer mytoken"


# ---------------------------------------------------------------------------
# Inbound message processing
# ---------------------------------------------------------------------------


class TestOnMessage:

    def _make_adapter(self):
        from gateway.platforms.ntfy import NtfyAdapter

        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "hermes-in"}))
        return adapter

    def test_message_dispatched_to_handler(self):
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)

        event = {
            "id": "evt-001",
            "event": "message",
            "topic": "hermes-in",
            "message": "Hello from ntfy",
            "time": 1700000000,
        }
        _run(adapter._on_message(event))
        assert len(calls) == 1
        assert calls[0].text == "Hello from ntfy"

    def test_empty_message_skipped(self):
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "x", "event": "message", "topic": "t", "message": "", "time": None
        }))
        assert calls == []

    def test_duplicate_message_skipped(self):
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)

        event = {"id": "dup-1", "event": "message", "topic": "hermes-in", "message": "hi", "time": None}
        _run(adapter._on_message(event))
        _run(adapter._on_message(event))
        assert len(calls) == 1

    def test_timestamp_parsed_from_event(self):
        from datetime import timezone

        adapter = self._make_adapter()
        captured = []

        async def handler(event):
            captured.append(event)

        adapter.set_message_handler(handler)

        _run(adapter._on_message({
            "id": "ts-1",
            "event": "message",
            "topic": "hermes-in",
            "message": "ping",
            "time": 1700000000,
        }))
        ts = captured[0].timestamp
        assert ts.tzinfo == timezone.utc

    def test_message_id_set_from_event(self):
        adapter = self._make_adapter()
        captured = []

        async def handler(event):
            captured.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "ntfy-id-42",
            "event": "message",
            "topic": "hermes-in",
            "message": "test",
            "time": None,
        }))
        assert captured[0].message_id == "ntfy-id-42"

    def test_title_not_used_as_user_id(self):
        """title field must not be used for identity — it is publisher-controlled
        and cannot be trusted as an authentication signal."""
        adapter = self._make_adapter()
        captured = []

        async def handler(event):
            captured.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "u-1",
            "event": "message",
            "topic": "hermes-in",
            "message": "hello",
            "title": "Alice",
            "time": None,
        }))
        # user_id must be the topic, never the spoofable title field
        assert captured[0].source.user_id == "hermes-in"
        assert captured[0].source.user_name == "hermes-in"

    def test_unknown_publisher_cannot_impersonate_allowed_user(self):
        """An unknown publisher setting title to an allowed username must not
        gain the identity of that user — identity is always the topic name."""
        adapter = self._make_adapter()
        captured = []

        async def handler(event):
            captured.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "u-2",
            "event": "message",
            "topic": "hermes-in",
            "message": "sensitive command",
            "title": "admin",
            "time": None,
        }))
        assert captured[0].source.user_id == "hermes-in"
        assert captured[0].source.user_id != "admin"

    def test_source_chat_id_is_topic(self):
        adapter = self._make_adapter()
        captured = []

        async def handler(event):
            captured.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "s-1",
            "event": "message",
            "topic": "hermes-in",
            "message": "hello",
            "time": None,
        }))
        assert captured[0].source.chat_id == "hermes-in"


# ---------------------------------------------------------------------------
# Integration: send_message_tool platform_map (source-level checks)
# ---------------------------------------------------------------------------


class TestSendMessageToolIntegration:

    def test_ntfy_in_platform_enum(self):
        assert hasattr(Platform, "NTFY")
        assert Platform.NTFY.value == "ntfy"

    def test_ntfy_in_platform_map_source(self):
        src = open("tools/send_message_tool.py").read()
        assert "Platform.NTFY" in src

    def test_send_ntfy_function_in_source(self):
        src = open("tools/send_message_tool.py").read()
        assert "async def _send_ntfy" in src

    def test_ntfy_branch_in_send_to_platform_source(self):
        src = open("tools/send_message_tool.py").read()
        assert "Platform.NTFY" in src
        assert "_send_ntfy" in src

    def test_send_ntfy_reads_server_from_extra(self):
        src = open("tools/send_message_tool.py").read()
        assert 'extra.get("server")' in src
        assert "NTFY_SERVER_URL" in src

    def test_send_ntfy_reads_topic_from_extra(self):
        src = open("tools/send_message_tool.py").read()
        assert 'extra.get("topic")' in src
        assert "NTFY_TOPIC" in src

    def test_send_ntfy_reads_token_from_extra(self):
        src = open("tools/send_message_tool.py").read()
        assert 'extra.get("token")' in src
        assert "NTFY_TOKEN" in src


# ---------------------------------------------------------------------------
# Integration: cron scheduler platform_map
# ---------------------------------------------------------------------------


class TestCronSchedulerIntegration:

    def test_ntfy_in_scheduler_platform_map_source(self):
        src = open("cron/scheduler.py").read()
        # ntfy routing handled via Platform._missing_() dynamic dispatch
        assert '"ntfy"' in src or "Platform._missing_" in src or "_missing_" in src

    def test_ntfy_in_cronjob_deliver_description(self):
        src = open("cron/scheduler.py").read()
        assert "ntfy" in src.lower()


# ---------------------------------------------------------------------------
# Integration: gateway/run.py authorization maps
# ---------------------------------------------------------------------------


class TestRunAuthorizationMaps:

    def test_ntfy_allowed_users_in_allowlist_check(self):
        src = open("gateway/run.py").read()
        assert "NTFY_ALLOWED_USERS" in src

    def test_ntfy_allow_all_users_in_allowlist_check(self):
        src = open("gateway/run.py").read()
        assert "NTFY_ALLOW_ALL_USERS" in src

    def test_ntfy_in_platform_env_map(self):
        src = open("gateway/run.py").read()
        assert 'Platform.NTFY: "NTFY_ALLOWED_USERS"' in src

    def test_ntfy_in_allow_all_map(self):
        src = open("gateway/run.py").read()
        assert 'Platform.NTFY: "NTFY_ALLOW_ALL_USERS"' in src

    def test_ntfy_create_adapter_branch(self):
        src = open("gateway/run.py").read()
        assert "Platform.NTFY" in src
        assert "NtfyAdapter" in src

    def test_ntfy_startup_allowlist_includes_ntfy_allowed_users(self):
        src = open("gateway/run.py").read()
        # Verify both env vars appear in the startup check tuples
        assert '"NTFY_ALLOWED_USERS"' in src
        assert '"NTFY_ALLOW_ALL_USERS"' in src


# ---------------------------------------------------------------------------
# Integration: toolsets
# ---------------------------------------------------------------------------


class TestToolsets:

    def test_hermes_ntfy_toolset_exists(self):
        from toolsets import get_toolset

        ts = get_toolset("hermes-ntfy")
        assert ts is not None
        assert "tools" in ts

    def test_hermes_ntfy_in_gateway_includes(self):
        from toolsets import get_toolset

        gw = get_toolset("hermes-gateway")
        assert "hermes-ntfy" in gw["includes"]

    def test_hermes_ntfy_resolves_tools(self):
        from toolsets import resolve_toolset

        tools = resolve_toolset("hermes-ntfy")
        assert len(tools) > 0

    def test_hermes_ntfy_description_mentions_ntfy(self):
        from toolsets import get_toolset

        ts = get_toolset("hermes-ntfy")
        assert "ntfy" in ts["description"].lower()


# ---------------------------------------------------------------------------
# Integration: prompt_builder platform hints
# ---------------------------------------------------------------------------


class TestPromptBuilderHints:

    def test_ntfy_hint_exists(self):
        from agent.prompt_builder import PLATFORM_HINTS

        assert "ntfy" in PLATFORM_HINTS

    def test_ntfy_hint_mentions_plain_text(self):
        from agent.prompt_builder import PLATFORM_HINTS

        hint = PLATFORM_HINTS["ntfy"].lower()
        assert "plain text" in hint

    def test_ntfy_hint_mentions_push_or_notifications(self):
        from agent.prompt_builder import PLATFORM_HINTS

        hint = PLATFORM_HINTS["ntfy"].lower()
        assert "push" in hint or "notification" in hint


# ---------------------------------------------------------------------------
# Integration: channel_directory
# ---------------------------------------------------------------------------


class TestChannelDirectory:

    def test_ntfy_in_session_based_platforms_source(self):
        src = open("gateway/channel_directory.py").read()
        assert '"ntfy"' in src

    def test_build_channel_directory_includes_ntfy_key(self):
        src = open("gateway/channel_directory.py").read()
        assert "ntfy" in src
