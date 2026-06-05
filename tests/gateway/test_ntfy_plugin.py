"""Tests for the ntfy platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_ntfy`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.

Most tests target the adapter class directly. The plugin-shape tests
(``register()``, ``_env_enablement``, ``_standalone_send``, registry
presence) replace the core-file grep tests from the original PR — the
ntfy adapter no longer modifies ``gateway/config.py``, ``gateway/run.py``,
``cron/scheduler.py``, ``toolsets.py``, etc.  Everything routes through
the ``platform_registry``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_ntfy = load_plugin_adapter("ntfy")

NtfyAdapter = _ntfy.NtfyAdapter
check_requirements = _ntfy.check_requirements
validate_config = _ntfy.validate_config
is_connected = _ntfy.is_connected
register = _ntfy.register
_env_enablement = _ntfy._env_enablement
_standalone_send = _ntfy._standalone_send
DEFAULT_SERVER = _ntfy.DEFAULT_SERVER
DEDUP_WINDOW_SECONDS = _ntfy.DEDUP_WINDOW_SECONDS
DEDUP_MAX_SIZE = _ntfy.DEDUP_MAX_SIZE
MAX_MESSAGE_LENGTH = _ntfy.MAX_MESSAGE_LENGTH


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------


def test_platform_enum_resolves_via_plugin_scan():
    """The plugin filesystem scan should expose Platform("ntfy")."""
    from gateway.config import Platform
    p = Platform("ntfy")
    assert p.value == "ntfy"
    # Identity stability — repeated lookups return the same pseudo-member
    assert Platform("ntfy") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------


class TestNtfyRequirements:

    def test_returns_false_when_httpx_unavailable(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-test")
        monkeypatch.setattr(_ntfy, "HTTPX_AVAILABLE", False)
        assert check_requirements() is False

    def test_returns_false_when_topic_not_set(self, monkeypatch):
        monkeypatch.setattr(_ntfy, "HTTPX_AVAILABLE", True)
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        assert check_requirements() is False

    def test_returns_true_when_topic_set_via_env(self, monkeypatch):
        monkeypatch.setattr(_ntfy, "HTTPX_AVAILABLE", True)
        monkeypatch.setenv("NTFY_TOPIC", "hermes-test")
        assert check_requirements() is True

    def test_validate_config_requires_topic(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        assert validate_config(PlatformConfig(enabled=True, extra={})) is False
        assert validate_config(
            PlatformConfig(enabled=True, extra={"topic": "t"})
        ) is True

    def test_is_connected_from_extra(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        assert is_connected(PlatformConfig(enabled=True, extra={"topic": "t"})) is True
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False

    def test_is_connected_from_env(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "env-topic")
        assert is_connected(PlatformConfig(enabled=True, extra={})) is True


# ---------------------------------------------------------------------------
# 3. Adapter init
# ---------------------------------------------------------------------------


class TestNtfyAdapterInit:

    def test_default_server_url(self, monkeypatch):
        monkeypatch.delenv("NTFY_SERVER_URL", raising=False)
        config = PlatformConfig(enabled=True, extra={"topic": "hermes-in"})
        adapter = NtfyAdapter(config)
        assert adapter._server == DEFAULT_SERVER.rstrip("/")

    def test_topic_read_from_extra(self):
        config = PlatformConfig(enabled=True, extra={"topic": "my-topic"})
        adapter = NtfyAdapter(config)
        assert adapter._topic == "my-topic"

    def test_topic_read_from_env(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "env-topic")
        config = PlatformConfig(enabled=True, extra={})
        adapter = NtfyAdapter(config)
        assert adapter._topic == "env-topic"

    def test_publish_topic_falls_back_to_topic(self, monkeypatch):
        monkeypatch.delenv("NTFY_PUBLISH_TOPIC", raising=False)
        config = PlatformConfig(enabled=True, extra={"topic": "hermes-in"})
        adapter = NtfyAdapter(config)
        assert adapter._publish_topic == "hermes-in"

    def test_publish_topic_uses_extra_value(self):
        config = PlatformConfig(
            enabled=True,
            extra={"topic": "hermes-in", "publish_topic": "hermes-out"},
        )
        adapter = NtfyAdapter(config)
        assert adapter._publish_topic == "hermes-out"

    def test_token_read_from_extra(self):
        config = PlatformConfig(enabled=True, extra={"topic": "t", "token": "tok-123"})
        adapter = NtfyAdapter(config)
        assert adapter._token == "tok-123"

    def test_token_read_from_env(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOKEN", "env-token")
        config = PlatformConfig(enabled=True, extra={"topic": "t"})
        adapter = NtfyAdapter(config)
        assert adapter._token == "env-token"

    def test_server_trailing_slash_stripped(self):
        config = PlatformConfig(
            enabled=True,
            extra={"topic": "t", "server": "https://ntfy.example.com/"},
        )
        adapter = NtfyAdapter(config)
        assert not adapter._server.endswith("/")

    def test_initial_state(self):
        config = PlatformConfig(enabled=True, extra={"topic": "t"})
        adapter = NtfyAdapter(config)
        assert adapter._stream_task is None
        assert adapter._http_client is None
        assert adapter._seen_messages == {}


# ---------------------------------------------------------------------------
# 4. Auth headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:

    def _make_adapter(self, token=""):
        config = PlatformConfig(enabled=True, extra={"topic": "t", "token": token})
        return NtfyAdapter(config)

    def test_no_token_returns_empty_dict(self):
        adapter = self._make_adapter(token="")
        assert adapter._auth_headers() == {}

    def test_bearer_token_for_plain_token(self):
        adapter = self._make_adapter(token="myapitoken")
        headers = adapter._auth_headers()
        assert headers["Authorization"] == "Bearer myapitoken"

    def test_basic_auth_for_user_colon_password(self):
        adapter = self._make_adapter(token="user:pass")
        headers = adapter._auth_headers()
        assert headers["Authorization"].startswith("Basic ")
        import base64
        expected = "Basic " + base64.b64encode(b"user:pass").decode()
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
# 5. Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def _make_adapter(self):
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
        adapter = self._make_adapter()
        for i in range(DEDUP_MAX_SIZE + 20):
            adapter._is_duplicate(f"msg-{i}")
        assert len(adapter._seen_messages) <= DEDUP_MAX_SIZE + 20

    def test_expired_id_can_be_seen_again(self):
        import time
        adapter = self._make_adapter()
        adapter._seen_messages["old-msg"] = time.time() - DEDUP_WINDOW_SECONDS - 1
        for i in range(DEDUP_MAX_SIZE + 1):
            adapter._is_duplicate(f"fill-{i}")
        assert adapter._is_duplicate("old-msg") is False


# ---------------------------------------------------------------------------
# 6. connect() / disconnect()
# ---------------------------------------------------------------------------


class TestConnect:

    def test_connect_fails_when_httpx_unavailable(self, monkeypatch):
        monkeypatch.setattr(_ntfy, "HTTPX_AVAILABLE", False)
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        result = _run(adapter.connect())
        assert result is False

    def test_connect_fails_when_no_topic(self, monkeypatch):
        monkeypatch.setattr(_ntfy, "HTTPX_AVAILABLE", True)
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        config = PlatformConfig(enabled=True, extra={})
        adapter = NtfyAdapter(config)
        result = _run(adapter.connect())
        assert result is False

    def test_connect_starts_stream_task(self, monkeypatch):
        monkeypatch.setattr(_ntfy, "HTTPX_AVAILABLE", True)
        config = PlatformConfig(enabled=True, extra={"topic": "hermes-test"})
        adapter = NtfyAdapter(config)

        with patch.object(adapter, "_run_stream", new_callable=AsyncMock):
            with patch.object(_ntfy, "httpx") as mock_httpx:
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
# 7. send()
# ---------------------------------------------------------------------------


class TestSend:

    def _make_adapter(self, topic="hermes-in", publish_topic="", token="", markdown=False):
        extra: dict = {"topic": topic, "token": token}
        if publish_topic:
            extra["publish_topic"] = publish_topic
        if markdown:
            extra["markdown"] = True
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

        posted_url = mock_client.post.call_args[0][0]
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
        adapter = self._make_adapter(topic="hermes-in")

        class _FakeTimeout(Exception):
            pass

        fake_httpx = MagicMock()
        fake_httpx.TimeoutException = _FakeTimeout

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=_FakeTimeout("timed out"))
        adapter._http_client = mock_client

        with patch.object(_ntfy, "httpx", fake_httpx):
            result = _run(adapter.send("hermes-in", "Hello!"))

        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_send_truncates_to_max_length(self):
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
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        _run(adapter.send_typing("t"))  # must not raise

    def test_get_chat_info_returns_dict(self):
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

    def test_send_emits_markdown_header_when_enabled(self):
        adapter = self._make_adapter(topic="hermes-in", markdown=True)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        _run(adapter.send("hermes-in", "**bold**"))
        call_headers = mock_client.post.call_args[1]["headers"]
        assert call_headers.get("X-Markdown") == "true"

    def test_send_omits_markdown_header_when_disabled(self):
        adapter = self._make_adapter(topic="hermes-in", markdown=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        _run(adapter.send("hermes-in", "plain"))
        call_headers = mock_client.post.call_args[1]["headers"]
        assert "X-Markdown" not in call_headers

    def test_send_emits_echo_tag_header(self):
        """Outgoing messages carry the echo-prevention tag so the adapter
        can recognise and skip its own replies when subscribe topic ==
        publish topic (the default config that causes the loop)."""
        adapter = self._make_adapter(topic="hermes-in")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc123"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        _run(adapter.send("hermes-in", "Hello!"))
        call_headers = mock_client.post.call_args[1]["headers"]
        assert call_headers.get("X-Tags") == _ntfy._ECHO_TAG


# ---------------------------------------------------------------------------
# 8. Inbound message processing (identity invariant — security-critical)
# ---------------------------------------------------------------------------


class TestOnMessage:

    def _make_adapter(self):
        return NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "hermes-in"}))

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

    def test_own_tagged_message_skipped(self):
        """An incoming event carrying the adapter's echo tag is the agent's
        own reply echoed back by ntfy — it must not be dispatched, otherwise
        the agent replies to itself forever (issue #34447)."""
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "echo-1",
            "event": "message",
            "topic": "hermes-in",
            "message": "my own reply",
            "tags": [_ntfy._ECHO_TAG],
            "time": None,
        }))
        assert calls == []

    def test_message_with_other_tags_still_dispatched(self):
        """Tags unrelated to the echo sentinel must not suppress genuine
        user messages."""
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "user-1",
            "event": "message",
            "topic": "hermes-in",
            "message": "hello",
            "tags": ["warning", "skull"],
            "time": None,
        }))
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
        """title field must not be used for identity — it is publisher-controlled."""
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
        assert captured[0].source.user_id == "hermes-in"
        assert captured[0].source.user_name == "hermes-in"

    def test_unknown_publisher_cannot_impersonate_allowed_user(self):
        """An unknown publisher setting title=admin must not gain admin identity."""
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
# 9. _env_enablement() — env-only auto-config
# ---------------------------------------------------------------------------


class TestEnvEnablement:

    def test_returns_none_without_topic(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        assert _env_enablement() is None

    def test_seeds_topic_and_server(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.delenv("NTFY_SERVER_URL", raising=False)
        seed = _env_enablement()
        assert seed is not None
        assert seed["topic"] == "hermes-in"
        assert seed["server"] == DEFAULT_SERVER

    def test_custom_server_url(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_SERVER_URL", "https://ntfy.example.com/")
        seed = _env_enablement()
        assert seed["server"] == "https://ntfy.example.com"  # trailing slash stripped

    def test_publish_topic_seeded(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_PUBLISH_TOPIC", "hermes-out")
        seed = _env_enablement()
        assert seed["publish_topic"] == "hermes-out"

    def test_token_seeded(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_TOKEN", "tk_abc")
        seed = _env_enablement()
        assert seed["token"] == "tk_abc"

    def test_markdown_truthy_values(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        for val in ("true", "1", "yes", "TRUE"):
            monkeypatch.setenv("NTFY_MARKDOWN", val)
            assert _env_enablement()["markdown"] is True

    def test_markdown_falsy_values(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        for val in ("false", "0", "no", "anything"):
            monkeypatch.setenv("NTFY_MARKDOWN", val)
            assert _env_enablement()["markdown"] is False

    def test_home_channel_defaults_to_topic(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.delenv("NTFY_HOME_CHANNEL", raising=False)
        seed = _env_enablement()
        assert seed["home_channel"]["chat_id"] == "hermes-in"
        assert seed["home_channel"]["name"] == "hermes-in"

    def test_home_channel_override(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_HOME_CHANNEL", "alerts")
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alerts Channel")
        seed = _env_enablement()
        assert seed["home_channel"]["chat_id"] == "alerts"
        assert seed["home_channel"]["name"] == "Alerts Channel"


# ---------------------------------------------------------------------------
# 10. _standalone_send() — out-of-process cron delivery
# ---------------------------------------------------------------------------


class TestStandaloneSend:

    def test_errors_without_topic(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        monkeypatch.delenv("NTFY_PUBLISH_TOPIC", raising=False)
        pconfig = MagicMock()
        pconfig.extra = {}
        result = _run(_standalone_send(pconfig, "", "hello"))
        assert "error" in result
        assert "NTFY_TOPIC" in result["error"]

    def test_posts_to_server(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        pconfig = MagicMock()
        pconfig.extra = {"server": "https://ntfy.example.com", "topic": "hermes-in"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "id-42"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(_ntfy, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = _run(_standalone_send(pconfig, "hermes-in", "hello"))

        assert result.get("success") is True
        assert result["platform"] == "ntfy"
        assert result["message_id"] == "id-42"
        posted_url = mock_client.post.call_args[0][0]
        assert posted_url == "https://ntfy.example.com/hermes-in"

    def test_emits_echo_tag_header(self, monkeypatch):
        """Out-of-process cron / send_message deliveries also carry the echo
        tag, so a gateway subscribed to the same topic skips them too."""
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "id-99"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(_ntfy, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            _run(_standalone_send(pconfig, "hermes-in", "hi"))

        headers = mock_client.post.call_args[1]["headers"]
        assert headers.get("X-Tags") == _ntfy._ECHO_TAG

    def test_emits_bearer_token_when_configured(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in", "token": "tk_xyz"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(_ntfy, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            _run(_standalone_send(pconfig, "hermes-in", "hi"))

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer tk_xyz"

    def test_basic_auth_when_token_has_colon(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in", "token": "user:pass"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(_ntfy, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            _run(_standalone_send(pconfig, "hermes-in", "hi"))

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Authorization"].startswith("Basic ")

    def test_returns_error_on_http_failure(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in"}

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(_ntfy, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = _run(_standalone_send(pconfig, "hermes-in", "hi"))

        assert "error" in result
        assert "403" in result["error"]


# ---------------------------------------------------------------------------
# 11. register() — plugin-side metadata
# ---------------------------------------------------------------------------


def test_register_calls_register_platform():
    ctx = MagicMock()
    register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "ntfy"
    assert kwargs["label"] == "ntfy"
    assert kwargs["required_env"] == ["NTFY_TOPIC"]
    assert kwargs["allowed_users_env"] == "NTFY_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "NTFY_ALLOW_ALL_USERS"
    assert kwargs["cron_deliver_env_var"] == "NTFY_HOME_CHANNEL"
    assert kwargs["max_message_length"] == MAX_MESSAGE_LENGTH
    assert callable(kwargs["check_fn"])
    assert callable(kwargs["validate_config"])
    assert callable(kwargs["is_connected"])
    assert callable(kwargs["env_enablement_fn"])
    assert callable(kwargs["standalone_sender_fn"])
    assert callable(kwargs["adapter_factory"])
    # ntfy has no user-identifying PII (only topic names)
    assert kwargs["pii_safe"] is True
    assert "ntfy" in kwargs["platform_hint"].lower()


def test_adapter_factory_returns_ntfy_adapter():
    ctx = MagicMock()
    register(ctx)
    factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
    cfg = PlatformConfig(enabled=True, extra={"topic": "t"})
    adapter = factory(cfg)
    assert isinstance(adapter, NtfyAdapter)


# ---------------------------------------------------------------------------
# 12. Robustness — token hygiene + fatal-state propagation
# ---------------------------------------------------------------------------


class TestTokenHygiene:
    """``_build_auth_header`` must strip pasted-token whitespace; pasted
    tokens often carry trailing newlines that break the Authorization line."""

    def test_trailing_whitespace_stripped(self):
        assert _ntfy._build_auth_header("  tok123  ") == {"Authorization": "Bearer tok123"}

    def test_trailing_newline_stripped(self):
        assert _ntfy._build_auth_header("tok123\n") == {"Authorization": "Bearer tok123"}

    def test_whitespace_only_returns_empty(self):
        assert _ntfy._build_auth_header("   \n  ") == {}

    def test_basic_auth_token_also_stripped(self):
        h = _ntfy._build_auth_header("  user:pass  ")
        assert h["Authorization"].startswith("Basic ")
        import base64
        assert h["Authorization"] == "Basic " + base64.b64encode(b"user:pass").decode()

    def test_adapter_strips_token_via_helper(self):
        """The adapter delegates to _build_auth_header, so token whitespace
        passed via config.extra is also stripped."""
        config = PlatformConfig(enabled=True, extra={"topic": "t", "token": "  tok\n"})
        adapter = NtfyAdapter(config)
        assert adapter._auth_headers() == {"Authorization": "Bearer tok"}


class TestFatalErrorPropagation:
    """When the stream hits 401/404, the adapter must transition to the
    ``fatal`` state via ``_set_fatal_error`` so the gateway's runtime
    status reflects reality instead of staying 'connected'."""

    def test_401_sets_fatal_unauthorized(self):
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        adapter._http_client = MagicMock()

        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.status_code = 401
        # async-context-manager flavor for httpx.stream
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        adapter._http_client.stream = MagicMock(return_value=mock_cm)

        fake_httpx = MagicMock()
        fake_httpx.Timeout = MagicMock()
        with patch.object(_ntfy, "httpx", fake_httpx):
            with pytest.raises(_ntfy._FatalStreamError):
                _run(adapter._consume_stream("https://ntfy.example/t/json", {}))

        assert adapter.has_fatal_error is True
        assert adapter._fatal_error_code == "ntfy_unauthorized"
        assert adapter._fatal_error_retryable is False

    def test_404_sets_fatal_topic_not_found(self):
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "missing-topic"}))
        adapter._http_client = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        adapter._http_client.stream = MagicMock(return_value=mock_cm)

        fake_httpx = MagicMock()
        fake_httpx.Timeout = MagicMock()
        with patch.object(_ntfy, "httpx", fake_httpx):
            with pytest.raises(_ntfy._FatalStreamError):
                _run(adapter._consume_stream("https://ntfy.example/missing-topic/json", {}))

        assert adapter.has_fatal_error is True
        assert adapter._fatal_error_code == "ntfy_topic_not_found"
        assert "missing-topic" in adapter._fatal_error_message
        assert adapter._fatal_error_retryable is False


class TestTruncateHelper:
    """``_truncate_body`` is shared between adapter.send() (inline truncation
    today, may migrate) and ``_standalone_send``. It must cap to
    MAX_MESSAGE_LENGTH and return bytes."""

    def test_short_message_passes_through(self):
        assert _ntfy._truncate_body("hi", context="test") == b"hi"

    def test_long_message_truncated(self):
        long = "x" * (MAX_MESSAGE_LENGTH + 50)
        result = _ntfy._truncate_body(long, context="test")
        assert isinstance(result, bytes)
        assert len(result) == MAX_MESSAGE_LENGTH

    def test_unicode_message_encoded(self):
        result = _ntfy._truncate_body("héllo 🔔", context="test")
        assert result == "héllo 🔔".encode("utf-8")
