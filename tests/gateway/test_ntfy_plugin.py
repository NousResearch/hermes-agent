"""Tests for the ntfy platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_ntfy`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests in the same process.

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

from gateway.config import HomeChannel, Platform, PlatformConfig
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



    def test_is_connected_from_extra(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        assert is_connected(PlatformConfig(enabled=True, extra={"topic": "t"})) is True
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False


# ---------------------------------------------------------------------------
# 3. Adapter init
# ---------------------------------------------------------------------------


class TestNtfyAdapterInit:


    def test_topic_read_from_env(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "env-topic")
        config = PlatformConfig(enabled=True, extra={})
        adapter = NtfyAdapter(config)
        assert adapter._topic == "env-topic"




    def test_token_read_from_env(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOKEN", "env-token")
        config = PlatformConfig(enabled=True, extra={"topic": "t"})
        adapter = NtfyAdapter(config)
        assert adapter._token == "env-token"


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


# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def _make_adapter(self):
        return NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))

    def test_first_message_not_duplicate(self):
        adapter = self._make_adapter()
        assert adapter._dedup.is_duplicate("msg-1") is False

    def test_second_occurrence_is_duplicate(self):
        adapter = self._make_adapter()
        adapter._dedup.is_duplicate("msg-1")
        assert adapter._dedup.is_duplicate("msg-1") is True


# ---------------------------------------------------------------------------
# 6. connect() / disconnect()
# ---------------------------------------------------------------------------


class TestConnect:


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


    def test_disconnect_cancels_stream_task(self):
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))

        async def _hang():
            await asyncio.sleep(0.2)

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


# ---------------------------------------------------------------------------
# 9. _env_enablement() — env-only auto-config
# ---------------------------------------------------------------------------


class TestEnvEnablement:

    def test_returns_none_without_topic(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        assert _env_enablement() is None


    def test_markdown_truthy_values(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        for val in ("true", "1", "yes", "TRUE"):
            monkeypatch.setenv("NTFY_MARKDOWN", val)
            assert _env_enablement()["markdown"] is True


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


# ---------------------------------------------------------------------------
# 10b. X-Title — a named channel titles its notifications
# ---------------------------------------------------------------------------


def _post_mock():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"id": "id-1"}
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _gateway_headers(extra=None, metadata=None, home_channel=None):
    adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "hermes-in", **(extra or {})}, home_channel=home_channel))
    adapter._http_client = _post_mock()
    result = _run(adapter.send("hermes-in", "hi", metadata=metadata))
    return result, adapter._http_client.post.call_args[1]["headers"]


def _standalone_headers(extra=None, home_channel=None):
    pconfig = MagicMock()
    pconfig.extra = {"topic": "hermes-in", **(extra or {})}
    pconfig.home_channel = home_channel
    client = _post_mock()
    with patch.object(_ntfy, "httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = client
        _run(_standalone_send(pconfig, "hermes-in", "hi"))
    return client.post.call_args[1]["headers"]


class TestTitleHeader:

    @pytest.fixture(autouse=True)
    def _no_name(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.delenv("NTFY_HOME_CHANNEL_NAME", raising=False)

    def test_configured_name_titles_both_paths(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alfred")
        assert _gateway_headers()[1]["X-Title"] == "Alfred"
        assert _standalone_headers()["X-Title"] == "Alfred"

    def test_unnamed_install_sends_no_title(self):
        assert "X-Title" not in _gateway_headers()[1]
        assert "X-Title" not in _standalone_headers()

    def test_title_source_order(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alfred")
        assert _gateway_headers({"title": "Extra"})[1]["X-Title"] == "Extra"
        assert _gateway_headers({"title": "Extra"}, {"title": "Restart"})[1]["X-Title"] == "Restart"
        assert _standalone_headers({"title": "Extra"})["X-Title"] == "Extra"

    def test_empty_source_falls_through_to_the_next(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alfred")
        assert _gateway_headers({"title": "  "}, {"title": ""})[1]["X-Title"] == "Alfred"

    def test_seeded_home_channel_name_is_not_a_title(self, monkeypatch):
        """``_env_enablement`` seeds the home channel's name as "Home" when only the
        topic is set; that placeholder must not title every notification."""
        monkeypatch.delenv("NTFY_HOME_CHANNEL", raising=False)
        seeded = _env_enablement()
        assert seeded["home_channel"]["name"] == "Home"
        assert "X-Title" not in _gateway_headers(seeded)[1]

    def test_saved_home_channel_name_titles_both_paths(self):
        """A name written under ``home_channel:`` loads as ``PlatformConfig.home_channel``, not ``extra`` or
        the environment."""
        home = HomeChannel(platform=Platform("ntfy"), chat_id="hermes-in", name="Clippy")
        assert _gateway_headers(home_channel=home)[1]["X-Title"] == "Clippy"
        assert _standalone_headers(home_channel=home)["X-Title"] == "Clippy"

    def test_saved_home_channel_name_ranks_after_the_env_name(self, monkeypatch):
        home = HomeChannel(platform=Platform("ntfy"), chat_id="hermes-in", name="Clippy")
        assert _gateway_headers(home_channel=home)[1]["X-Title"] == "Clippy"
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alfred")
        assert _gateway_headers(home_channel=home)[1]["X-Title"] == "Alfred"
        assert _gateway_headers({"title": "Extra"}, home_channel=home)[1]["X-Title"] == "Extra"
        assert _standalone_headers(home_channel=home)["X-Title"] == "Alfred"

    def test_saved_home_placeholder_is_not_a_title(self):
        """A saved ``name: Home`` is the placeholder the core writes for an unnamed home channel."""
        home = HomeChannel.from_dict({"platform": "ntfy", "chat_id": "hermes-in"})
        assert home.name == "Home"
        assert "X-Title" not in _gateway_headers(home_channel=home)[1]
        assert "X-Title" not in _standalone_headers(home_channel=home)

    @pytest.mark.parametrize("name", ["Home ", "home", "HOME", "Hermes-In", " hermes-in", "hermes-in"])
    def test_placeholder_and_topic_match_ignores_case_and_space(self, name):
        home = HomeChannel(platform=Platform("ntfy"), chat_id="hermes-in", name=name)
        assert "X-Title" not in _gateway_headers(home_channel=home)[1]
        assert "X-Title" not in _standalone_headers(home_channel=home)

    @pytest.mark.parametrize("name", [2026, True, None, ["Clippy"]])
    def test_saved_name_that_is_not_text_gets_no_title(self, name):
        home = HomeChannel(platform=Platform("ntfy"), chat_id="hermes-in", name=name)
        result, headers = _gateway_headers(home_channel=home)
        assert result.success is True and "X-Title" not in headers

    def test_saved_name_equal_to_the_topic_is_not_a_title(self):
        """``/sethome`` run from ntfy saves the topic as the name (``chat_name`` is the topic): ntfy already
        shows the topic, so that is no label."""
        home = HomeChannel(platform=Platform("ntfy"), chat_id="hermes-in", name="hermes-in")
        assert "X-Title" not in _gateway_headers(home_channel=home)[1]
        assert "X-Title" not in _standalone_headers(home_channel=home)

    def test_saved_home_channel_name_is_cleaned_like_any_title(self):
        home = HomeChannel(platform=Platform("ntfy"), chat_id="hermes-in", name="Caf\u00e9 \u4e2d")
        result, headers = _gateway_headers(home_channel=home)
        assert result.success is True and "X-Title" not in headers
        home = HomeChannel(platform=Platform("ntfy"), chat_id="hermes-in", name="Caf\u00e9")
        assert _gateway_headers(home_channel=home)[1]["X-Title"] == "Cafe"

    @pytest.mark.parametrize("hostile", [
        "Café", "Alfred 中", "Łukasz", "line\r\nInjected: yes", "bell\x07 ring", "tab\t\tgap", "中文", "\x00", 12345,
    ])
    def test_hostile_title_never_blocks_the_publish(self, monkeypatch, hostile):
        """The value reaches an HTTP header: whatever it holds, the request must build
        and the notice must be sent, and the header must be one clean ASCII line."""
        import httpx
        result, headers = _gateway_headers(metadata={"title": hostile})
        assert result.success is True
        httpx.Request("POST", "http://ntfy.invalid/t", headers=headers)  # raises on a bad header value
        title = headers.get("X-Title", "")
        assert title.isascii() and title == " ".join(title.split()) and title.isprintable()

    def test_accented_name_is_folded_to_ascii(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Café")
        assert _gateway_headers()[1]["X-Title"] == "Cafe"
        assert _standalone_headers()["X-Title"] == "Cafe"

    @pytest.mark.parametrize("name", ["Łukasz", "Straße", "Alfred 中", "中文", "\U0001F3E0"])
    def test_name_that_ascii_cannot_hold_gets_no_title_not_a_fragment(self, monkeypatch, name):
        """Dropping the letters ASCII lacks would send "ukasz" or "Strae" as the heading."""
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", name)
        result, headers = _gateway_headers()
        assert result.success is True and "X-Title" not in headers
        assert "X-Title" not in _standalone_headers()

    def test_unusable_source_falls_through_to_the_next(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alfred")
        assert _gateway_headers(metadata={"title": "中文"})[1]["X-Title"] == "Alfred"

    def test_very_long_name_is_capped_and_still_sends(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "A" * 70000)
        result, headers = _gateway_headers()
        assert result.success is True
        assert headers["X-Title"] == "A" * _ntfy._MAX_TITLE_LENGTH

    def test_other_headers_unchanged_by_a_title(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alfred")
        _, headers = _gateway_headers({"markdown": True, "token": "tk"})
        assert headers["X-Tags"] == _ntfy._ECHO_TAG
        assert headers["X-Markdown"] == "true"
        assert headers["Authorization"] == "Bearer tk"
        assert headers["Content-Type"] == "text/plain; charset=utf-8"


# ---------------------------------------------------------------------------
# 11. register() — plugin-side metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 12. Robustness — token hygiene + fatal-state propagation
# ---------------------------------------------------------------------------


class TestTokenHygiene:
    """``_build_auth_header`` must strip pasted-token whitespace; pasted
    tokens often carry trailing newlines that break the Authorization line."""

    def test_trailing_whitespace_stripped(self):
        assert _ntfy._build_auth_header("  tok123  ") == {"Authorization": "Bearer tok123"}


    def test_whitespace_only_returns_empty(self):
        assert _ntfy._build_auth_header("   \n  ") == {}


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




# ---------------------------------------------------------------------------
# 13. Multiplex secondary-profile scope
# ---------------------------------------------------------------------------
#
# __init__'s server/topic/publish_topic, _env_enablement's topic/server/
# publish_topic/markdown/home_channel, and check_requirements/validate_config/
# is_connected's topic reads, all previously read raw os.getenv
# unconditionally (only NTFY_TOKEN was already scoped). Under multiplex,
# os.environ holds the DEFAULT profile's YAML-to-env bridge output -- a
# secondary profile with its own (different or absent) ntfy config would
# silently subscribe to / publish on the default profile's topic, or get
# auto-enabled using the default profile's topic entirely. Mirrors the
# LINE/Buzz/SimpleX fix for #98738.

@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("NTFY_TOPIC", "default-topic")
    monkeypatch.setenv("NTFY_SERVER_URL", "https://default.example.com")
    monkeypatch.setenv("NTFY_PUBLISH_TOPIC", "default-out")


class TestMultiplexProfileScope:

    def test_secondary_extra_wins_over_default_profile_env(
        self, multiplex_scope, default_profile_env
    ):
        """The secondary profile's own config.yaml extra is authoritative,
        not the default profile's bridged topic/server/publish_topic."""
        multiplex_scope()
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "topic": "profile-topic",
                "server": "https://profile.example.com",
                "publish_topic": "profile-out",
            },
        )
        adapter = NtfyAdapter(cfg)
        assert adapter._topic == "profile-topic"
        assert adapter._server == "https://profile.example.com"
        assert adapter._publish_topic == "profile-out"

    def test_secondary_missing_keys_fail_closed(
        self, multiplex_scope, default_profile_env
    ):
        """Keys absent from the profile's own scope must NOT borrow the
        default profile's bridged env values -- that would silently
        subscribe/publish on the wrong topic."""
        multiplex_scope()
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter._topic == ""
        assert adapter._server == DEFAULT_SERVER
        assert adapter._publish_topic == ""
        # Nor may the registry auto-enable ntfy for this profile off the default's topic.
        assert _env_enablement() is None
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False

