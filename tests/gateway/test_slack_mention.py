"""
Tests for Slack mention gating (require_mention / free_response_channels).

Follows the same pattern as test_whatsapp_group_gating.py.
"""

import sys
from unittest.mock import MagicMock

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same as test_slack.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import gateway.platforms.slack as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOT_USER_ID = "U_BOT_123"
CHANNEL_ID = "C0AQWDLHY9M"
OTHER_CHANNEL_ID = "C9999999999"


def _make_adapter(require_mention=None, strict_mention=None, free_response_channels=None):
    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if strict_mention is not None:
        extra["strict_mention"] = strict_mention
    if free_response_channels is not None:
        extra["free_response_channels"] = free_response_channels

    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {}
    return adapter


# ---------------------------------------------------------------------------
# Tests: _slack_require_mention
# ---------------------------------------------------------------------------

def test_require_mention_defaults_to_true(monkeypatch):
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_require_mention() is True


def test_require_mention_false():
    adapter = _make_adapter(require_mention=False)
    assert adapter._slack_require_mention() is False


def test_require_mention_true():
    adapter = _make_adapter(require_mention=True)
    assert adapter._slack_require_mention() is True


def test_require_mention_string_true():
    adapter = _make_adapter(require_mention="true")
    assert adapter._slack_require_mention() is True


def test_require_mention_string_false():
    adapter = _make_adapter(require_mention="false")
    assert adapter._slack_require_mention() is False


def test_require_mention_string_no():
    adapter = _make_adapter(require_mention="no")
    assert adapter._slack_require_mention() is False


def test_require_mention_string_yes():
    adapter = _make_adapter(require_mention="yes")
    assert adapter._slack_require_mention() is True


def test_require_mention_empty_string_stays_true():
    """Empty/malformed strings keep gating ON (explicit-false parser)."""
    adapter = _make_adapter(require_mention="")
    assert adapter._slack_require_mention() is True


def test_require_mention_malformed_string_stays_true():
    """Unrecognised values keep gating ON (fail-closed)."""
    adapter = _make_adapter(require_mention="maybe")
    assert adapter._slack_require_mention() is True


def test_require_mention_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_REQUIRE_MENTION", "false")
    adapter = _make_adapter()  # no config value -> falls back to env
    assert adapter._slack_require_mention() is False


def test_require_mention_env_var_default_true(monkeypatch):
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_require_mention() is True


# ---------------------------------------------------------------------------
# Tests: _slack_strict_mention
# ---------------------------------------------------------------------------

def test_strict_mention_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SLACK_STRICT_MENTION", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_strict_mention() is False


def test_strict_mention_true():
    adapter = _make_adapter(strict_mention=True)
    assert adapter._slack_strict_mention() is True


def test_strict_mention_false():
    adapter = _make_adapter(strict_mention=False)
    assert adapter._slack_strict_mention() is False


def test_strict_mention_string_true():
    adapter = _make_adapter(strict_mention="true")
    assert adapter._slack_strict_mention() is True


def test_strict_mention_string_off():
    adapter = _make_adapter(strict_mention="off")
    assert adapter._slack_strict_mention() is False


def test_strict_mention_malformed_stays_false():
    """Unrecognised values keep strict mode OFF (fail-open to legacy behavior)."""
    adapter = _make_adapter(strict_mention="maybe")
    assert adapter._slack_strict_mention() is False


def test_strict_mention_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_STRICT_MENTION", "true")
    adapter = _make_adapter()  # no config value -> falls back to env
    assert adapter._slack_strict_mention() is True


# ---------------------------------------------------------------------------
# Tests: _slack_free_response_channels
# ---------------------------------------------------------------------------

def test_free_response_channels_default_empty(monkeypatch):
    monkeypatch.delenv("SLACK_FREE_RESPONSE_CHANNELS", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_free_response_channels() == set()


def test_free_response_channels_list():
    adapter = _make_adapter(free_response_channels=[CHANNEL_ID, OTHER_CHANNEL_ID])
    result = adapter._slack_free_response_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


def test_free_response_channels_csv_string():
    adapter = _make_adapter(free_response_channels=f"{CHANNEL_ID}, {OTHER_CHANNEL_ID}")
    result = adapter._slack_free_response_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


def test_free_response_channels_empty_string():
    adapter = _make_adapter(free_response_channels="")
    assert adapter._slack_free_response_channels() == set()


def test_free_response_channels_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_FREE_RESPONSE_CHANNELS", f"{CHANNEL_ID},{OTHER_CHANNEL_ID}")
    adapter = _make_adapter()  # no config value → falls back to env
    result = adapter._slack_free_response_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


# ---------------------------------------------------------------------------
# Tests: mention gating integration (simulating _handle_slack_message logic)
# ---------------------------------------------------------------------------

def _would_process(adapter, *, is_dm=False, channel_id=CHANNEL_ID,
                   text="hello", mentioned=False, thread_reply=False,
                   active_session=False):
    """Simulate the mention gating logic from _handle_slack_message.

    Returns True if the message would be processed, False if it would be
    skipped (returned early).
    """
    bot_uid = adapter._team_bot_user_ids.get("T1", adapter._bot_user_id)
    if mentioned:
        text = f"<@{bot_uid}> {text}"
    is_mentioned = bot_uid and f"<@{bot_uid}>" in text

    if not is_dm:
        if channel_id in adapter._slack_free_response_channels():
            return True
        elif not adapter._slack_require_mention():
            return True
        elif not is_mentioned:
            if thread_reply and active_session:
                return True
            else:
                return False
    return True


def test_default_require_mention_channel_without_mention_ignored():
    adapter = _make_adapter()  # default: require_mention=True
    assert _would_process(adapter, text="hello everyone") is False


def test_require_mention_false_channel_without_mention_processed():
    adapter = _make_adapter(require_mention=False)
    assert _would_process(adapter, text="hello everyone") is True


def test_channel_in_free_response_processed_without_mention():
    adapter = _make_adapter(
        require_mention=True,
        free_response_channels=[CHANNEL_ID],
    )
    assert _would_process(adapter, channel_id=CHANNEL_ID, text="hello") is True


def test_other_channel_not_in_free_response_still_gated():
    adapter = _make_adapter(
        require_mention=True,
        free_response_channels=[CHANNEL_ID],
    )
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, text="hello") is False


def test_dm_always_processed_regardless_of_setting():
    adapter = _make_adapter(require_mention=True)
    assert _would_process(adapter, is_dm=True, text="hello") is True


def test_mentioned_message_always_processed():
    adapter = _make_adapter(require_mention=True)
    assert _would_process(adapter, mentioned=True, text="what's up") is True


def test_thread_reply_with_active_session_processed():
    adapter = _make_adapter(require_mention=True)
    assert _would_process(
        adapter, text="followup",
        thread_reply=True, active_session=True,
    ) is True


def test_thread_reply_without_active_session_ignored():
    adapter = _make_adapter(require_mention=True)
    assert _would_process(
        adapter, text="followup",
        thread_reply=True, active_session=False,
    ) is False


def test_bot_uid_none_processes_channel_message():
    """When bot_uid is None (before auth_test), channel messages pass through.

    This preserves the old behavior: the gating block is skipped entirely
    when bot_uid is falsy, so messages are not silently dropped during
    startup or for new workspaces.
    """
    adapter = _make_adapter(require_mention=True)
    adapter._bot_user_id = None
    adapter._team_bot_user_ids = {}

    # With bot_uid=None, the `if not is_dm and bot_uid:` condition is False,
    # so the gating block is skipped — message passes through.
    bot_uid = adapter._team_bot_user_ids.get("T1", adapter._bot_user_id)
    assert bot_uid is None

    # Simulate: gating block not entered when bot_uid is falsy
    is_dm = False
    if not is_dm and bot_uid:
        result = False  # would enter gating
    else:
        result = True  # gating skipped, message processed
    assert result is True


# ---------------------------------------------------------------------------
# Tests: config bridging
# ---------------------------------------------------------------------------

def test_config_bridges_slack_free_response_channels(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  require_mention: false\n"
        "  free_response_channels:\n"
        "    - C0AQWDLHY9M\n"
        "    - C9999999999\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("SLACK_FREE_RESPONSE_CHANNELS", raising=False)

    config = load_gateway_config()

    assert config is not None
    slack_extra = config.platforms[Platform.SLACK].extra
    assert slack_extra.get("require_mention") is False
    assert slack_extra.get("free_response_channels") == ["C0AQWDLHY9M", "C9999999999"]
    # Verify env vars were set by config bridging
    import os as _os
    assert _os.environ["SLACK_REQUIRE_MENTION"] == "false"
    assert _os.environ["SLACK_FREE_RESPONSE_CHANNELS"] == "C0AQWDLHY9M,C9999999999"


def test_config_bridges_slack_reply_in_thread(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  reply_in_thread: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    config = load_gateway_config()

    assert config is not None
    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.extra.get("reply_in_thread") is False

    adapter = SlackAdapter(slack_config)
    assert adapter._resolve_thread_ts(reply_to="171.000", metadata={}) is None

    # Top-level channel messages arrive with metadata.thread_id == reply_to
    # because the inbound handler uses event.ts as a session-keying fallback.
    # Those must be treated as non-threaded so reply_in_thread=false takes
    # effect in channels, not just DMs.
    assert adapter._resolve_thread_ts(
        reply_to="171.000",
        metadata={"thread_id": "171.000"},
    ) is None

    # Real thread replies (reply_to differs from thread parent) must still
    # resolve to the parent thread so conversation context is preserved.
    assert adapter._resolve_thread_ts(
        reply_to="171.500",
        metadata={"thread_id": "171.000"},
    ) == "171.000"


def test_config_bridges_slack_strict_mention(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  strict_mention: true\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SLACK_STRICT_MENTION", raising=False)

    config = load_gateway_config()

    assert config is not None
    import os as _os
    assert _os.environ["SLACK_STRICT_MENTION"] == "true"


# ---------------------------------------------------------------------------
# Regression: strict mode must NOT persist mentions into _mentioned_threads
# ---------------------------------------------------------------------------
# Prevents agent-to-agent ack loops — if a strict-mode bot remembered every
# thread it was mentioned in, the next message from the other agent in that
# thread would re-trigger the bot and defeat the entire feature.

def test_mention_in_strict_mode_does_not_register_thread():
    adapter = _make_adapter(strict_mention=True)
    adapter._bot_user_id = "U_BOT"
    adapter._mentioned_threads = set()
    adapter._MENTIONED_THREADS_MAX = 5000

    thread_ts = "1700000000.100200"
    event_thread_ts = thread_ts  # incoming message is inside an existing thread

    # Mirror the handler's @mention + strict-mode guard that protects
    # _mentioned_threads.add(). If strict is on, we must skip the add.
    text = "<@U_BOT> hello"
    is_mentioned = f"<@{adapter._bot_user_id}>" in text
    assert is_mentioned
    if event_thread_ts and not adapter._slack_strict_mention():
        adapter._mentioned_threads.add(event_thread_ts)

    assert thread_ts not in adapter._mentioned_threads


def test_mention_outside_strict_mode_still_registers_thread():
    adapter = _make_adapter(strict_mention=False)
    adapter._bot_user_id = "U_BOT"
    adapter._mentioned_threads = set()
    adapter._MENTIONED_THREADS_MAX = 5000

    thread_ts = "1700000000.100200"
    event_thread_ts = thread_ts

    text = "<@U_BOT> hello"
    is_mentioned = f"<@{adapter._bot_user_id}>" in text
    assert is_mentioned
    if event_thread_ts and not adapter._slack_strict_mention():
        adapter._mentioned_threads.add(event_thread_ts)

    assert thread_ts in adapter._mentioned_threads
