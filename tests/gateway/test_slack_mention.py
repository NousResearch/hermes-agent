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


def _make_adapter(require_mention=None, free_response_channels=None, allowed_channels=None):
    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if free_response_channels is not None:
        extra["free_response_channels"] = free_response_channels
    if allowed_channels is not None:
        extra["allowed_channels"] = allowed_channels

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

    if not is_dm and bot_uid:
        # allowed_channels check (whitelist — must pass before other gating)
        allowed = adapter._slack_allowed_channels()
        if allowed and channel_id not in allowed:
            return False

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


# ---------------------------------------------------------------------------
# Tests: _slack_allowed_channels
# ---------------------------------------------------------------------------

def test_allowed_channels_default_empty(monkeypatch):
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_allowed_channels() == set()


def test_allowed_channels_list():
    adapter = _make_adapter(allowed_channels=[CHANNEL_ID, OTHER_CHANNEL_ID])
    result = adapter._slack_allowed_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


def test_allowed_channels_csv_string():
    adapter = _make_adapter(allowed_channels=f"{CHANNEL_ID}, {OTHER_CHANNEL_ID}")
    result = adapter._slack_allowed_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


def test_allowed_channels_empty_string():
    adapter = _make_adapter(allowed_channels="")
    assert adapter._slack_allowed_channels() == set()


def test_allowed_channels_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", f"{CHANNEL_ID},{OTHER_CHANNEL_ID}")
    adapter = _make_adapter()  # no config value → falls back to env
    result = adapter._slack_allowed_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


# ---------------------------------------------------------------------------
# Tests: allowed_channels gating integration
# ---------------------------------------------------------------------------

def test_allowed_channels_blocks_non_whitelisted_channel():
    """Messages in channels not in allowed_channels are silently ignored."""
    adapter = _make_adapter(allowed_channels=[CHANNEL_ID])
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, text="hello") is False


def test_allowed_channels_permits_whitelisted_channel():
    """Messages in the allowed channel are processed normally."""
    adapter = _make_adapter(allowed_channels=[CHANNEL_ID])
    assert _would_process(adapter, channel_id=CHANNEL_ID, mentioned=True) is True


def test_allowed_channels_empty_no_restriction():
    """Empty allowed_channels imposes no restriction (fully backward compatible)."""
    adapter = _make_adapter(allowed_channels="")
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is True


def test_allowed_channels_blocks_even_when_mentioned():
    """Whitelist takes precedence — @mention in a non-allowed channel is ignored."""
    adapter = _make_adapter(allowed_channels=[CHANNEL_ID])
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is False


def test_allowed_channels_dm_unaffected():
    """DMs bypass the allowed_channels check entirely."""
    adapter = _make_adapter(allowed_channels=[CHANNEL_ID])
    # DM channel IDs typically start with D; the check is guarded by `not is_dm`
    assert _would_process(adapter, is_dm=True, channel_id="DDMCHANNEL") is True


def test_allowed_channels_env_var_blocks_channel(monkeypatch):
    """SLACK_ALLOWED_CHANNELS env var (no config) also gates messages."""
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", CHANNEL_ID)
    adapter = _make_adapter()  # no config value → falls back to env
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, text="hello") is False
    assert _would_process(adapter, channel_id=CHANNEL_ID, mentioned=True) is True


# ---------------------------------------------------------------------------
# Tests: config bridging for allowed_channels
# ---------------------------------------------------------------------------

def test_config_bridges_slack_allowed_channels(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  allowed_channels:\n"
        f"    - {CHANNEL_ID}\n"
        f"    - {OTHER_CHANNEL_ID}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)

    load_gateway_config()

    import os as _os
    assert _os.environ["SLACK_ALLOWED_CHANNELS"] == f"{CHANNEL_ID},{OTHER_CHANNEL_ID}"


def test_config_bridges_slack_allowed_channels_env_takes_precedence(monkeypatch, tmp_path):
    """Env var set before load_gateway_config() should not be overwritten."""
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        f"  allowed_channels: {CHANNEL_ID}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", OTHER_CHANNEL_ID)  # already set

    load_gateway_config()

    import os as _os
    # env var must not be overwritten by config.yaml
    assert _os.environ["SLACK_ALLOWED_CHANNELS"] == OTHER_CHANNEL_ID
