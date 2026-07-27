"""
Tests for Slack mention gating (require_mention / free_response_channels).

Follows the same pattern as test_whatsapp_group_gating.py.
"""

import sys
import inspect
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

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

import plugins.platforms.slack.adapter as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOT_USER_ID = "U_BOT_123"
CHANNEL_ID = "C0AQWDLHY9M"
OTHER_CHANNEL_ID = "C9999999999"


def _make_adapter(require_mention=None, strict_mention=None, free_response_channels=None,
                  allowed_channels=None, mention_patterns=None):
    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if strict_mention is not None:
        extra["strict_mention"] = strict_mention
    if free_response_channels is not None:
        extra["free_response_channels"] = free_response_channels
    if allowed_channels is not None:
        extra["allowed_channels"] = allowed_channels
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns

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


def test_free_response_channels_bare_int():
    # YAML `free_response_channels: 1491973769726791812` (single bare integer)
    # is loaded as an int and would previously fall through the isinstance(str)
    # branch to return an empty set.  Coerce scalar → str so single-channel
    # config without quoting works as users expect.
    adapter = _make_adapter(free_response_channels=1491973769726791812)
    result = adapter._slack_free_response_channels()
    assert result == {"1491973769726791812"}


def test_free_response_channels_int_list():
    # YAML list form with bare numeric entries — each element should be coerced.
    adapter = _make_adapter(free_response_channels=[1491973769726791812, 99999])
    result = adapter._slack_free_response_channels()
    assert result == {"1491973769726791812", "99999"}


# ---------------------------------------------------------------------------
# Tests: mention gating integration (simulating _handle_slack_message logic)
# ---------------------------------------------------------------------------

def _would_process(adapter, *, is_dm=False, channel_id=CHANNEL_ID,
                   text="hello", mentioned=False, thread_reply=False,
                   active_session=False, channel_type=None):
    """Simulate the mention gating logic from _handle_slack_message.

    Returns True if the message would be processed, False if it would be
    skipped (returned early).

    ``channel_type`` mirrors the real Slack payload ("im" = 1:1 DM,
    "mpim" = group DM, "" = channel). When omitted it is derived from the
    legacy ``is_dm`` flag as a 1:1 IM, preserving existing callers. Gating
    keys off ``is_one_to_one_dm`` (only a true 1:1 IM is exempt); MPIMs are
    shared surfaces and go through the same gating as channels.
    """
    if channel_type is None:
        channel_type = "im" if is_dm else ""
    is_one_to_one_dm = channel_type == "im"

    bot_uid = adapter._team_bot_user_ids.get("T1", adapter._bot_user_id)
    if mentioned:
        text = f"<@{bot_uid}> {text}"
    is_mentioned = bool(
        (bot_uid and f"<@{bot_uid}>" in text)
        or adapter._slack_message_matches_mention_patterns(text)
    )

    if not is_one_to_one_dm and bot_uid:
        # allowed_channels check (whitelist — must pass before other gating)
        allowed = adapter._slack_allowed_channels()
        if allowed and channel_id not in allowed:
            return False

        if channel_id in adapter._slack_free_response_channels():
            return True
        elif not adapter._slack_require_mention():
            return True
        elif adapter._slack_strict_mention() and not is_mentioned:
            return False
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


# ---------------------------------------------------------------------------
# Tests: MPIM / group-DM shared-surface gating (regression for the group-DM
# routing bug introduced by PRs #4633 / #54632 / #54663, which classified
# mpim as a DM and thereby exempted it from mention gating + reaction guards).
# ---------------------------------------------------------------------------

def _reaction_guard(channel_type, is_mentioned):
    """Mirror of the production reaction guard in ``_handle_slack_message``:

        _should_react = (is_one_to_one_dm or is_mentioned) and reactions_enabled

    Only a true 1:1 IM or an explicit @mention earns a reaction; MPIMs and
    channels must be @mentioned. ``test_reaction_guard_pinned_to_production_expression``
    pins this to the real source so the two cannot silently drift.
    """
    is_one_to_one_dm = channel_type == "im"
    return is_one_to_one_dm or is_mentioned


def test_mpim_unmentioned_strict_mention_ignored():
    """MPIM, not mentioned, strict_mention on -> dropped (shared surface)."""
    adapter = _make_adapter(require_mention=True, strict_mention=True,
                            free_response_channels=[])
    assert _would_process(adapter, channel_type="mpim", text="hello") is False


def test_mpim_unmentioned_require_mention_ignored():
    """MPIM, not mentioned, require_mention on (non-strict) -> dropped."""
    adapter = _make_adapter(require_mention=True)
    assert _would_process(adapter, channel_type="mpim", text="hello") is False


def test_mpim_mentioned_processed():
    """MPIM with an @mention is processed like any addressed message."""
    adapter = _make_adapter(require_mention=True, strict_mention=True)
    assert _would_process(adapter, channel_type="mpim", mentioned=True,
                          text="hello") is True


def test_mpim_not_in_allowed_channels_dropped():
    """MPIM absent from a non-empty allowed_channels whitelist is dropped,
    even when mentioned."""
    adapter = _make_adapter(require_mention=True, allowed_channels=["C_ALLOWED"])
    assert _would_process(adapter, channel_type="mpim", channel_id="C_BLOCKED",
                          mentioned=True, text="hello") is False


def test_mpim_in_free_response_processed_without_mention():
    """An MPIM explicitly listed in free_response_channels still opts in."""
    adapter = _make_adapter(require_mention=True,
                            free_response_channels=["G_MPIM"])
    assert _would_process(adapter, channel_type="mpim", channel_id="G_MPIM",
                          text="hello") is True


def test_one_to_one_im_still_exempt():
    """1:1 IM behavior is preserved: mention-exempt regardless of settings."""
    adapter = _make_adapter(require_mention=True, strict_mention=True)
    assert _would_process(adapter, channel_type="im", text="hello") is True


def test_mpim_unmentioned_does_not_react():
    """Reaction guard: only a 1:1 IM or an @mention earns a reaction. An
    unmentioned MPIM message must NOT get :eyes:/:white_check_mark: noise."""
    assert _reaction_guard("mpim", False) is False   # the reported spam case
    assert _reaction_guard("mpim", True) is True      # addressed -> ok
    assert _reaction_guard("im", False) is True        # 1:1 DM -> ok
    assert _reaction_guard("", False) is False         # channel, unmentioned


def test_reaction_guard_pinned_to_production_expression():
    """Regression teeth for the reaction guard.

    ``_reaction_guard`` mirrors the production expression at the
    ``_should_react = (is_one_to_one_dm or is_mentioned) ...`` site in
    ``adapter.py``. This test pins that source line so a revert of the fix
    (back to ``is_dm or is_mentioned``, which reacts to unmentioned MPIMs)
    fails here instead of silently passing a self-referential lambda.
    """
    src = inspect.getsource(SlackAdapter._handle_slack_message)
    assert "(is_one_to_one_dm or is_mentioned)" in src, (
        "reaction guard no longer keys off is_one_to_one_dm — an unmentioned "
        "MPIM would react again (regression of the group-DM fix)"
    )
    assert "(is_dm or is_mentioned)" not in src, (
        "reaction guard reverted to is_dm — MPIMs would react when unmentioned"
    )


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
    _os.environ.pop("SLACK_REQUIRE_MENTION", None)
    _os.environ.pop("SLACK_FREE_RESPONSE_CHANNELS", None)


def test_top_level_slack_settings_do_not_disable_env_token_setup(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  require_mention: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)

    config = load_gateway_config()

    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.enabled is True
    assert slack_config.token == "xoxb-test"
    assert slack_config.extra.get("require_mention") is False
    assert "_enabled_explicit" not in slack_config.extra


def test_explicit_top_level_slack_enabled_false_wins_over_env_token(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  enabled: false\n"
        "  require_mention: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)

    config = load_gateway_config()

    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.enabled is False
    assert slack_config.token == "xoxb-test"
    assert slack_config.extra.get("require_mention") is False
    assert "_enabled_explicit" not in slack_config.extra


def test_explicit_platforms_slack_enabled_false_wins_over_env_token(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  slack:\n"
        "    enabled: false\n"
        "    extra:\n"
        "      reply_in_thread: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    config = load_gateway_config()

    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.enabled is False
    assert slack_config.token == "xoxb-test"
    assert slack_config.extra.get("reply_in_thread") is False
    assert "_enabled_explicit" not in slack_config.extra


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


def test_config_bridges_slack_cron_continuable_surface_toplevel(monkeypatch, tmp_path):
    """The cron_continuable_surface key bridges from a top-level ``slack:`` block
    into slack.extra, mirroring reply_in_thread (specs D1/D6)."""
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  cron_continuable_surface: in_channel\n"
        "  reply_in_thread: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    config = load_gateway_config()

    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.extra.get("cron_continuable_surface") == "in_channel"
    # The adapter resolver reads the bridged key.
    adapter = SlackAdapter(slack_config)
    assert adapter._cron_continuable_surface() == "in_channel"


def test_config_bridges_slack_cron_continuable_surface_nested(monkeypatch, tmp_path):
    """The key also bridges from the nested ``platforms.slack.extra`` path."""
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  slack:\n"
        "    enabled: false\n"
        "    extra:\n"
        "      cron_continuable_surface: in_channel\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    config = load_gateway_config()

    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.extra.get("cron_continuable_surface") == "in_channel"


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
    _os.environ.pop("SLACK_STRICT_MENTION", None)


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


@pytest.mark.asyncio
async def test_block_extraction_debug_log_does_not_include_message_preview(caplog):
    secret_block_text = "private incident token: customer-id-12345"
    adapter = _make_adapter(allowed_channels=[CHANNEL_ID])
    adapter._dedup = MagicMock(is_duplicate=MagicMock(return_value=False))
    adapter._lookup_assistant_thread_metadata = MagicMock(return_value={})
    adapter._channel_team = {}
    adapter._CHANNEL_TEAM_MAX = 10000
    # Wave-2 mention gating probes users.info for bot detection on several
    # paths; this fixture has no web client, so pin the sender as human.
    adapter._resolve_user_is_bot = AsyncMock(return_value=False)
    adapter._resolve_user_name = AsyncMock(return_value="testuser")
    adapter.handle_message = AsyncMock()

    event = {
        "channel": OTHER_CHANNEL_ID,
        "channel_type": "channel",
        "ts": "1710000000.000100",
        "team": "T1",
        "user": "U_USER",
        # Human-authored messages carry client_msg_id; without it the
        # unlabeled-bot probe path calls users.info, which this fixture
        # doesn't wire up.
        "client_msg_id": "cmid-block-priv",
        "text": "<@U_BOT_123> see quoted message",
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_quote",
                        "elements": [
                            {
                                "type": "rich_text_section",
                                "elements": [
                                    {"type": "text", "text": secret_block_text}
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.slack.adapter"):
        await adapter._handle_slack_message(event)

    assert "extracted additional text from blocks" in caplog.text
    assert "chars=" in caplog.text
    assert secret_block_text not in caplog.text


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
    _os.environ.pop("SLACK_ALLOWED_CHANNELS", None)


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


# ---------------------------------------------------------------------------
# Tests: mention_patterns (wake words) — parity with other adapters (#50732)
# ---------------------------------------------------------------------------

def test_mention_patterns_default_no_match(monkeypatch):
    monkeypatch.delenv("SLACK_MENTION_PATTERNS", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_mention_patterns() == []
    assert adapter._slack_message_matches_mention_patterns("hello there") is False


def test_mention_patterns_list_matches():
    adapter = _make_adapter(mention_patterns=["hey hermes", "hermes,"])
    assert adapter._slack_message_matches_mention_patterns("hey hermes, you there?") is True
    assert adapter._slack_message_matches_mention_patterns("just chatting") is False


def test_mention_patterns_case_insensitive():
    adapter = _make_adapter(mention_patterns=["hey hermes"])
    assert adapter._slack_message_matches_mention_patterns("HEY HERMES!") is True


def test_mention_patterns_single_string():
    adapter = _make_adapter(mention_patterns="^hermes")
    assert adapter._slack_message_matches_mention_patterns("hermes do this") is True
    assert adapter._slack_message_matches_mention_patterns("ok hermes") is False


def test_mention_patterns_invalid_regex_skipped_without_crash():
    # An invalid pattern is dropped; valid siblings still work.
    adapter = _make_adapter(mention_patterns=["(unclosed", "hey hermes"])
    assert adapter._slack_message_matches_mention_patterns("hey hermes") is True


def test_mention_patterns_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_MENTION_PATTERNS", '["hey hermes", "hermes,"]')
    adapter = _make_adapter()  # no config value -> falls back to env
    assert adapter._slack_message_matches_mention_patterns("hey hermes") is True


def test_mention_patterns_env_var_csv_fallback_splits_patterns(monkeypatch):
    monkeypatch.setenv("SLACK_MENTION_PATTERNS", "hey hermes,hermes,")
    adapter = _make_adapter()  # no config value -> falls back to env

    patterns = adapter._slack_mention_patterns()

    assert [pattern.pattern for pattern in patterns] == ["hey hermes", "hermes"]
    assert adapter._slack_message_matches_mention_patterns("hey hermes") is True


def test_mention_patterns_trigger_in_channel_without_literal_mention():
    """A wake word triggers the bot in a channel even with require_mention on."""
    adapter = _make_adapter(require_mention=True, mention_patterns=["hey hermes"])
    assert _would_process(adapter, text="hey hermes what's the status") is True
    # Unrelated channel chatter is still ignored.
    assert _would_process(adapter, text="lunch anyone?") is False


# ---------------------------------------------------------------------------
# Tests: Block-Kit-only mention detection (#52387)
# ---------------------------------------------------------------------------

from plugins.platforms.slack.adapter import _slack_mention_detection_text  # noqa: E402


def _blockkit_mention_event(bot_user_id=BOT_USER_ID, flat_text="Release notification"):
    """A Slack event whose @mention lives ONLY inside Block Kit blocks."""
    return {
        "text": flat_text,
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {"type": "text", "text": "Hey "},
                            {"type": "user", "user_id": bot_user_id},
                            {"type": "text", "text": "! I will do a release"},
                        ],
                    }
                ],
            }
        ],
    }


def test_mention_detection_text_recovers_blockkit_mention():
    event = _blockkit_mention_event()
    merged = _slack_mention_detection_text(event)
    # The flat text alone never contains the mention...
    assert f"<@{BOT_USER_ID}>" not in event.get("text", "")
    # ...but the merged detection text does.
    assert f"<@{BOT_USER_ID}>" in merged


def test_mention_detection_text_no_blocks_returns_flat_text():
    event = {"text": f"<@{BOT_USER_ID}> hello"}
    assert _slack_mention_detection_text(event) == f"<@{BOT_USER_ID}> hello"


def test_mention_detection_text_no_mention_anywhere():
    event = {
        "text": "lunch anyone?",
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": "lunch anyone?"}],
                    }
                ],
            }
        ],
    }
    assert f"<@{BOT_USER_ID}>" not in _slack_mention_detection_text(event)


def test_mention_detection_text_ignores_quoted_blockkit_mention():
    """A mention inside rich_text_quote (forwarded content) must NOT count."""
    event = {
        "text": "please review",
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_quote",
                        "elements": [
                            {
                                "type": "rich_text_section",
                                "elements": [
                                    {"type": "text", "text": "Contains "},
                                    {"type": "user", "user_id": BOT_USER_ID},
                                    {"type": "text", "text": " in quoted text"},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    assert f"<@{BOT_USER_ID}>" not in _slack_mention_detection_text(event)
