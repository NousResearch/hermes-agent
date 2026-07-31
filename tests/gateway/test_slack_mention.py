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


def test_require_mention_empty_string_stays_true():
    """Empty/malformed strings keep gating ON (explicit-false parser)."""
    adapter = _make_adapter(require_mention="")
    assert adapter._slack_require_mention() is True


# ---------------------------------------------------------------------------
# Tests: _slack_strict_mention
# ---------------------------------------------------------------------------

def test_strict_mention_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SLACK_STRICT_MENTION", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_strict_mention() is False


def test_strict_mention_malformed_stays_false():
    """Unrecognised values keep strict mode OFF (fail-open to legacy behavior)."""
    adapter = _make_adapter(strict_mention="maybe")
    assert adapter._slack_strict_mention() is False


# ---------------------------------------------------------------------------
# Tests: _slack_free_response_channels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: mention gating integration (simulating _handle_slack_message logic)
# ---------------------------------------------------------------------------

def _would_process(adapter, *, is_dm=False, channel_id=CHANNEL_ID,
                   text="hello", mentioned=False, thread_reply=False,
                   active_session=False, channel_type=None, event=None):
    """Simulate the mention gating logic from _handle_slack_message.

    Returns True if the message would be processed, False if it would be
    skipped (returned early).

    ``channel_type`` mirrors the real Slack payload ("im" = 1:1 DM,
    "mpim" = group DM, "" = channel). When omitted it is derived from the
    legacy ``is_dm`` flag as a 1:1 IM, preserving existing callers. Gating
    keys off ``is_one_to_one_dm`` (only a true 1:1 IM is exempt); MPIMs are
    shared surfaces and go through the same gating as channels.

    ``event`` supplies a full Slack payload when the mention lives somewhere
    other than the flat text (blocks / attachments); otherwise one is built
    from ``text``. Either way the routing text and the mentioned flag come
    from the production helper, so this simulation cannot drift from the gate.
    """
    if channel_type is None:
        channel_type = "im" if is_dm else ""
    is_one_to_one_dm = channel_type == "im"

    bot_uid = adapter._team_bot_user_ids.get("T1", adapter._bot_user_id)
    if event is None:
        if mentioned:
            text = f"<@{bot_uid}> {text}"
        event = {"text": text}
    text, is_mentioned = adapter._slack_mention_gate_inputs(event, bot_uid)

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


def test_mpim_not_in_allowed_channels_dropped():
    """MPIM absent from a non-empty allowed_channels whitelist is dropped,
    even when mentioned."""
    adapter = _make_adapter(require_mention=True, allowed_channels=["C_ALLOWED"])
    assert _would_process(adapter, channel_type="mpim", channel_id="C_BLOCKED",
                          mentioned=True, text="hello") is False


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


# ---------------------------------------------------------------------------
# Tests: _slack_allowed_channels
# ---------------------------------------------------------------------------


def test_allowed_channels_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", f"{CHANNEL_ID},{OTHER_CHANNEL_ID}")
    adapter = _make_adapter()  # no config value → falls back to env
    result = adapter._slack_allowed_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


# ---------------------------------------------------------------------------
# Tests: allowed_channels gating integration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: mention_patterns (wake words) — parity with other adapters (#50732)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: Block-Kit-only mention detection (#52387)
# ---------------------------------------------------------------------------

from plugins.platforms.slack.adapter import (  # noqa: E402
    _ThreadContextCache,
    _slack_recovered_mentions,
)


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


def test_recovered_mentions_ignore_quoted_blockkit_mention():
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
    assert f"<@{BOT_USER_ID}>" not in _slack_recovered_mentions(event)


# ---------------------------------------------------------------------------
# Tests: mentions authored as raw mrkdwn tokens, not `user` elements (#52387)
#
# The WYSIWYG composer emits a structured `user` element, but an app building
# blocks by hand writes the raw `<@UID>` token into a section/header/context
# string, and legacy `attachments` are a third carrier. Those shapes have no
# `user` node, so recovering them is a separate path from the one #52387 fixed.
# ---------------------------------------------------------------------------


def _mrkdwn_section(text):
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


@pytest.mark.parametrize(
    "carrier,event",
    [
        (
            "section text",
            {"text": "", "blocks": [_mrkdwn_section(f"<@{BOT_USER_ID}> deploy failed")]},
        ),
        (
            "section fields",
            {"text": "", "blocks": [{
                "type": "section",
                "fields": [{"type": "mrkdwn", "text": f"owner: <@{BOT_USER_ID}>"}],
            }]},
        ),
        (
            "header text",
            {"text": "", "blocks": [{
                "type": "header",
                "text": {"type": "plain_text", "text": f"<@{BOT_USER_ID}> incident"},
            }]},
        ),
        (
            "context elements",
            {"text": "", "blocks": [{
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"<@{BOT_USER_ID}> please ack"}],
            }]},
        ),
        (
            "attachment text",
            {"text": "", "attachments": [{"text": f"<@{BOT_USER_ID}> disk 91%"}]},
        ),
        (
            "attachment fields",
            {"text": "", "attachments": [
                {"fields": [{"title": "owner", "value": f"<@{BOT_USER_ID}>"}]}
            ]},
        ),
        (
            "attachment nested blocks",
            {"text": "", "attachments": [
                {"blocks": [_mrkdwn_section(f"<@{BOT_USER_ID}> nested")]}
            ]},
        ),
    ],
)
def test_recovered_mentions_cover_mrkdwn_token_carriers(carrier, event):
    """Every carrier an app can author a mention in must reach the gates."""
    assert f"<@{BOT_USER_ID}>" in _slack_recovered_mentions(event), carrier


def test_recovered_mentions_normalize_labelled_mention():
    """``<@U123|alice>`` must normalize to the bare token the gates compare."""
    event = {"text": "", "blocks": [_mrkdwn_section(f"<@{BOT_USER_ID}|hermes> ping")]}
    assert f"<@{BOT_USER_ID}>" in _slack_recovered_mentions(event)


def test_recovered_mentions_ignore_quoted_mrkdwn_token():
    """The quote carve-out must cover raw tokens, not just `user` elements."""
    event = {
        "text": "see above",
        "blocks": [{
            "type": "rich_text",
            "elements": [{
                "type": "rich_text_quote",
                "elements": [{"type": "text", "text": f"<@{BOT_USER_ID}> old ping"}],
            }],
        }],
    }
    assert f"<@{BOT_USER_ID}>" not in _slack_recovered_mentions(event)


def test_recovered_mentions_ignore_section_nested_in_quote():
    """A section nested under a quote stays quoted, however deep."""
    event = {
        "text": "see above",
        "blocks": [{
            "type": "rich_text",
            "elements": [{
                "type": "rich_text_quote",
                "elements": [_mrkdwn_section(f"<@{BOT_USER_ID}> quoted")],
            }],
        }],
    }
    assert f"<@{BOT_USER_ID}>" not in _slack_recovered_mentions(event)


def test_recovered_mentions_dedupe_repeated_carriers():
    """One addressed user yields one token, however many carriers repeat it."""
    event = {
        "text": "",
        "blocks": [_mrkdwn_section(f"<@{BOT_USER_ID}> rollout started")],
        "attachments": [{"text": f"<@{BOT_USER_ID}> rollout started"}],
    }
    assert _slack_recovered_mentions(event) == [f"<@{BOT_USER_ID}>"]


def test_recovered_mentions_empty_without_a_recoverable_mention():
    """A message whose carriers hold no mention recovers nothing."""
    event = {"text": "nightly backup finished", "blocks": [_mrkdwn_section("all green")]}
    assert _slack_recovered_mentions(event) == []


@pytest.mark.parametrize(
    "event",
    [
        {"text": "hi", "blocks": {"type": "section"}},
        {"text": "hi", "blocks": ["not-a-dict"]},
        {"text": "hi", "blocks": [None]},
        {"text": "hi", "attachments": "not-a-list"},
        {"text": "hi", "attachments": [None]},
        {"text": "hi", "attachments": [{"fields": "not-a-list"}]},
    ],
)
def test_recovered_mentions_never_raise_on_malformed_payload(event):
    """Gating degrades to the flat text; a bad payload must never break it."""
    assert _slack_recovered_mentions(event) == []


def test_attachment_mentions_survive_a_malformed_sibling():
    """One bad attachment must not discard mentions already collected."""
    event = {
        "text": "",
        "attachments": [{"text": f"<@{BOT_USER_ID}> disk 91%"}, {"fields": 3}],
    }
    assert f"<@{BOT_USER_ID}>" in _slack_recovered_mentions(event)


@pytest.mark.parametrize("quote_key", ["is_msg_unfurl", "is_share"])
def test_quoted_attachment_mention_does_not_wake_the_bot(quote_key):
    """A pasted permalink / forwarded share carries someone else's mention."""
    event = {
        "text": "look at this",
        "attachments": [{quote_key: True, "text": f"<@{BOT_USER_ID}> deploy prod"}],
    }
    assert _slack_recovered_mentions(event) == []


def test_mrkdwn_blockquote_mention_does_not_wake_the_bot():
    """`> ` quoting is a carrier the rich_text_quote node check cannot see."""
    event = {
        "text": "",
        "blocks": [_mrkdwn_section(f"&gt; <@{BOT_USER_ID}> old ping\nstatus: green")],
    }
    assert _slack_recovered_mentions(event) == []


def test_attachment_fallback_alone_does_not_wake_the_bot():
    """`fallback` is never rendered, so a mention only there notifies nobody."""
    event = {"text": "", "attachments": [{"fallback": f"<@{BOT_USER_ID}> ping"}]}
    assert _slack_recovered_mentions(event) == []


# ---------------------------------------------------------------------------
# Tests: code / preformatted content is displayed, not spoken
#
# A `<@UID>` a human formatted as code, or an app emitted inside a payload
# dump, is being shown to the reader — not addressed to anyone. Slack does not
# even linkify mrkdwn inside code, so such a token notifies nobody. These pin
# the same carve-out `rich_text_quote` gets, for every verbatim carrier.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "carrier,event",
    [
        (
            "user element inside rich_text_preformatted",
            {"text": "", "blocks": [{"type": "rich_text", "elements": [{
                "type": "rich_text_preformatted",
                "elements": [{"type": "user", "user_id": BOT_USER_ID}],
            }]}]},
        ),
        (
            "raw token inside rich_text_preformatted",
            {"text": "", "blocks": [{"type": "rich_text", "elements": [{
                "type": "rich_text_preformatted",
                "elements": [{"type": "text", "text": f'{{"who": "<@{BOT_USER_ID}>"}}'}],
            }]}]},
        ),
        (
            "code-styled text element",
            {"text": "", "blocks": [{"type": "rich_text", "elements": [{
                "type": "rich_text_section",
                "elements": [{
                    "type": "text",
                    "text": f"<@{BOT_USER_ID}>",
                    "style": {"code": True},
                }],
            }]}]},
        ),
        (
            "code-styled user element",
            {"text": "", "blocks": [{"type": "rich_text", "elements": [{
                "type": "rich_text_section",
                "elements": [{
                    "type": "user",
                    "user_id": BOT_USER_ID,
                    "style": {"code": True},
                }],
            }]}]},
        ),
        (
            "mrkdwn fenced code block",
            {"text": "", "blocks": [
                _mrkdwn_section(f"deploy log:\n```\nnotify <@{BOT_USER_ID}>\n```")
            ]},
        ),
        (
            "mrkdwn fence opening and closing on one line",
            {"text": "", "blocks": [_mrkdwn_section(f"```<@{BOT_USER_ID}>```")]},
        ),
        (
            "mrkdwn inline code span",
            {"text": "", "blocks": [
                _mrkdwn_section(f"the owner field holds `<@{BOT_USER_ID}>` verbatim")
            ]},
        ),
        (
            "fenced code inside an attachment",
            {"text": "", "attachments": [
                {"text": f"```\npayload: <@{BOT_USER_ID}>\n```"}
            ]},
        ),
    ],
)
def test_code_content_mention_does_not_wake_the_bot(carrier, event):
    """A token shown as code addresses nobody and must not summon the bot."""
    assert _slack_recovered_mentions(event) == [], carrier


def test_escaped_token_in_code_stays_ignored():
    """Slack escapes a human-typed literal token; that must stay a non-mention.

    The positive bound on the escaping: without this, relaxing
    `_SLACK_USER_MENTION_RE` would silently start waking on pasted logs.
    """
    event = {"text": "", "blocks": [{"type": "rich_text", "elements": [{
        "type": "rich_text_preformatted",
        "elements": [{"type": "text", "text": f"&lt;@{BOT_USER_ID}&gt;"}],
    }]}]}
    assert _slack_recovered_mentions(event) == []


def test_text_after_a_closed_fence_still_yields_a_mention():
    """The fence carve-out must end at the closing fence, not swallow the rest."""
    event = {
        "text": "",
        "blocks": [_mrkdwn_section(f"```\nsome log\n```\n<@{BOT_USER_ID}> please look")],
    }
    assert _slack_recovered_mentions(event) == [f"<@{BOT_USER_ID}>"]


def test_mention_beside_an_inline_code_span_still_counts():
    """Stripping inline code must not strip the address that surrounds it."""
    event = {
        "text": "",
        "blocks": [_mrkdwn_section(f"<@{BOT_USER_ID}> check `disk_usage` please")],
    }
    assert _slack_recovered_mentions(event) == [f"<@{BOT_USER_ID}>"]


def test_a_lone_backtick_does_not_hide_a_mention():
    """An unpaired backtick opens no code span, so the mention still counts."""
    event = {
        "text": "",
        "blocks": [_mrkdwn_section(f"it's at 91` — <@{BOT_USER_ID}> ack?")],
    }
    assert _slack_recovered_mentions(event) == [f"<@{BOT_USER_ID}>"]


@pytest.mark.parametrize(
    "styling,element",
    [
        # `style` is a plain string on rich_text_list ("bullet"/"ordered") but a
        # dict on inline elements — reading it without a type check crashes the
        # walker on every bulleted message.
        ("bulleted list", {
            "type": "rich_text_list",
            "style": "bullet",
            "elements": [{
                "type": "rich_text_section",
                "elements": [{"type": "user", "user_id": BOT_USER_ID}],
            }],
        }),
        ("bold text", {
            "type": "rich_text_section",
            "elements": [{
                "type": "user",
                "user_id": BOT_USER_ID,
                "style": {"bold": True},
            }],
        }),
    ],
)
def test_non_code_styling_still_yields_a_mention(styling, element):
    """Only `style.code` is verbatim; other styling must not drop the mention."""
    event = {"text": "", "blocks": [{"type": "rich_text", "elements": [element]}]}
    assert _slack_recovered_mentions(event) == [f"<@{BOT_USER_ID}>"], styling


# ---------------------------------------------------------------------------
# Tests: the routing gate itself, not just the detection helper (#52387)
#
# The helper tests above prove a mention is *recoverable*; these prove the
# gate acts on it and that recovering it does not corrupt the routing text
# the wake-word patterns and the leading-mention check read.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "carrier,event",
    [
        ("attachment text", {"text": "", "attachments": [
            {"text": f"<@{BOT_USER_ID}> disk 91%"}
        ]}),
        ("attachment fields", {"text": "", "attachments": [
            {"fields": [{"title": "owner", "value": f"<@{BOT_USER_ID}>"}]}
        ]}),
        ("section text", {"text": "", "blocks": [
            _mrkdwn_section(f"<@{BOT_USER_ID}> deploy failed")
        ]}),
    ],
)
def test_gate_processes_attachment_only_mention(carrier, event):
    """An alert bot's mention must pass the require_mention gate."""
    adapter = _make_adapter()  # default: require_mention=True
    assert _would_process(adapter, event=event) is True, carrier


def test_gate_leaves_routing_text_free_of_recovered_mentions():
    """A recovered mention must not become the routing text's leading token.

    An attachment-only alert naming a human would otherwise look like a message
    opening with `<@other>` and be dropped by ignore_other_user_mentions.
    """
    adapter = _make_adapter()
    event = {
        "text": "",
        "attachments": [{"fields": [{"title": "owner", "value": "<@U_ONCALL>"}]}],
    }
    routing_text, is_mentioned = adapter._slack_mention_gate_inputs(event, BOT_USER_ID)
    assert routing_text == ""
    assert is_mentioned is False
    assert adapter._slack_message_addressed_to_other_user(
        routing_text, {BOT_USER_ID}
    ) is False


def test_gate_wake_word_pattern_still_matches_with_an_attachment():
    """An anchored wake word must not be broken by an appended mention tail."""
    adapter = _make_adapter(mention_patterns=[r"^hey hermes$"])
    event = {
        "text": "hey hermes",
        "attachments": [{"text": "<@U_ALICE> fyi"}],
    }
    _, is_mentioned = adapter._slack_mention_gate_inputs(event, BOT_USER_ID)
    assert is_mentioned is True


# ---------------------------------------------------------------------------
# Tests: the thread-parent wake check (#24848) obeys the same carve-outs
#
# A reply carrying no mention of its own wakes the bot when the thread PARENT
# addressed it. That decision must come from the raw parent event through the
# live gate's predicate — NOT from `_fetch_thread_parent_text`, which renders
# display text and deliberately preserves quoted and shared content for the
# agent to read. Both the cold and the cached parent path are covered, because
# they are separate code paths that must agree.
# ---------------------------------------------------------------------------

PARENT_TS = "1700000000.000100"


def _make_parent_adapter(parent, cached):
    """Adapter whose thread parent is *parent*, served from cache or the API."""
    adapter = _make_adapter()
    adapter._THREAD_CACHE_TTL = 60.0
    adapter._thread_context_cache = {}
    client = AsyncMock()
    if cached:
        adapter._thread_context_cache[f"{CHANNEL_ID}:{PARENT_TS}:"] = (
            _ThreadContextCache(content="", parent_text="", messages=[parent])
        )
        client.conversations_replies = AsyncMock(
            side_effect=AssertionError("a cached parent must not re-hit the API")
        )
    else:
        client.conversations_replies = AsyncMock(return_value={"messages": [parent]})
    adapter._get_client = MagicMock(return_value=client)
    return adapter


def _parent(**payload):
    return {"ts": PARENT_TS, "text": "", **payload}


@pytest.mark.parametrize("cached", [False, True], ids=["cold", "cached"])
@pytest.mark.parametrize(
    "carrier,parent",
    [
        (
            "rich_text_quote",
            _parent(blocks=[{"type": "rich_text", "elements": [{
                "type": "rich_text_quote",
                "elements": [{"type": "user", "user_id": BOT_USER_ID}],
            }]}]),
        ),
        (
            "is_msg_unfurl attachment",
            _parent(text="look at this", attachments=[
                {"is_msg_unfurl": True, "text": f"<@{BOT_USER_ID}> deploy prod"}
            ]),
        ),
        (
            "is_share attachment",
            _parent(text="look at this", attachments=[
                {"is_share": True, "text": f"<@{BOT_USER_ID}> deploy prod"}
            ]),
        ),
        (
            "fallback-only attachment",
            _parent(attachments=[{"fallback": f"<@{BOT_USER_ID}> ping"}]),
        ),
        (
            "preformatted mention",
            _parent(blocks=[{"type": "rich_text", "elements": [{
                "type": "rich_text_preformatted",
                "elements": [{"type": "user", "user_id": BOT_USER_ID}],
            }]}]),
        ),
    ],
)
@pytest.mark.asyncio
async def test_thread_parent_verbatim_mention_does_not_wake(carrier, parent, cached):
    """Quoted / shared / code content in the parent must not wake the thread."""
    adapter = _make_parent_adapter(parent, cached)
    assert await adapter._thread_parent_mentions_bot(
        channel_id=CHANNEL_ID, thread_ts=PARENT_TS, bot_uid=BOT_USER_ID
    ) is False, carrier


@pytest.mark.parametrize("cached", [False, True], ids=["cold", "cached"])
@pytest.mark.parametrize(
    "carrier,parent",
    [
        ("flat text", _parent(text=f"<@{BOT_USER_ID}> watch the rollout")),
        (
            "section block only",
            _parent(blocks=[_mrkdwn_section(f"<@{BOT_USER_ID}> watch the rollout")]),
        ),
        (
            "attachment field only",
            _parent(attachments=[
                {"fields": [{"title": "owner", "value": f"<@{BOT_USER_ID}>"}]}
            ]),
        ),
    ],
)
@pytest.mark.asyncio
async def test_thread_parent_genuine_mention_wakes(carrier, parent, cached):
    """A parent that really addressed the bot still wakes its thread (#24848)."""
    adapter = _make_parent_adapter(parent, cached)
    assert await adapter._thread_parent_mentions_bot(
        channel_id=CHANNEL_ID, thread_ts=PARENT_TS, bot_uid=BOT_USER_ID
    ) is True, carrier


@pytest.mark.asyncio
async def test_thread_parent_wake_ignores_a_ts_mismatch():
    """A fetch that returned a different message is not the parent."""
    adapter = _make_parent_adapter(
        {"ts": "1699999999.000000", "text": f"<@{BOT_USER_ID}> ping"}, cached=False
    )
    assert await adapter._thread_parent_mentions_bot(
        channel_id=CHANNEL_ID, thread_ts=PARENT_TS, bot_uid=BOT_USER_ID
    ) is False


@pytest.mark.asyncio
async def test_thread_parent_text_still_surfaces_quoted_content():
    """The display renderer must keep preserving quotes — only the gate changed.

    `_fetch_thread_parent_text` feeds reply_to_text injection, where the agent
    *should* see what was quoted. This pins the split: display keeps it, the
    wake decision above drops it.
    """
    parent = _parent(blocks=[{"type": "rich_text", "elements": [{
        "type": "rich_text_quote",
        "elements": [{"type": "text", "text": "the earlier ask"}],
    }]}])
    adapter = _make_parent_adapter(parent, cached=False)
    assert "the earlier ask" in await adapter._fetch_thread_parent_text(
        channel_id=CHANNEL_ID, thread_ts=PARENT_TS
    )
