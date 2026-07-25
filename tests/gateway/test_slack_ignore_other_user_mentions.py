"""Tests for the Slack ``ignore_other_user_mentions`` option.

When enabled, the bot stays silent on a channel/thread message that opens by
@-mentioning someone other than itself (a message *addressed to* another user),
unless the bot is also mentioned. This is Slack parity for the Discord option
of the same name (PR #33501), adapted to Slack's thread model: the trigger is a
*leading* mention, so a message that merely references another user mid-sentence
(e.g. "loop in @rasha") still reaches the bot.

Helper-level tests exercise the real config/parse methods directly; the
integration tests drive the real ``SlackAdapter._handle_slack_message`` so the
gate is verified end-to-end rather than against a re-implementation.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same pattern as test_slack_mention.py)
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

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


BOT_USER_ID = "U_BOT_123"
OTHER_USER_ID = "U_OTHER_456"
CHANNEL_ID = "C0AQWDLHY9M"


def _make_adapter(**extra):
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra=dict(extra))
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {}
    return adapter


# ---------------------------------------------------------------------------
# _slack_ignore_other_user_mentions()
# ---------------------------------------------------------------------------

def test_ignore_other_user_mentions_defaults_off(monkeypatch):
    monkeypatch.delenv("SLACK_IGNORE_OTHER_USER_MENTIONS", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_ignore_other_user_mentions() is False


def test_ignore_other_user_mentions_extra_true():
    adapter = _make_adapter(ignore_other_user_mentions=True)
    assert adapter._slack_ignore_other_user_mentions() is True


def test_ignore_other_user_mentions_extra_false():
    adapter = _make_adapter(ignore_other_user_mentions=False)
    assert adapter._slack_ignore_other_user_mentions() is False


def test_ignore_other_user_mentions_extra_string_forms():
    assert _make_adapter(ignore_other_user_mentions="on")._slack_ignore_other_user_mentions() is True
    assert _make_adapter(ignore_other_user_mentions="true")._slack_ignore_other_user_mentions() is True
    assert _make_adapter(ignore_other_user_mentions="off")._slack_ignore_other_user_mentions() is False


def test_ignore_other_user_mentions_env_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_IGNORE_OTHER_USER_MENTIONS", "true")
    adapter = _make_adapter()
    assert adapter._slack_ignore_other_user_mentions() is True


def test_ignore_other_user_mentions_extra_overrides_env(monkeypatch):
    monkeypatch.setenv("SLACK_IGNORE_OTHER_USER_MENTIONS", "true")
    adapter = _make_adapter(ignore_other_user_mentions=False)
    assert adapter._slack_ignore_other_user_mentions() is False


# ---------------------------------------------------------------------------
# _slack_message_addressed_to_other_user()
# ---------------------------------------------------------------------------

SELF_UIDS = {BOT_USER_ID}


def _addressed(text):
    return _make_adapter()._slack_message_addressed_to_other_user(text, SELF_UIDS)


def test_addressed_leading_other_user_mention():
    assert _addressed(f"<@{OTHER_USER_ID}> check this out") is True


def test_addressed_leading_bot_mention_is_not_other():
    assert _addressed(f"<@{BOT_USER_ID}> hello") is False


def test_addressed_pipe_form_other_user():
    assert _addressed(f"<@{OTHER_USER_ID}|rasha> check this out") is True


def test_addressed_pipe_form_bot_is_not_other():
    assert _addressed(f"<@{BOT_USER_ID}|hermes> hello") is False


def test_addressed_no_mention():
    assert _addressed("hello there, thanks for that") is False


def test_addressed_mid_text_mention_not_leading():
    assert _addressed(f"can you loop in <@{OTHER_USER_ID}> on this?") is False


def test_addressed_leading_whitespace_then_other_mention():
    assert _addressed(f"   <@{OTHER_USER_ID}> take a look") is True


def test_addressed_empty_and_blank():
    assert _addressed("") is False
    assert _addressed("   ") is False


def test_addressed_channel_and_broadcast_tokens_are_not_users():
    # <!here>, <!channel>, and channel refs address the room, not a person.
    assert _addressed("<!here> standup in 5") is False
    assert _addressed("<#C0000000000|general> see here") is False


# ---------------------------------------------------------------------------
# _slack_message_mentions_self()
# ---------------------------------------------------------------------------

def _mentions_self(text):
    return _make_adapter()._slack_message_mentions_self(text, SELF_UIDS)


def test_mentions_self_plain_form():
    assert _mentions_self(f"hello <@{BOT_USER_ID}>") is True


def test_mentions_self_pipe_form():
    assert _mentions_self(f"hello <@{BOT_USER_ID}|hermes>") is True


def test_mentions_self_other_user_only():
    assert _mentions_self(f"hello <@{OTHER_USER_ID}|rasha>") is False


def test_mentions_self_id_prefix_is_not_a_match():
    # <@U_BOT_123X> is a different user whose ID merely starts with ours.
    assert _mentions_self(f"hello <@{BOT_USER_ID}X>") is False


def test_mentions_self_empty():
    assert _mentions_self("") is False


# ---------------------------------------------------------------------------
# Integration: real _handle_slack_message
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = BOT_USER_ID
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    # Keep gating driven by config.extra, not ambient env.
    monkeypatch.delenv("SLACK_IGNORE_OTHER_USER_MENTIONS", raising=False)
    monkeypatch.delenv("SLACK_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)


def _event(text, ts, thread_ts=None):
    event = {
        "channel": CHANNEL_ID,
        "channel_type": "channel",
        "user": "U_HUMAN",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


async def _run(adapter, event):
    with patch.object(
        adapter, "_resolve_user_name", new=AsyncMock(return_value="human")
    ), patch.object(
        adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)
    ), patch.object(
        adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value="")
    ), patch.object(
        adapter, "_has_active_session_for_thread", return_value=False
    ):
        await adapter._handle_slack_message(event)


@pytest.mark.asyncio
async def test_free_response_ignores_message_addressed_to_other_user(adapter):
    adapter.config.extra["free_response_channels"] = CHANNEL_ID
    adapter.config.extra["ignore_other_user_mentions"] = True

    await _run(adapter, _event(f"<@{OTHER_USER_ID}> this is for you", ts="1700000000.000001"))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_free_response_can_opt_out_of_ignoring_other_user_mentions(adapter):
    adapter.config.extra["free_response_channels"] = CHANNEL_ID
    adapter.config.extra["ignore_other_user_mentions"] = False

    await _run(adapter, _event(f"<@{OTHER_USER_ID}> still ambient chatter", ts="1700000000.000002"))

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_free_response_replies_when_bot_also_mentioned(adapter):
    adapter.config.extra["free_response_channels"] = CHANNEL_ID
    adapter.config.extra["ignore_other_user_mentions"] = True

    await _run(
        adapter,
        _event(f"<@{OTHER_USER_ID}> and <@{BOT_USER_ID}> please compare", ts="1700000000.000003"),
    )

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_free_response_replies_when_bot_mentioned_in_pipe_form(adapter):
    """A pipe-form bot mention (``<@U123|name>``) counts as "also mentioned"
    even though the exact-markup ``is_mentioned`` check misses it."""
    adapter.config.extra["free_response_channels"] = CHANNEL_ID
    adapter.config.extra["ignore_other_user_mentions"] = True

    await _run(
        adapter,
        _event(
            f"<@{OTHER_USER_ID}|rasha> and <@{BOT_USER_ID}|hermes> please compare",
            ts="1700000000.000004",
        ),
    )

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_mentioned_thread_ignores_followup_addressed_to_other_user(adapter):
    """Ben's case: once the bot has been mentioned in a thread it auto-follows,
    but a follow-up addressed to another human should not wake it."""
    thread_ts = "1700000000.000010"
    adapter._mentioned_threads.add(thread_ts)
    adapter.config.extra["ignore_other_user_mentions"] = True

    await _run(
        adapter,
        _event(f"<@{OTHER_USER_ID}> check this out", ts="1700000000.000011", thread_ts=thread_ts),
    )

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mentioned_thread_still_answers_plain_followup(adapter):
    """No over-suppression: a plain follow-up (no leading mention) in a
    mentioned thread is still answered when the option is on."""
    thread_ts = "1700000000.000020"
    adapter._mentioned_threads.add(thread_ts)
    adapter.config.extra["ignore_other_user_mentions"] = True

    await _run(
        adapter,
        _event("thanks, that makes sense", ts="1700000000.000021", thread_ts=thread_ts),
    )

    adapter.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Config bridge: config.yaml slack.ignore_other_user_mentions → extra + env
# ---------------------------------------------------------------------------

def test_config_bridges_slack_ignore_other_user_mentions(monkeypatch, tmp_path):
    """config.yaml ``slack.ignore_other_user_mentions`` reaches the runtime env
    var the adapter reads (same wiring as ``strict_mention`` — via the plugin's
    apply_yaml_config_fn bridge, not the generic shared-key allowlist)."""
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n  ignore_other_user_mentions: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SLACK_IGNORE_OTHER_USER_MENTIONS", raising=False)

    load_gateway_config()

    import os as _os
    assert _os.environ["SLACK_IGNORE_OTHER_USER_MENTIONS"] == "true"


def test_ignore_other_user_mentions_env_wins_over_yaml(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n  ignore_other_user_mentions: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_IGNORE_OTHER_USER_MENTIONS", "false")

    load_gateway_config()

    import os as _os
    assert _os.environ["SLACK_IGNORE_OTHER_USER_MENTIONS"] == "false"
