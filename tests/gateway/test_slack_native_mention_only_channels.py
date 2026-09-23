"""Tests for the Slack ``native_mention_only_channels`` per-channel override.

Channels listed here accept ONLY the platform's native ``<@BOTUID>`` mention —
``mention_patterns`` wake-word regexes no longer count as mentions there.
Wake checks (bot-authored thread, previously mentioned thread, active session)
still apply, so ongoing conversations are not cut off, and wake words keep
working in every channel NOT listed.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


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
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config  # noqa: E402

BOT_USER_ID = "U_BOT"
CHANNEL_ID = "C_NATIVE_ONLY"
OTHER_CHANNEL_ID = "C_ELSEWHERE"
WAKE_PATTERNS = ["^(미쿠야|미쿠)"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    for var in (
        "SLACK_REQUIRE_MENTION",
        "SLACK_REQUIRE_MENTION_CHANNELS",
        "SLACK_NATIVE_MENTION_ONLY_CHANNELS",
        "SLACK_MENTION_PATTERNS",
        "SLACK_FREE_RESPONSE_CHANNELS",
        "SLACK_STRICT_MENTION",
        "SLACK_THREAD_REQUIRE_MENTION",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(
        return_value={
            "user": {
                "is_bot": False,
                "profile": {"display_name": "Test User"},
                "real_name": "Test User",
            }
        }
    )
    a._bot_user_id = BOT_USER_ID
    a._running = True
    a.handle_message = AsyncMock()
    a._fetch_thread_context = AsyncMock(return_value="")
    a._fetch_thread_parent_text = AsyncMock(return_value="")
    a._has_active_session_for_thread = MagicMock(return_value=False)
    return a


def _event(text, ts="100.000", thread_ts=None, channel=CHANNEL_ID):
    event = {
        "type": "message",
        "channel": channel,
        "channel_type": "channel",
        "user": "U_HUMAN",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


# ---------------------------------------------------------------------------
# _slack_native_mention_only_channels() parsing
# ---------------------------------------------------------------------------


def _make(extra=None):
    a = object.__new__(SlackAdapter)
    a.config = PlatformConfig(enabled=True, extra=dict(extra or {}))
    return a


def test_native_mention_only_channels_csv_and_list():
    assert _make(
        {"native_mention_only_channels": "C1, C2"}
    )._slack_native_mention_only_channels() == {"C1", "C2"}
    assert _make(
        {"native_mention_only_channels": ["C1", "C2"]}
    )._slack_native_mention_only_channels() == {"C1", "C2"}


def test_native_mention_only_channels_empty_and_none():
    assert _make()._slack_native_mention_only_channels() == set()
    assert _make({"native_mention_only_channels": ""})._slack_native_mention_only_channels() == set()
    assert (
        _make({"native_mention_only_channels": None})._slack_native_mention_only_channels()
        == set()
    )


def test_native_mention_only_channels_env_var_fallback(monkeypatch):
    monkeypatch.setenv(
        "SLACK_NATIVE_MENTION_ONLY_CHANNELS", f"{CHANNEL_ID},{OTHER_CHANNEL_ID}"
    )
    result = _make()._slack_native_mention_only_channels()  # no config value → env
    assert result == {CHANNEL_ID, OTHER_CHANNEL_ID}


def test_yaml_bridge_sets_env(monkeypatch):
    monkeypatch.delenv("SLACK_NATIVE_MENTION_ONLY_CHANNELS", raising=False)
    _apply_yaml_config({}, {"native_mention_only_channels": ["C1", "C2"]})

    assert os.environ["SLACK_NATIVE_MENTION_ONLY_CHANNELS"] == "C1,C2"
    monkeypatch.delenv("SLACK_NATIVE_MENTION_ONLY_CHANNELS", raising=False)


def test_yaml_bridge_does_not_overwrite_preset_env(monkeypatch):
    monkeypatch.setenv("SLACK_NATIVE_MENTION_ONLY_CHANNELS", "C_ENV")
    _apply_yaml_config({}, {"native_mention_only_channels": ["C1", "C2"]})

    assert os.environ["SLACK_NATIVE_MENTION_ONLY_CHANNELS"] == "C_ENV"


# ---------------------------------------------------------------------------
# Routing behaviour
# ---------------------------------------------------------------------------


def _configure_native_only(adapter):
    adapter.config.extra["mention_patterns"] = list(WAKE_PATTERNS)
    adapter.config.extra["native_mention_only_channels"] = CHANNEL_ID


@pytest.mark.asyncio
async def test_wake_word_ignored_in_native_only_channel(adapter):
    """A wake-word-only message in a listed channel no longer counts as a
    mention, so the default require_mention gate drops it."""
    _configure_native_only(adapter)

    await adapter._handle_slack_message(_event("미쿠야 안녕"))

    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_native_mention_still_processed_in_native_only_channel(adapter):
    """A literal <@bot> mention keeps working in a listed channel."""
    _configure_native_only(adapter)

    await adapter._handle_slack_message(_event(f"<@{BOT_USER_ID}> hi"))

    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_wake_word_still_processed_in_other_channel(adapter):
    """Wake-word patterns stay live in channels NOT listed."""
    _configure_native_only(adapter)

    await adapter._handle_slack_message(
        _event("미쿠야 안녕", channel=OTHER_CHANNEL_ID)
    )

    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_wake_word_does_not_register_thread_in_native_only_channel(adapter):
    """A wake-word message in a listed channel must NOT persist into
    _mentioned_threads — otherwise the next unmentioned reply in that thread
    would auto-trigger the bot and defeat the feature (mirrors the strict-mode
    guard in test_slack_mention.py::test_mention_in_strict_mode_does_not_register_thread)."""
    _configure_native_only(adapter)

    await adapter._handle_slack_message(_event("미쿠야 안녕", ts="100.000"))

    adapter.handle_message.assert_not_called()
    assert adapter._mentioned_threads == set()


@pytest.mark.asyncio
async def test_native_only_channel_wake_checks_still_apply(adapter):
    """A previously mentioned thread still auto-follows in a listed channel."""
    _configure_native_only(adapter)
    adapter._mentioned_threads.add("100.000")

    await adapter._handle_slack_message(
        _event("follow-up", ts="101.000", thread_ts="100.000")
    )

    adapter.handle_message.assert_called_once()


def test_extra_config_wins_over_env(monkeypatch):
    """Config precedence contract: extra is consulted first, env only when
    the extra key is absent."""
    monkeypatch.setenv("SLACK_NATIVE_MENTION_ONLY_CHANNELS", "C_ENV")
    assert _make(
        {"native_mention_only_channels": "C_CONFIG"}
    )._slack_native_mention_only_channels() == {"C_CONFIG"}


@pytest.mark.asyncio
async def test_strict_mention_wake_word_still_dropped_in_native_only_channel(adapter):
    """strict_mention gates on the same is_mentioned computation — a wake word
    in a listed channel must not satisfy strict mode either, while a native
    mention still does."""
    _configure_native_only(adapter)
    adapter.config.extra["strict_mention"] = True

    await adapter._handle_slack_message(_event("미쿠야 안녕"))
    adapter.handle_message.assert_not_called()

    await adapter._handle_slack_message(_event(f"<@{BOT_USER_ID}> hi", ts="102.000"))
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_pairs_with_require_mention_channels_when_require_mention_off(adapter):
    """With require_mention globally off the free-response path never consults
    is_mentioned, so a native-only channel must ALSO be listed in
    require_mention_channels (the pairing the helper docstring advertises):
    wake words drop, native mentions pass, unlisted channels stay
    free-response."""
    _configure_native_only(adapter)
    adapter.config.extra["require_mention"] = False
    adapter.config.extra["require_mention_channels"] = CHANNEL_ID

    await adapter._handle_slack_message(_event("미쿠야 안녕"))
    adapter.handle_message.assert_not_called()

    await adapter._handle_slack_message(_event(f"<@{BOT_USER_ID}> hi", ts="102.000"))
    adapter.handle_message.assert_called_once()

    await adapter._handle_slack_message(
        _event("just chatting", ts="103.000", channel=OTHER_CHANNEL_ID)
    )
    assert adapter.handle_message.call_count == 2
