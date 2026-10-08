"""Slack ``natural_thread_channels`` and strict-mention clarify replies.

Under ``strict_mention`` / ``thread_require_mention`` every message needs an @mention.
Two exceptions route un-mentioned thread replies to the bot:

* channels in ``natural_thread_channels`` once the thread was engaged by a mention;
* a typed answer from the user whose thread session has a pending clarify prompt.
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
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config  # noqa: E402

BOT = "U_BOT"
NATURAL = "C_NATURAL"
STRICT = "C_STRICT"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    for var in (
        "SLACK_REQUIRE_MENTION", "SLACK_STRICT_MENTION", "SLACK_THREAD_REQUIRE_MENTION",
        "SLACK_NATURAL_THREAD_CHANNELS", "SLACK_FREE_RESPONSE_CHANNELS",
        "SLACK_REQUIRE_MENTION_CHANNELS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-test", extra={
        "require_mention": True, "strict_mention": True, "thread_require_mention": True,
        "natural_thread_channels": [NATURAL],
    })
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(return_value={
        "user": {"is_bot": False, "profile": {"display_name": "Tester"}, "real_name": "Tester"}})
    a._bot_user_id = BOT
    a._running = True
    a.handle_message = AsyncMock()
    a._fetch_thread_context = AsyncMock(return_value="")
    a._fetch_thread_parent_text = AsyncMock(return_value="")
    a._has_active_session_for_thread = MagicMock(return_value=False)
    a._thread_has_pending_clarify = MagicMock(return_value=False)
    return a


def _event(text, ts, thread_ts=None, channel=NATURAL, user="U_HUMAN"):
    event = {"type": "message", "channel": channel, "channel_type": "channel",
             "user": user, "text": text, "ts": ts}
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


@pytest.mark.asyncio
async def test_natural_channel_follows_engaged_thread_without_mention(adapter):
    await adapter._handle_slack_message(_event(f"<@{BOT}> check job 1234", ts="100.000"))
    await adapter._handle_slack_message(_event("also the visit date", ts="101.000", thread_ts="100.000"))
    assert adapter.handle_message.call_count == 2


@pytest.mark.asyncio
async def test_natural_channel_wakes_after_restart_from_mentioned_parent(adapter):
    adapter._fetch_thread_parent_text = AsyncMock(return_value=f"<@{BOT}> check job 1234")
    await adapter._handle_slack_message(_event("follow-up", ts="201.000", thread_ts="200.000"))
    await adapter._handle_slack_message(_event("second", ts="202.000", thread_ts="200.000"))
    assert adapter.handle_message.call_count == 2
    assert adapter._fetch_thread_parent_text.await_count == 1  # remembered after first wake


@pytest.mark.asyncio
async def test_natural_channel_unengaged_thread_stays_silent(adapter):
    await adapter._handle_slack_message(_event("side chat", ts="301.000", thread_ts="300.000"))
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_natural_channel_top_level_still_needs_mention(adapter):
    await adapter._handle_slack_message(_event("hello team", ts="400.000"))
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_strict_channel_engaged_thread_still_needs_mention(adapter):
    await adapter._handle_slack_message(_event(f"<@{BOT}> start", ts="500.000", channel=STRICT))
    await adapter._handle_slack_message(_event("follow", ts="501.000", thread_ts="500.000", channel=STRICT))
    assert adapter.handle_message.call_count == 1


@pytest.mark.asyncio
async def test_pending_clarify_reply_accepted_without_mention_in_strict_channel(adapter):
    adapter._thread_has_pending_clarify = MagicMock(return_value=True)
    await adapter._handle_slack_message(_event("2", ts="601.000", thread_ts="600.000", channel=STRICT))
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_pending_clarify_does_not_bypass_authorization(adapter):
    adapter._thread_has_pending_clarify = MagicMock(return_value=True)
    adapter._early_reject_unauthorized = MagicMock(return_value=True)
    await adapter._handle_slack_message(_event("2", ts="701.000", thread_ts="700.000", channel=STRICT))
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_pending_clarify_does_not_bypass_allowed_channels(adapter):
    adapter.config.extra["allowed_channels"] = [NATURAL]
    adapter._thread_has_pending_clarify = MagicMock(return_value=True)
    await adapter._handle_slack_message(_event("2", ts="801.000", thread_ts="800.000", channel=STRICT))
    adapter.handle_message.assert_not_called()


def _clarify_adapter(origin_user):
    from types import SimpleNamespace
    a = object.__new__(SlackAdapter)
    a._build_thread_session_key = MagicMock(return_value="sk-thread")
    store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={
        "sk-thread": SimpleNamespace(origin=SimpleNamespace(user_id=origin_user))})
    a._session_store = store
    return a


def _reset_clarify():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()


def test_pending_clarify_accepts_only_requester_with_valid_answer():
    from tools import clarify_gateway as cm
    _reset_clarify()
    a = _clarify_adapter("U1")
    assert a._thread_has_pending_clarify("C1", "1.0", "U1", text="2") is False  # nothing pending
    cm.register("cid-p", "sk-thread", "Which job?", ["Job 1", "Job 2"])
    try:
        assert a._thread_has_pending_clarify("C1", "1.0", "U1", text="2") is True
        assert a._thread_has_pending_clarify("C1", "1.0", "U1", text="job 2") is True
        assert a._thread_has_pending_clarify("C1", "1.0", "U1", text="lol nice") is False
        assert a._thread_has_pending_clarify("C1", "1.0", "U2", text="2") is False  # bystander
        assert a._thread_has_pending_clarify("C1", "1.0", "U1", text="9") is False
    finally:
        _reset_clarify()


def test_open_ended_clarify_accepts_any_text_from_requester_only():
    from tools import clarify_gateway as cm
    _reset_clarify()
    a = _clarify_adapter("U1")
    cm.register("cid-o", "sk-thread", "What address?", None)
    try:
        assert a._thread_has_pending_clarify("C1", "1.0", "U1", text="123 Main St") is True
        assert a._thread_has_pending_clarify("C1", "1.0", "U2", text="123 Main St") is False
    finally:
        _reset_clarify()


def test_unknown_session_origin_fails_closed():
    from tools import clarify_gateway as cm
    _reset_clarify()
    a = _clarify_adapter("")
    cm.register("cid-u", "sk-thread", "Which?", ["a", "b"])
    try:
        assert a._thread_has_pending_clarify("C1", "1.0", "U1", text="1") is False
    finally:
        _reset_clarify()


def test_natural_thread_channels_yaml_bridge(monkeypatch):
    extra = _apply_yaml_config({}, {"natural_thread_channels": ["C1", "C2"]})
    assert extra["natural_thread_channels"] == ["C1", "C2"]
    assert os.environ["SLACK_NATURAL_THREAD_CHANNELS"] == "C1,C2"
    a = object.__new__(SlackAdapter)
    a.config = PlatformConfig(enabled=True, extra={"natural_thread_channels": "C1, C2"})
    assert a._slack_natural_thread_channels() == {"C1", "C2"}
    monkeypatch.delenv("SLACK_NATURAL_THREAD_CHANNELS", raising=False)
