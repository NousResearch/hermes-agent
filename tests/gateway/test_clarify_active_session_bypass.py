"""Regression tests for clarify replies while a gateway session is busy."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text="custom answer"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.mark.asyncio
async def test_active_session_routes_typed_choice_clarify_reply_to_runner_not_busy_queue():
    """Typed text must resolve a pending choice clarify even while the agent is busy.

    Telegram button clarifies keep the adapter session active while the agent
    thread blocks on ``wait_for_response``.  If the adapter only bypasses for
    entries already marked ``awaiting_text``, typed replies to the visible
    multi-choice prompt are handled as busy follow-ups and the clarify wait is
    never resolved.
    """
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_active_session_bypass_uses_profile_namespaced_key_under_multiplex():
    """Regression for issue #82975: under a named-profile multiplex, the
    adapter's clarify bypass lookup must use the SAME profile-namespaced
    session key that the runner registers pending clarifies under
    (SessionStore._generate_session_key() includes
    profile=self._resolve_profile_for_key(source)), not the legacy
    unnamespaced key. Otherwise the lookup misses, and a user's answer to
    a pending clarify is routed to the busy-session queue instead of
    resolving it -- the turn then hangs until the clarify's 3600s timeout."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")

    # A session_store configured for profile multiplexing, matching what
    # the runner's SessionStore._generate_session_key() actually produces.
    session_store = MagicMock()
    session_store._resolve_profile_for_key.return_value = "ops"
    adapter._session_store = session_store

    profile_namespaced_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        profile="ops",
    )
    # Sanity: the profile-namespaced key really is different from the
    # legacy unnamespaced one -- otherwise this test wouldn't distinguish
    # the fixed behavior from the bug.
    legacy_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    assert profile_namespaced_key != legacy_key

    adapter._active_sessions[profile_namespaced_key] = asyncio.Event()
    # The runner registers the pending clarify under its own
    # profile-namespaced key, exactly as it would in a real multiplexed
    # deployment.
    cm.register("clarify-1", profile_namespaced_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_gateway_batch_text_reply_skips_button_resolved_card_and_answers_next_question():
    """The real gateway intercept must target the next unresolved batch card."""
    _clear_clarify_state()
    from gateway.run import GatewayRunner
    from tools import clarify_gateway as cm

    cm.register("q1", "telegram:batch", "First?", ["one", "two"])
    second = cm.register("q2", "telegram:batch", "Second?", ["red", "blue"])
    assert cm.resolve_gateway_clarify("q1", "one") is True

    runner = object.__new__(GatewayRunner)
    runner._pending_event_audio_paths = lambda event: []
    runner._adapter_for_source = lambda source: None

    async def prepare_reply(event):
        return "2"

    runner._prepare_clarify_reply_text = prepare_reply
    event = _event("2")
    result = await runner._hm_clarify_reply(event, event.source, "telegram:batch")

    assert result == ""
    assert second.event.is_set()
    assert second.response == "blue"


@pytest.mark.asyncio
async def test_late_reply_to_another_card_does_not_cancel_current_batch():
    """A direct reply to batch A must not release batch B just because its id differs."""
    _clear_clarify_state()
    from gateway.run import GatewayRunner
    from tools import clarify_gateway as cm

    screenshot = cm.register(
        "screenshot", "telegram:batch", "Send screenshot?", None,
        require_text_reply_binding=True,
    )
    assert cm.bind_text_reply_to("screenshot", "prompt-42") is True

    runner = object.__new__(GatewayRunner)
    runner._pending_event_audio_paths = lambda event: []
    runner._adapter_for_source = lambda source: None

    event = _event("und?")
    event.reply_to_message_id = "some-other-message"
    result = await runner._hm_clarify_reply(event, event.source, "telegram:batch")

    assert result is None
    assert not screenshot.event.is_set()
    assert screenshot.response is None


@pytest.mark.asyncio
async def test_unbound_batch_followup_releases_wait_and_visibly_invalidates_cards():
    """Bare prose releases the batch and calls the adapter's stale-card hook."""
    _clear_clarify_state()
    from gateway.run import GatewayRunner
    from tools import clarify_gateway as cm

    screenshot = cm.register("screenshot", "telegram:batch", "Send screenshot?", None,
                             require_text_reply_binding=True)
    assert cm.bind_text_reply_to("screenshot", "prompt-42") is True
    adapter = MagicMock()
    adapter.invalidate_clarify_batch_for_session = AsyncMock()
    runner = object.__new__(GatewayRunner)
    runner._pending_event_audio_paths = lambda event: []
    runner._adapter_for_source = lambda source: adapter
    event = _event("new unrelated follow-up")

    assert await runner._hm_clarify_reply(event, event.source, "telegram:batch") is None
    assert screenshot.event.is_set() and screenshot.response == ""
    adapter.invalidate_clarify_batch_for_session.assert_awaited_once_with("telegram:batch")


@pytest.mark.asyncio
async def test_missing_delivery_id_and_cross_user_reply_cannot_change_batch_state():
    """No id and a reply from another user both fail closed without cancellation."""
    _clear_clarify_state()
    from gateway.run import GatewayRunner
    from tools import clarify_gateway as cm

    card = cm.register("card", "telegram:batch", "Text?", None, require_text_reply_binding=True)
    runner = object.__new__(GatewayRunner)
    runner._pending_event_audio_paths = lambda event: []
    runner._adapter_for_source = lambda source: None
    missing_id = _event("answer")
    assert await runner._hm_clarify_reply(missing_id, missing_id.source, "telegram:batch") is None
    assert not card.event.is_set()

    assert cm.bind_text_reply_to("card", "prompt-42", chat_id="12345", user_id="user1")
    other_user = _event("answer")
    other_user.source.user_id = "user2"
    other_user.reply_to_message_id = "prompt-42"
    assert await runner._hm_clarify_reply(other_user, other_user.source, "telegram:batch") is None
    assert not card.event.is_set()


@pytest.mark.asyncio
async def test_bound_reply_to_open_batch_card_is_consumed_as_its_answer():
    """The explicit Telegram reply anchor still permits the intended screenshot answer."""
    _clear_clarify_state()
    from gateway.run import GatewayRunner
    from tools import clarify_gateway as cm

    screenshot = cm.register(
        "screenshot", "telegram:batch", "Send screenshot?", None,
        require_text_reply_binding=True,
    )
    assert cm.bind_text_reply_to("screenshot", "prompt-42") is True

    runner = object.__new__(GatewayRunner)
    runner._pending_event_audio_paths = lambda event: []
    runner._adapter_for_source = lambda source: None

    event = _event("here is the screenshot")
    event.reply_to_message_id = "prompt-42"
    result = await runner._hm_clarify_reply(event, event.source, "telegram:batch")

    assert result == ""
    assert screenshot.event.is_set()
    assert screenshot.response == "here is the screenshot"


