"""Regression tests for #62034 — pending multi-choice clarify prompts must not
swallow unrelated thread follow-up messages.

When a NATIVE interactive multi-choice clarify (buttons rendered,
``awaiting_text=False``) is pending, the gateway text-intercept used to
consume ANY non-command message in the session as the clarify answer —
arbitrary prose vanished into clarify resolution and the agent appeared to
ignore the user's thread messages.

After the fix (``tools/clarify_gateway._coerce_text_response`` rejects
arbitrary prose for native multi-choice prompts):

  * numeric selections ("2") and exact choice labels still resolve, and
  * arbitrary prose falls through the intercept and continues as a normal
    message-handling turn.

Open-ended clarifies, explicit "Other" text-capture mode, and the base
adapter's numbered-text fallback (which flips ``awaiting_text`` at send time)
keep accepting free text.
"""

from unittest.mock import patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource


SESSION_KEY = "agent:main:slack:dm:D123:1111.2222"


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.SLACK)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "im"}


class _FellThroughIntercept(Exception):
    """Sentinel: _handle_message got PAST the clarify text-intercept."""


def _event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="D123",
            chat_type="dm",
            user_id="U1",
            thread_id="1111.2222",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _make_runner(adapter):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: SESSION_KEY
    runner._adapter_for_source = lambda source: adapter
    runner._update_prompt_pending = {}
    return runner


async def _dispatch(runner, event):
    """Run _handle_message with a tripwire installed AFTER the clarify
    intercept (the slash-confirm pending lookup is the next statement), so a
    raised ``_FellThroughIntercept`` proves the message was NOT swallowed."""
    import tools.slash_confirm as slash_confirm_mod

    def _tripwire(_key):
        raise _FellThroughIntercept()

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]), \
            patch.object(slash_confirm_mod, "get_pending", _tripwire):
        return await runner._handle_message(event)


@pytest.mark.asyncio
async def test_thread_prose_not_swallowed_by_native_multi_choice_clarify():
    """Arbitrary prose during a pending button-clarify continues as a normal turn."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _StubAdapter()
    runner = _make_runner(adapter)
    # Native interactive multi-choice prompt: awaiting_text stays False.
    entry = cm.register("cl-native", SESSION_KEY, "Pick a UI variant", ["buttons", "dropdown"])
    assert entry.awaiting_text is False

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(runner, _event("just checking the visual UI, no need to pass any data"))

    # The clarify entry must still be pending and unresolved.
    with cm._lock:
        entry = cm._entries.get("cl-native")
    assert entry is not None
    assert not entry.event.is_set()
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_numeric_reply_still_resolves_native_multi_choice_clarify():
    """Typed "2" keeps resolving the button prompt through the same intercept."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _StubAdapter()
    runner = _make_runner(adapter)
    cm.register("cl-num", SESSION_KEY, "Pick a UI variant", ["buttons", "dropdown"])

    result = await _dispatch(runner, _event("2"))

    assert result == ""  # intercepted + acknowledged silently
    with cm._lock:
        entry = cm._entries.get("cl-num")
    assert entry is not None
    assert entry.event.is_set()
    assert entry.response == "dropdown"
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_exact_label_reply_still_resolves_native_multi_choice_clarify():
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _StubAdapter()
    runner = _make_runner(adapter)
    cm.register("cl-label", SESSION_KEY, "Pick a UI variant", ["buttons", "dropdown"])

    result = await _dispatch(runner, _event("Buttons"))

    assert result == ""
    with cm._lock:
        entry = cm._entries.get("cl-label")
    assert entry is not None
    assert entry.event.is_set()
    assert entry.response == "buttons"
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_prose_still_accepted_after_other_flips_text_capture():
    """After the user taps 'Other', free text IS the answer — must resolve."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _StubAdapter()
    runner = _make_runner(adapter)
    cm.register("cl-other", SESSION_KEY, "Pick a UI variant", ["buttons", "dropdown"])
    assert cm.mark_awaiting_text("cl-other") is True

    result = await _dispatch(runner, _event("a carousel actually"))

    assert result == ""
    with cm._lock:
        entry = cm._entries.get("cl-other")
    assert entry is not None
    assert entry.event.is_set()
    assert entry.response == "a carousel actually"
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_prose_still_accepted_for_open_ended_clarify():
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _StubAdapter()
    runner = _make_runner(adapter)
    cm.register("cl-open", SESSION_KEY, "What should I name it?", None)

    result = await _dispatch(runner, _event("call it hermes-ux"))

    assert result == ""
    with cm._lock:
        entry = cm._entries.get("cl-open")
    assert entry is not None
    assert entry.event.is_set()
    assert entry.response == "call it hermes-ux"
    _clear_clarify_state()
