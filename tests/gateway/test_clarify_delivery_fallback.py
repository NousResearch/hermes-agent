"""A clarify card that cannot render is retried as plain text, and a card that fails late does
not pass for user inactivity (#112684).

Real ``TurnRunner._clarify_callback_sync`` + real ``tools.clarify_gateway`` + a real gateway loop
thread; the adapter is Telegram-shaped (``SendResult`` shapes from
``plugins/platforms/telegram/adapter.py::_send_prompt``) with a native ``send_clarify`` override.
"""

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from tools import clarify_gateway as cm

UNDELIVERED = "[clarify prompt could not be delivered"  # prefix shared by every delivery notice


class _CardAdapter(BasePlatformAdapter):
    """Native card override; ``card`` is the coroutine function the card send runs."""

    def __init__(self, card):
        self.card = card
        self.sent_text: list[str] = []
        self.cards = 0

    def pause_typing_for_chat(self, chat_id):
        return None

    def resume_typing_for_chat(self, chat_id):
        return None

    async def send_clarify(self, **kwargs):
        self.cards += 1
        return await self.card()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent_text.append(content)
        return SendResult(success=True, message_id="t1")

    async def retire_clarify_card(self, clarify_id, notice):
        return None

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def _runner(adapter, loop, monkeypatch, timeout=5):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="42", _status_thread_metadata=None,
        session_key="sk-fallback", stream_consumer_holder=[None], _loop_for_step=loop)
    runner._close_native_stream_boundary = lambda *a, **k: None
    monkeypatch.setattr(cm, "get_clarify_timeout", lambda: timeout)
    return runner


def _answer_once_text_prompt_is_seen(adapter, text):
    """Answer ONLY after the plain-text prompt was observed — never on a deadline, so a missing
    fallback leaves the waiter blocked and the test fails on elapsed time / sent_text, not luck."""
    def _wait_then_answer():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not adapter.sent_text:
            time.sleep(0.02)
        if not adapter.sent_text:
            return
        time.sleep(0.1)
        cm.resolve_text_response_for_session("sk-fallback", text)
    threading.Thread(target=_wait_then_answer, daemon=True).start()


# --- Atom: the platform rejects the native card -> plain-text question instead of a sentinel ---


def test_rejected_card_is_reasked_as_plain_text_and_the_typed_answer_counts(loop, monkeypatch):
    async def rejected():
        return SendResult(success=False, error="Bad Request: BUTTON_TYPE_INVALID")

    adapter = _CardAdapter(rejected)
    _answer_once_text_prompt_is_seen(adapter, "2")
    response = _runner(adapter, loop, monkeypatch)._clarify_callback_sync("Pick?", ["alpha", "beta"])
    assert response == "beta"  # the numbered text prompt maps "2" back to the choice
    assert adapter.cards == 1
    assert len(adapter.sent_text) == 1 and "1. alpha" in adapter.sent_text[0]


def test_declined_card_is_never_reasked_as_text(loop, monkeypatch):
    """A connector egress DECLINE refused the destination: re-sending the question as text into
    that chat is the exfiltration the guard exists to stop."""
    async def declined():
        return SendResult(success=False, error="egress declined: destination not allowed")

    adapter = _CardAdapter(declined)
    response = _runner(adapter, loop, monkeypatch)._clarify_callback_sync("Pick?", ["alpha", "beta"])
    assert response.startswith(UNDELIVERED)
    assert adapter.sent_text == []


# --- Atom: the card send outruns the ack window, then fails -------------------------------


def test_card_failing_after_the_ack_window_falls_back_to_text_instead_of_waiting(loop, monkeypatch):
    from gateway import run_turn_runner_clarify_delivery as delivery

    monkeypatch.setattr(delivery, "SEND_ACK_WINDOW", 0.2)

    async def late_failure():
        await asyncio.sleep(0.6)
        return SendResult(success=False, error="Timed out: pool timeout")

    adapter = _CardAdapter(late_failure)
    _answer_once_text_prompt_is_seen(adapter, "beta")
    started = time.monotonic()
    response = _runner(adapter, loop, monkeypatch, timeout=30)._clarify_callback_sync("Pick?", ["alpha", "beta"])
    elapsed = time.monotonic() - started
    assert response == "beta"
    assert len(adapter.sent_text) == 1  # the plain-text prompt was actually sent, once
    assert elapsed < 0.2 + 2  # released right after the ack window + late failure, never clarify_timeout


def test_card_resolving_ambiguous_after_the_ack_window_stays_armed_for_a_button_tap(loop, monkeypatch):
    """A relay lost-ack (``raw_response.ambiguous``) after the window means the card MAY have posted:
    the registration must stay armed so the user's later button tap still answers — the same
    invariant ``_abort_for_outcome`` keeps for an immediate ambiguous outcome. Before: the late watch
    treated it like a definitive failure, released the wait with the delivery notice and the tap was
    lost (no pending entry)."""
    from gateway import run_turn_runner_clarify_delivery as delivery

    monkeypatch.setattr(delivery, "SEND_ACK_WINDOW", 0.2)

    async def late_ambiguous():
        await asyncio.sleep(0.6)
        return SendResult(success=False, raw_response={"ambiguous": True})

    adapter = _CardAdapter(late_ambiguous)
    pending_at_tap = []

    def _tap_button():
        time.sleep(1.5)
        entry = cm.get_pending_for_session("sk-fallback", include_choice_prompts=True)
        pending_at_tap.append(entry)
        if entry is not None:
            cm.resolve_gateway_clarify(entry.clarify_id, "beta")
    tap = threading.Thread(target=_tap_button, daemon=True)
    tap.start()

    response, answered = _runner(adapter, loop, monkeypatch, timeout=30)._ask_clarify_question(
        "Pick?", ["alpha", "beta"], False)
    tap.join(timeout=5)  # never let a late tap leak into the next test's registration
    assert pending_at_tap and pending_at_tap[0] is not None  # still armed when the tap arrived
    assert (response, answered) == ("beta", True)
    assert adapter.sent_text == []  # possibly-delivered: never re-sent as text


def test_card_and_text_both_failing_late_release_the_wait_with_the_delivery_notice(loop, monkeypatch):
    """Before: a card whose send failed after the window left the agent blocked for the whole
    clarify_timeout and then reported ``[user did not respond within Nm]`` for a question the user
    never saw."""
    from gateway import run_turn_runner_clarify_delivery as delivery

    monkeypatch.setattr(delivery, "SEND_ACK_WINDOW", 0.2)

    async def late_failure():
        await asyncio.sleep(0.6)
        return SendResult(success=False, error="Timed out: pool timeout")

    class _NoTextEither(_CardAdapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent_text.append(content)
            return SendResult(success=False, error="Timed out")

    adapter = _NoTextEither(late_failure)
    started = time.monotonic()
    response, answered = _runner(adapter, loop, monkeypatch, timeout=30)._ask_clarify_question(
        "Pick?", ["alpha", "beta"], False)
    assert (response, answered) == (UNDELIVERED + "]", False)
    assert time.monotonic() - started < 10
    assert len(adapter.sent_text) == 1  # the text fallback was tried exactly once


def test_card_declined_after_the_ack_window_releases_with_the_declined_notice(loop, monkeypatch):
    """A late connector DECLINE is as definitive as an immediate one: no text retry, and the
    notice names the refusal (``UNDELIVERED_DECLINED``) instead of the generic delivery failure."""
    from gateway import run_turn_runner_clarify_delivery as delivery

    monkeypatch.setattr(delivery, "SEND_ACK_WINDOW", 0.2)

    async def late_decline():
        await asyncio.sleep(0.6)
        return SendResult(success=False, error="egress declined: destination not allowed")

    adapter = _CardAdapter(late_decline)
    response, answered = _runner(adapter, loop, monkeypatch, timeout=30)._ask_clarify_question(
        "Pick?", ["alpha", "beta"], False)
    assert (response, answered) == (delivery.UNDELIVERED_DECLINED, False)
    assert adapter.sent_text == []


# --- Atom: no chat surface at all -----------------------------------------------------------


def test_no_status_adapter_reports_the_missing_surface_not_inactivity(loop, monkeypatch):
    runner = _runner(None, loop, monkeypatch)
    payload = json.loads(runner._clarify_callback_sync(
        "", None, questions=[{"qid": "q0", "question": "One?", "choices": ["a"]}]))
    assert payload["timed_out"] is True
    assert payload["notice"].startswith(UNDELIVERED)
    assert runner._clarify_callback_sync("One?", ["a"]).startswith(UNDELIVERED)
