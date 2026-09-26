"""A clarify prompt reaches the messaging chat that ORIGINATED a session driven elsewhere.

A Desktop-driven turn inside a Telegram topic (or a Discord thread) renders the clarify card on
the desktop, so the topic showed the operator's prompt and the final reply with the question and
the decision that shaped it missing entirely (#103209) — on a phone the thread reads as an answer
to a question that was never asked. The relay posts both to that originating chat, and only when
it is a different chat from the one being asked.

Real ``TurnRunner._clarify_callback_sync`` + real ``tools.clarify_gateway`` + a real gateway loop
thread. The status adapter plays the Desktop renderer; the origin adapter is a recording double,
so what the originating surface received (text, chat id, thread metadata) is directly observable.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from tools import clarify_gateway as cm

SESSION_KEY = "telegram:topic-77"


class _RecordingAdapter(BasePlatformAdapter):
    """``send`` records every relayed message; ``send_clarify`` counts native cards."""

    def __init__(self, fail_send: bool = False):
        self.sent: list = []
        self.cards = 0
        self.fail_send = fail_send

    # Card path: what the surface being ASKED does with the prompt.
    async def send_clarify(self, **kwargs):
        self.cards += 1
        return SendResult(success=True, message_id="card1")

    async def retire_clarify_card(self, clarify_id, notice):
        return None

    # Relay path: what the ORIGINATING surface receives.
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if self.fail_send:
            raise RuntimeError("origin surface is unreachable")
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="m1")

    def pause_typing_for_chat(self, chat_id):
        return None

    def resume_typing_for_chat(self, chat_id):
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


def _origin_source(chat_id: str = "topic-77"):
    """The session's persisted origin: the Telegram topic that created it."""
    return SimpleNamespace(platform="telegram", chat_id=chat_id, thread_id=chat_id)


def _runner(status_adapter, origin_adapter, loop, monkeypatch, *, origin=None, session_key=SESSION_KEY):
    from gateway.run_turn_runner import TurnRunner

    origin = origin if origin is not None else _origin_source()
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=status_adapter, _status_chat_id="desk-1", _status_thread_metadata=None,
        session_key=session_key, stream_consumer_holder=[None], _loop_for_step=loop)
    runner._close_native_stream_boundary = lambda *a, **k: None
    runner._runner = SimpleNamespace(
        session_store=SimpleNamespace(
            _ensure_loaded=lambda: None,
            _entries={session_key: SimpleNamespace(origin=origin, transport_profile=None)}),
        _restored_source=lambda entry: entry.origin,
        _get_cached_session_source=lambda key: None,
        _delivery_adapter_for=lambda src: origin_adapter if src is origin else None,
        _run_agent_progress_threading=lambda src, mid, cards: (None, None, {"thread_id": "77"}),
    )
    monkeypatch.setattr(cm, "get_clarify_timeout", lambda: 5)
    return runner


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.02)
    return predicate()


def _answer_once(predicate, text: str) -> None:
    """Answer ONLY after ``predicate`` held — never on a deadline, so a missing relay leaves the
    waiter blocked and the test fails on its assertions rather than on timing luck."""
    def _wait_then_answer():
        if _wait_until(predicate):
            time.sleep(0.05)
            cm.resolve_text_response_for_session(SESSION_KEY, text)

    threading.Thread(target=_wait_then_answer, daemon=True).start()


def test_the_originating_chat_sees_the_question_and_the_answer(loop, monkeypatch):
    status, origin = _RecordingAdapter(), _RecordingAdapter()
    runner = _runner(status, origin, loop, monkeypatch)
    # Waits for the relayed question itself: on a run without the relay nothing answers and the
    # clarify ends in its timeout sentinel.
    _answer_once(lambda: status.cards and origin.sent, "beta")

    assert runner._clarify_callback_sync("Pick?", ["alpha", "beta"]) == "beta"
    assert status.cards == 1
    assert _wait_until(lambda: len(origin.sent) >= 2), f"relay incomplete: {origin.sent}"

    chat_id, question, metadata = origin.sent[0]
    assert chat_id == "topic-77"  # the origin chat, not the desktop's
    assert (metadata or {}).get("thread_id") == "77"
    # The numbered choices match what the card offers, so the topic shows the real question.
    assert "❓ Pick?" in question and "1. alpha" in question and "2. beta" in question
    assert origin.sent[1][1].startswith("✅ answered: beta")


def test_a_session_native_to_its_chat_relays_nothing(loop, monkeypatch):
    """The surface already showing the card IS the origin: a second copy would double every prompt."""
    status = _RecordingAdapter()
    runner = _runner(status, status, loop, monkeypatch)
    _answer_once(lambda: status.cards, "beta")

    assert runner._clarify_callback_sync("Pick?", ["alpha", "beta"]) == "beta"
    assert status.cards == 1
    assert status.sent == []


def test_a_dead_origin_surface_never_breaks_the_card(loop, monkeypatch):
    """The relay is a side effect: an unreachable origin must leave the operator's card intact."""
    status, origin = _RecordingAdapter(), _RecordingAdapter(fail_send=True)
    runner = _runner(status, origin, loop, monkeypatch)
    _answer_once(lambda: status.cards, "beta")

    assert runner._clarify_callback_sync("Pick?", ["alpha", "beta"]) == "beta"
    assert status.cards == 1  # the card path ran to completion regardless


def test_a_timed_out_clarify_relays_no_answer(loop, monkeypatch):
    """No choice was made: relaying the timeout sentinel would read as a decision."""
    status, origin = _RecordingAdapter(), _RecordingAdapter()
    runner = _runner(status, origin, loop, monkeypatch)

    response = runner._clarify_callback_sync("Pick?", ["alpha", "beta"])
    assert response.startswith("[user did not respond")
    assert _wait_until(lambda: len(origin.sent) >= 1)
    assert origin.sent[0][1].startswith("❓ Pick?")  # the question still reached the topic
    assert [text for _chat, text, _meta in origin.sent if "answered" in text] == []
