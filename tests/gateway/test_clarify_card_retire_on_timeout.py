"""A clarify that ends without a click retires the adapter's native card (#110821, #111019).

Drives the real ``TurnRunner._clarify_callback_sync`` against a duck-typed adapter with a
persistent card: on timeout the gateway schedules ``retire_clarify_card`` with the expired
notice; an adapter without that method (text-prompt platforms) gets nothing scheduled.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.platforms.base import SendResult


class _CardAdapter:
    def __init__(self):
        self.retired: list[tuple[str, str]] = []
        self.asked: list[str] = []
        self.resumed: int = 0

    def pause_typing_for_chat(self, chat_id):
        return None

    def resume_typing_for_chat(self, chat_id):
        self.resumed += 1

    async def send_clarify(self, **kwargs):
        self.asked.append(kwargs["question"])
        return SendResult(success=True, message_id="1.2")

    async def retire_clarify_card(self, clarify_id, notice):
        self.retired.append((clarify_id, notice))


class _TextAdapter(_CardAdapter):
    retire_clarify_card = None  # type: ignore[assignment]


_ONE_QUESTION = [{"qid": "q0", "question": "Pick one", "choices": ["a", "b"], "multi_select": False}]


def _asked(adapter):
    return [text.split("\n")[0] for text in adapter.asked]


def _run_clarify(adapter, questions=_ONE_QUESTION, answers=(), via_tool=False):
    """Returns (clarify result, labels of every coroutine the runner scheduled).

    Question i resolves from ``answers[i]``, and ``None`` leaves it unanswered so it expires.
    ``via_tool`` routes the questions through ``clarify_tool`` itself rather than calling the
    runner's callback directly."""
    from gateway.run_turn_runner import TurnRunner
    from tools import clarify_gateway as cm

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata={},
        session_key="sk1", stream_consumer_holder=[None])
    labels: list[str] = []

    class _Fut:
        def __init__(self, r): self._r = r
        def result(self, timeout=None): return self._r

    def _schedule(coro, label):
        labels.append(label)
        return _Fut(asyncio.run(coro))

    runner._schedule = _schedule
    runner._close_native_stream_boundary = lambda *a, **k: None
    real_register = cm.register
    seen = {"n": 0}

    def _register(**kwargs):
        entry = real_register(**kwargs)
        index = seen["n"]
        seen["n"] += 1
        target = answers[index] if index < len(answers) else None
        if target is not None:
            cm.resolve_gateway_clarify(kwargs["clarify_id"], target)
        return entry

    with patch.object(cm, "register", _register), \
            patch("tools.clarify_gateway.get_clarify_timeout", return_value=0.05):
        if via_tool:
            from tools.clarify_tool import clarify_tool
            return clarify_tool(questions, callback=runner._clarify_callback_sync), labels
        return runner._clarify_callback_sync(questions), labels


def test_timeout_retires_the_native_card_with_the_expired_notice():
    adapter = _CardAdapter()
    reply, _labels = _run_clarify(adapter)
    assert reply["outcome"] == "timed_out"
    assert reply["notice"].startswith("[user did not respond")
    assert len(adapter.retired) == 1
    assert "expired" in adapter.retired[0][1].lower()


def test_timeout_schedules_nothing_for_adapters_without_a_card():
    _response, labels = _run_clarify(_TextAdapter())
    assert labels == ["Clarify send failed to schedule"]


def test_real_answer_starting_with_a_bracket_is_not_mistaken_for_a_sentinel():
    """'[A] staging' is a user answer, not a timeout: no card retirement, typing re-armed."""
    adapter = _CardAdapter()
    reply, labels = _run_clarify(adapter, answers=("[A] staging",))
    assert reply == {"answers": {"q0": "[A] staging"}, "outcome": "submitted"}
    assert adapter.retired == []
    assert labels == ["Clarify send failed to schedule"]
    assert adapter.resumed == 1  # a lone card re-arms typing the moment it is answered


# --- Batches: one card per question, stop at the first unanswered one -----


_THREE_QUESTIONS = [{"qid": f"q{i}", "question": q, "choices": ["a", "b"], "multi_select": False}
                    for i, q in enumerate(("One?", "Two?", "Three?"))]


@pytest.mark.parametrize("answers,asked,payload,resumed", [
    # Nobody answers question 1: the batch ends there instead of re-asking — every further
    # question used to cost another full clarify_timeout — and reports the walk-away.
    ((), ["One?"], {"answers": {}, "outcome": "timed_out", "notice": "[user did not respond within 0m]"}, 0),
    # Answers already given survive; the unanswered question is not invented.
    (("use postgres",), ["One?", "Two?"],
     {"answers": {"q0": "use postgres"}, "outcome": "timed_out", "notice": "[user did not respond within 0m]"}, 0),
    # A fully answered batch re-arms typing once, at the end: between two cards the re-arm
    # would only open a bubble the next question's boundary finalizes.
    (("one", "two", "three"), ["One?", "Two?", "Three?"],
     {"answers": {"q0": "one", "q1": "two", "q2": "three"}, "outcome": "submitted"}, 1),
])
def test_batch_routing(answers, asked, payload, resumed):
    adapter = _CardAdapter()
    reply, _labels = _run_clarify(adapter, questions=_THREE_QUESTIONS, answers=answers)
    assert _asked(adapter) == asked
    assert reply == payload
    assert adapter.resumed == resumed
    if payload["outcome"] == "timed_out":
        assert len(adapter.retired) == 1  # the card that expired is retired


class _UndeliverableAdapter(_CardAdapter):
    """Telegram's ``_send_prompt`` shape when the Bot API rejects the card."""

    async def send_clarify(self, **kwargs):
        self.asked.append(kwargs["question"])
        return SendResult(success=False, error="Bad Request: chat not found")


@pytest.mark.parametrize("adapter_cls,outcome,notice", [
    (_CardAdapter, "timed_out", "[user did not respond within 0m]"),
    # #112684: an undelivered card must not read as user inactivity — the outcome says
    # undelivered and the delivery notice rides along.
    (_UndeliverableAdapter, "undelivered", "[clarify prompt could not be delivered]"),
])
def test_clarify_tool_batch_route_answers_and_stops_together(adapter_cls, outcome, notice):
    """End to end: the tool sees unanswered questions + the outcome + the surface's notice."""
    adapter = adapter_cls()
    raw, _labels = _run_clarify(
        adapter, questions=_THREE_QUESTIONS[:2], via_tool=True)
    assert _asked(adapter) == ["One?"]
    result = json.loads(raw)
    assert result["outcome"] == outcome
    assert result["notice"] == notice
    assert [r["status"] for r in result["responses"]] == ["unanswered", "unanswered"]
    assert [r["user_response"] for r in result["responses"]] == [None, None]
