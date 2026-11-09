"""An explicit empty clarify answer is a skip, not a timeout.

Desktop (``api/client.ts``) and the TUI (``useMainApp.ts``) send ``""`` for a skipped card and
document it as a skip. Single-question delivery used to fold that into the timeout branch, so
the model was told "[user did not respond within Nm]" with outcome ``timed_out`` and the rest
of a batch was never asked. A typed "skip" already arrives as ``SKIPPED``; the empty answer
now takes the same path. Desktop sends the same ``""`` for Stop, so during an interrupted turn
it reads as a cancel, and a Stop that lands between cards ends the batch before the next one.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from gateway.run_turn_runner_clarify_delivery import UNDELIVERED, _clarify_send_then_wait
from tests.gateway.test_clarify_card_retire_on_timeout import (
    _THREE_QUESTIONS,
    _CardAdapter,
    _run_clarify,
    _UndeliverableAdapter,
)
from tools.clarify_gateway import CANCELLED, SKIPPED


class _Sent:
    class _Result:
        success = True
        error = None

    def result(self, timeout=None):
        return self._Result()


def _delivery_with(response):
    clarify_mod = MagicMock()
    clarify_mod.CANCELLED = CANCELLED
    clarify_mod.SKIPPED = SKIPPED
    clarify_mod.get_clarify_timeout.return_value = 600
    clarify_mod.wait_for_response.return_value = response
    return _clarify_send_then_wait(_Sent(), clarify_id="cid", session_key="sk", clarify_mod=clarify_mod)


def _statuses(out):
    result = json.loads(out)
    return [r["status"] for r in result["responses"]], result


def _interrupted_after(calls):
    """``is_interrupted`` that reads False for the first *calls* checks, then True (a Stop)."""
    seq = iter([False] * calls)
    return lambda: next(seq, True)


# --- Delivery: what one card's answer turns into -----------------------------------


def test_empty_answer_is_a_skip():
    assert _delivery_with("") == (SKIPPED, True)


def test_empty_answer_during_a_stop_is_a_cancel():
    with patch("tools.interrupt.is_interrupted", return_value=True):
        assert _delivery_with("") == (CANCELLED, False)


def test_a_real_answer_during_a_stop_is_kept():
    with patch("tools.interrupt.is_interrupted", return_value=True):
        assert _delivery_with("use postgres") == ("use postgres", True)


def test_no_answer_is_still_a_timeout():
    assert _delivery_with(None) == ("[user did not respond within 10m]", False)


def test_cancelled_is_unchanged():
    assert _delivery_with(CANCELLED) == (CANCELLED, False)


# --- Batches: what the model is told --------------------------------------------------


def test_skip_mid_batch_keeps_asking_and_records_the_skip():
    adapter = _CardAdapter()
    reply, _labels = _run_clarify(adapter, questions=_THREE_QUESTIONS, answers=("one", "", "three"))
    assert reply == {"answers": {"q0": "one", "q1": None, "q2": "three"}, "outcome": "submitted"}
    assert [q.split("\n")[0] for q in adapter.asked] == ["One?", "Two?", "Three?"]
    assert adapter.retired == []  # nothing expired


def test_the_model_sees_skipped_not_unanswered():
    out, _labels = _run_clarify(_CardAdapter(), questions=_THREE_QUESTIONS, answers=("one", "", "three"),
                                via_tool=True)
    statuses, result = _statuses(out)
    assert statuses == ["answered", "skipped", "answered"]
    assert result["outcome"] == "submitted"
    assert "notice" not in result


def test_skip_on_a_single_question():
    out, _labels = _run_clarify(_CardAdapter(), answers=("",), via_tool=True)
    statuses, result = _statuses(out)
    assert statuses == ["skipped"] and result["outcome"] == "submitted" and "notice" not in result


def test_skip_on_the_last_question_rearms_typing_once():
    adapter = _CardAdapter()
    reply, _labels = _run_clarify(adapter, questions=_THREE_QUESTIONS, answers=("one", "two", ""))
    assert reply == {"answers": {"q0": "one", "q1": "two", "q2": None}, "outcome": "submitted"}
    assert adapter.resumed == 1


def test_every_question_skipped():
    out, _labels = _run_clarify(_CardAdapter(), questions=_THREE_QUESTIONS, answers=("", "", ""), via_tool=True)
    statuses, result = _statuses(out)
    assert statuses == ["skipped"] * 3 and result["outcome"] == "submitted"


def test_skip_on_a_multi_select_question():
    question = [{"qid": "q0", "question": "Pick any", "choices": ["a", "b", "c"], "multi_select": True}]
    out, _labels = _run_clarify(_CardAdapter(), questions=question, answers=("",), via_tool=True)
    statuses, result = _statuses(out)
    assert statuses == ["skipped"] and result["outcome"] == "submitted"


def test_whitespace_only_answer_reads_as_skipped():
    out, _labels = _run_clarify(_CardAdapter(), answers=("   ",), via_tool=True)
    statuses, _result = _statuses(out)
    assert statuses == ["skipped"]


def test_skip_then_no_answer_is_a_timeout_after_the_skip():
    adapter = _CardAdapter()
    reply, _labels = _run_clarify(adapter, questions=_THREE_QUESTIONS[:2], answers=("",))
    assert reply["answers"] == {"q0": None}
    assert reply["outcome"] == "timed_out"
    assert reply["notice"].startswith("[user did not respond")
    assert len(adapter.retired) == 1  # only the card that expired


def test_typed_skip_word_still_reads_as_skipped():
    from tools import clarify_gateway as cm

    cm.register("cid-typed", "sk-typed", "Pick one", ["a", "b"])
    assert cm.resolve_text_response_for_session("sk-typed", "skip")
    assert cm.wait_for_response("cid-typed", timeout=1) == SKIPPED


# --- Stop -------------------------------------------------------------------------------


def test_stop_sent_as_empty_answer_cancels_the_batch():
    adapter = _CardAdapter()
    with patch("tools.interrupt.is_interrupted", return_value=True):
        reply, _labels = _run_clarify(adapter, questions=_THREE_QUESTIONS, answers=("",))
    assert reply == {"answers": {}, "outcome": "cancelled"}
    assert [q.split("\n")[0] for q in adapter.asked] == ["One?"]


@pytest.mark.parametrize("first", ["", "one"])
def test_stop_after_a_settled_card_sends_no_next_card(first):
    """The first card settles (skip or answer) before the Stop lands: no second card goes out."""
    adapter = _CardAdapter()
    checks = 1 if first == "" else 0  # a skip reads the interrupt flag once in delivery
    with patch("tools.interrupt.is_interrupted", _interrupted_after(checks)):
        reply, _labels = _run_clarify(adapter, questions=_THREE_QUESTIONS, answers=(first, "two", "three"))
    assert reply["outcome"] == "cancelled"
    assert reply["answers"] == {"q0": None if first == "" else "one"}
    assert [q.split("\n")[0] for q in adapter.asked] == ["One?"]


# --- Delivery failures win over the answer ---------------------------------------------


def test_undelivered_card_stays_undelivered():
    reply, _labels = _run_clarify(_UndeliverableAdapter(), answers=("",))
    assert reply["outcome"] == "undelivered"
    assert reply["notice"] == UNDELIVERED


# --- Plugin command choices --------------------------------------------------------------


@pytest.mark.parametrize("result", [(SKIPPED, True), (CANCELLED, False)])
def test_plugin_choice_reads_a_skip_or_stop_as_no_choice(result):
    from gateway import plugin_command_choice as pcc
    from gateway import run_turn_runner_clarify_delivery as delivery

    runner = MagicMock()
    runner._delivery_adapter_for.return_value = MagicMock()
    source = MagicMock(chat_id="C1")
    with patch.object(delivery, "_clarify_send_then_wait", return_value=result), \
            patch("asyncio.run_coroutine_threadsafe", return_value=MagicMock()), \
            patch("tools.clarify_gateway.register"):
        assert pcc._ask_chat(runner, source, "sk", MagicMock(), "Pick one", ["a", "b"]) is None
