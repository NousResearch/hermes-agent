"""Batch (multi-question) clarify panel state machine — CLI side.

Drives ``_clarify_callback`` with a ``questions`` list on a background
thread (the way the agent thread calls it) and simulates the keybinding
handlers by calling the same helper methods they call
(``_clarify_batch_set_active`` for Tab, ``_clarify_batch_enter`` for
Enter, ``_clarify_batch_lock`` for the freetext submit path). No real
terminal needed — mirrors tests/cli/test_cli_approval_ui.py.
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_multi_base = None
    cli._clarify_prefill = ""
    cli._clarify_deadline = None
    cli._paint_now = MagicMock()
    cli._persist_prompt_summary = MagicMock()
    return cli


def _q(index, question, choices=None, multi_select=False):
    """One normalized batch entry, shaped like _normalize_questions output."""
    return {
        "qid": f"q{index}",
        "id": None,
        "question": question,
        "choices": list(choices) if choices else None,
        "choices_offered": list(choices) if choices else None,
        "multi_select": bool(multi_select) and bool(choices),
    }


def _start_batch(cli, questions):
    """Run the batch callback on a thread; wait for the panel state."""
    result = {}

    def _run():
        result["value"] = cli._clarify_callback("", None, questions=questions)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.time() + 2
    while cli._clarify_state is None and time.time() < deadline:
        time.sleep(0.01)
    assert cli._clarify_state is not None
    return thread, result


class TestClarifyBatchPanel:
    def test_all_locked_returns_answers_dict_keyed_by_qid(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Color?", ["red", "blue"]),
            _q(1, "Size?", ["small", "large"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        assert state["active"] == 0
        assert state["choices"] == ["red", "blue"]

        # Enter locks the active question's highlighted choice, then the
        # cursor advances to the next unanswered question.
        cli._clarify_batch_enter(state)
        assert state["answers"] == {"q0": "red"}
        assert state["active"] == 1

        state["selected"] = 1
        cli._clarify_batch_enter(state)

        thread.join(timeout=2)
        assert result["value"] == {"answers": {"q0": "red", "q1": "large"}}
        assert cli._clarify_state is None

    def test_any_order_answering_via_tab_cycle(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "First?", ["a", "b"]),
            _q(1, "Second?", ["c", "d"]),
            _q(2, "Third?", ["e", "f"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        # Tab twice: q0 -> q1 -> q2 (what the tab keybinding does).
        cli._clarify_batch_set_active(state, (state["active"] + 1) % 3)
        cli._clarify_batch_set_active(state, (state["active"] + 1) % 3)
        assert state["active"] == 2

        cli._clarify_batch_enter(state)  # lock q2 = "e"
        # Advance wraps to the next unanswered question (q0).
        assert state["active"] == 0

        state["selected"] = 1
        cli._clarify_batch_enter(state)  # lock q0 = "b"
        assert state["active"] == 1

        cli._clarify_batch_enter(state)  # lock q1 = "c"

        thread.join(timeout=2)
        assert result["value"] == {
            "answers": {"q0": "b", "q1": "c", "q2": "e"}
        }

    def test_reanswer_overwrites_before_completion(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Approach?", ["quick", "thorough"]),
            _q(1, "Scope?", ["narrow", "wide"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        cli._clarify_batch_enter(state)  # lock q0 = "quick"
        assert state["answers"]["q0"] == "quick"
        assert state["active"] == 1

        # Tab back to the answered question and change the answer.
        cli._clarify_batch_set_active(state, 0)
        state["selected"] = 1
        cli._clarify_batch_enter(state)  # overwrite q0 = "thorough"
        assert state["answers"]["q0"] == "thorough"
        # Advance lands on the still-unanswered q1.
        assert state["active"] == 1

        cli._clarify_batch_enter(state)  # lock q1 = "narrow"

        thread.join(timeout=2)
        assert result["value"] == {
            "answers": {"q0": "thorough", "q1": "narrow"}
        }

    def test_timeout_returns_partials_with_timed_out_flag(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Answered?", ["yes", "no"]),
            _q(1, "Never answered?", ["x", "y"]),
        ]
        with patch(
            "tools.clarify_gateway.resolve_clarify_timeout", return_value=1
        ):
            thread, result = _start_batch(cli, questions)
            state = cli._clarify_state
            cli._clarify_batch_enter(state)  # lock q0 only
            thread.join(timeout=5)

        assert not thread.is_alive()
        assert result["value"] == {"answers": {"q0": "yes"}, "timed_out": True}
        assert cli._clarify_state is None

    def test_multi_select_lock_produces_json_array_string(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Toppings?", ["ham", "olives", "basil"], multi_select=True),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        assert state["multi_select"] is True
        state["selected_indices"].update({0, 2})
        cli._clarify_batch_enter(state)

        thread.join(timeout=2)
        answer = result["value"]["answers"]["q0"]
        assert isinstance(answer, str)
        assert json.loads(answer) == ["ham", "basil"]

    def test_open_ended_question_locks_typed_answer(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Anything else?"),
            _q(1, "Pick one", ["a", "b"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        # Open-ended active question drops straight into freetext.
        assert cli._clarify_freetext is True
        # The Enter freetext submit path locks the typed text.
        cli._clarify_freetext = False
        cli._clarify_batch_lock(state, "custom words")
        assert state["active"] == 1

        cli._clarify_batch_enter(state)

        thread.join(timeout=2)
        assert result["value"] == {
            "answers": {"q0": "custom words", "q1": "a"}
        }

    def test_locked_question_persists_scrollback_summary(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Color?", ["red", "blue"]),
            _q(1, "Size?", ["small", "large"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        cli._clarify_batch_enter(state)
        cli._clarify_batch_enter(state)
        thread.join(timeout=2)

        calls = cli._persist_prompt_summary.call_args_list
        assert len(calls) == 2
        assert calls[0].args == ("?", "Clarify", "Color?", "red")
        assert calls[1].args == ("?", "Clarify", "Size?", "small")

    def test_single_question_path_returns_plain_string(self):
        cli = _make_cli_stub()
        result = {}

        def _run():
            result["value"] = cli._clarify_callback("Pick?", ["a", "b"])

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        deadline = time.time() + 2
        while cli._clarify_state is None and time.time() < deadline:
            time.sleep(0.01)
        assert cli._clarify_state is not None
        assert "questions" not in cli._clarify_state

        cli._clarify_state["response_queue"].put("a")
        thread.join(timeout=2)
        assert result["value"] == "a"


class TestClarifyBatchNavigation:
    """Shift-Tab, answer restore on re-visit, and Other edit-prefill."""

    def test_backward_navigation_wraps(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Color?", ["red", "blue"]),
            _q(1, "Size?", ["small", "large"]),
            _q(2, "Speed?", ["slow", "fast"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        # Shift-Tab from question 0 wraps to the last question.
        cli._clarify_batch_set_active(state, (state["active"] - 1) % 3)
        assert state["active"] == 2
        cli._clarify_batch_set_active(state, (state["active"] - 1) % 3)
        assert state["active"] == 1

        state["response_queue"].put("cancel")
        thread.join(timeout=2)

    def test_revisit_choice_answer_restores_cursor(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Color?", ["red", "blue"]),
            _q(1, "Size?", ["small", "large"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        # Lock "blue" (index 1) on q0; the cursor advances to q1.
        state["selected"] = 1
        cli._clarify_batch_enter(state)
        assert state["active"] == 1

        # Tab back to q0: the cursor sits on the earlier answer, not row 0.
        cli._clarify_batch_set_active(state, 0)
        assert state["selected"] == 1

        state["response_queue"].put("cancel")
        thread.join(timeout=2)

    def test_revisit_other_answer_highlights_other_and_prefills_edit(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Color?", ["red", "blue"]),
            _q(1, "Size?", ["small", "large"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        # Answer q0 via Other: select the Other row, then the freetext
        # submit path locks the typed text with its meta.
        state["selected"] = 2
        cli._clarify_batch_enter(state)
        assert cli._clarify_freetext is True
        cli._clarify_freetext = False
        cli._clarify_batch_lock(
            state, "chartreuse", meta={"kind": "other", "other_text": "chartreuse"}
        )
        assert state["active"] == 1

        # Tab back to q0: the cursor highlights the Other row.
        cli._clarify_batch_set_active(state, 0)
        assert state["selected"] == 2

        # Enter on the answered Other switches to freetext and prefills the
        # earlier text for editing.
        cli._clarify_batch_enter(state)
        assert cli._clarify_freetext is True
        assert cli._clarify_prefill == "chartreuse"

        state["response_queue"].put("cancel")
        thread.join(timeout=2)

    def test_reanswer_overwrites_and_updates_meta(self):
        cli = _make_cli_stub()
        questions = [
            _q(0, "Color?", ["red", "blue"]),
            _q(1, "Size?", ["small", "large"]),
        ]
        thread, result = _start_batch(cli, questions)
        state = cli._clarify_state

        # First answer via Other.
        cli._clarify_batch_lock(
            state, "teal", meta={"kind": "other", "other_text": "teal"}
        )
        # Re-visit and overwrite with a plain choice.
        cli._clarify_batch_set_active(state, 0)
        assert state["selected"] == 2
        state["selected"] = 0
        cli._clarify_batch_enter(state)
        assert state["answers"]["q0"] == "red"
        assert state["answer_meta"]["q0"] == {"kind": "choice"}

        # Finish q1 so the batch resolves with the overwritten answer.
        cli._clarify_batch_set_active(state, 1)
        cli._clarify_batch_enter(state)
        thread.join(timeout=2)
        assert result["value"] == {"answers": {"q0": "red", "q1": "small"}}

