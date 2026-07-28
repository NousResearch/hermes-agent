"""Clarify choice mode must not destroy an answer the user typed.

While the clarify choice list is on screen the input buffer stays editable, so
a user can type a custom answer without first selecting "Other". Two paths used
to throw that answer away:

* Enter submitted the *highlighted* choice and never looked at the buffer.
* Every digit key was bound to immediate numbered selection, so typing an
  answer containing a digit resolved the prompt mid-word.

These tests drive ``HermesCLI._handle_clarify_enter`` and the handler object
returned by ``HermesCLI._make_clarify_number_handler`` — the exact callables the
``run()`` key bindings execute — rather than re-implementing the conditionals.
"""

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI


class _FakeBuffer:
    def __init__(self, text=""):
        self.text = text
        self.cursor_position = len(text)

    def reset(self, append_to_history=False):
        self.text = ""
        self.cursor_position = 0

    def insert_text(self, data):
        pos = self.cursor_position
        self.text = self.text[:pos] + data + self.text[pos:]
        self.cursor_position = pos + len(data)


def _make_event(buffer):
    return SimpleNamespace(
        app=SimpleNamespace(invalidate=MagicMock(), current_buffer=buffer)
    )


def _make_cli_stub(choices, *, selected=0, multi_select=False, checked=None):
    cli = HermesCLI.__new__(HermesCLI)
    cli._clarify_state = {
        "question": "Which cluster should I deploy to?",
        "choices": list(choices),
        "selected": selected,
        "multi_select": multi_select,
        "selected_indices": set(checked or ()) if multi_select else None,
        "response_queue": queue.Queue(),
    }
    cli._clarify_freetext = False
    cli._clarify_multi_base = None
    return cli


def _drain(cli_state):
    return cli_state["response_queue"].get_nowait()


class TestClarifyEnterTypedAnswer:
    def test_typed_answer_wins_over_highlighted_choice(self):
        cli = _make_cli_stub(["staging", "prod"], selected=0)
        state = cli._clarify_state
        buffer = _FakeBuffer("use the canary cluster")
        event = _make_event(buffer)

        assert cli._handle_clarify_enter(event) is True

        assert _drain(state) == "use the canary cluster"
        assert cli._clarify_state is None
        assert cli._clarify_freetext is False
        assert buffer.text == ""

    def test_typed_answer_is_appended_to_checked_multi_select_choices(self):
        cli = _make_cli_stub(
            ["choice_a", "choice_b", "choice_c"],
            multi_select=True,
            checked={0, 2},
        )
        state = cli._clarify_state
        event = _make_event(_FakeBuffer("and roll back on error"))

        assert cli._handle_clarify_enter(event) is True

        # Same shape the existing "Other"-plus-choices freetext path produces.
        assert _drain(state) == "choice_a, choice_c, and roll back on error"
        assert cli._clarify_state is None
        assert cli._clarify_multi_base is None

    def test_typed_answer_does_not_duplicate_the_other_entry(self):
        # Index 3 == len(choices) is the synthetic "Other" row; the typed text
        # *is* the other answer, so it must not also appear as a choice.
        cli = _make_cli_stub(
            ["choice_a", "choice_b", "choice_c"],
            multi_select=True,
            checked={1, 3},
        )
        state = cli._clarify_state
        event = _make_event(_FakeBuffer("something else entirely"))

        assert cli._handle_clarify_enter(event) is True
        assert _drain(state) == "choice_b, something else entirely"

    def test_typed_answer_with_nothing_checked_submits_only_the_text(self):
        cli = _make_cli_stub(["choice_a", "choice_b"], multi_select=True)
        state = cli._clarify_state
        event = _make_event(_FakeBuffer("neither, use the old one"))

        assert cli._handle_clarify_enter(event) is True
        assert _drain(state) == "neither, use the old one"

    def test_empty_buffer_still_submits_the_highlighted_choice(self):
        cli = _make_cli_stub(["staging", "prod"], selected=1)
        state = cli._clarify_state
        event = _make_event(_FakeBuffer(""))

        assert cli._handle_clarify_enter(event) is True
        assert _drain(state) == "prod"
        assert cli._clarify_state is None

    def test_empty_buffer_on_other_row_still_switches_to_freetext(self):
        cli = _make_cli_stub(["staging", "prod"], selected=2)
        state = cli._clarify_state
        event = _make_event(_FakeBuffer(""))

        assert cli._handle_clarify_enter(event) is True
        assert cli._clarify_freetext is True
        assert cli._clarify_state is state
        assert state["response_queue"].empty()

    def test_empty_buffer_multi_select_still_joins_checked_choices(self):
        cli = _make_cli_stub(
            ["choice_a", "choice_b", "choice_c"],
            multi_select=True,
            checked={0, 2},
        )
        state = cli._clarify_state
        event = _make_event(_FakeBuffer(""))

        assert cli._handle_clarify_enter(event) is True
        assert _drain(state) == "choice_a, choice_c"

    def test_empty_buffer_multi_select_with_nothing_checked_submits_empty(self):
        cli = _make_cli_stub(["choice_a", "choice_b"], multi_select=True)
        state = cli._clarify_state
        event = _make_event(_FakeBuffer(""))

        assert cli._handle_clarify_enter(event) is True
        assert _drain(state) == ""

    def test_empty_buffer_multi_select_other_plus_choices_stores_base(self):
        cli = _make_cli_stub(
            ["choice_a", "choice_b"], multi_select=True, checked={0, 2}
        )
        state = cli._clarify_state
        event = _make_event(_FakeBuffer(""))

        assert cli._handle_clarify_enter(event) is True
        assert cli._clarify_multi_base == ["choice_a"]
        assert cli._clarify_freetext is True
        assert state["response_queue"].empty()

    def test_returns_false_when_no_clarify_prompt_is_open(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli._clarify_state = None
        cli._clarify_freetext = False
        assert cli._handle_clarify_enter(_make_event(_FakeBuffer("hello"))) is False

    def test_returns_false_in_freetext_mode(self):
        cli = _make_cli_stub(["staging", "prod"])
        cli._clarify_freetext = True
        assert cli._handle_clarify_enter(_make_event(_FakeBuffer("hi"))) is False


class TestClarifyDigitShortcut:
    def test_numeric_custom_answer_is_typed_not_selected(self):
        # "0" maps to choice index 9 — the case that made numeric answers unsafe.
        cli = _make_cli_stub(["staging", "prod"])
        state = cli._clarify_state
        buffer = _FakeBuffer("port 808")
        handler = cli._make_clarify_number_handler(9, "0")

        handler(_make_event(buffer))

        assert buffer.text == "port 8080"
        assert state["response_queue"].empty()
        assert cli._clarify_state is state
        assert cli._clarify_freetext is False

    def test_digit_within_range_is_typed_when_a_draft_exists(self):
        cli = _make_cli_stub(["staging", "prod"])
        state = cli._clarify_state
        buffer = _FakeBuffer("keep v")
        handler = cli._make_clarify_number_handler(1, "2")

        handler(_make_event(buffer))

        assert buffer.text == "keep v2"
        assert state["response_queue"].empty()
        assert cli._clarify_state is state

    def test_digit_above_choice_count_is_typed_instead_of_swallowed(self):
        cli = _make_cli_stub(["staging", "prod"])
        buffer = _FakeBuffer("scale to ")
        handler = cli._make_clarify_number_handler(6, "7")

        handler(_make_event(buffer))

        assert buffer.text == "scale to 7"

    def test_digit_is_typed_in_multi_select_mode_when_a_draft_exists(self):
        cli = _make_cli_stub(
            ["choice_a", "choice_b"], multi_select=True
        )
        state = cli._clarify_state
        buffer = _FakeBuffer("only after step ")
        handler = cli._make_clarify_number_handler(0, "1")

        handler(_make_event(buffer))

        assert buffer.text == "only after step 1"
        assert state["selected_indices"] == set()

    def test_empty_draft_still_quick_selects(self):
        cli = _make_cli_stub(["staging", "prod"])
        state = cli._clarify_state
        handler = cli._make_clarify_number_handler(1, "2")

        handler(_make_event(_FakeBuffer("")))

        assert _drain(state) == "prod"
        assert cli._clarify_state is None

    def test_empty_draft_still_toggles_multi_select_checkboxes(self):
        cli = _make_cli_stub(["choice_a", "choice_b"], multi_select=True)
        state = cli._clarify_state
        handler = cli._make_clarify_number_handler(0, "1")

        handler(_make_event(_FakeBuffer("")))
        assert state["selected_indices"] == {0}

        handler(_make_event(_FakeBuffer("")))
        assert state["selected_indices"] == set()

    def test_empty_draft_on_other_index_still_switches_to_freetext(self):
        cli = _make_cli_stub(["staging", "prod"])
        handler = cli._make_clarify_number_handler(2, "3")

        handler(_make_event(_FakeBuffer("")))

        assert cli._clarify_freetext is True
        assert cli._clarify_state is not None
