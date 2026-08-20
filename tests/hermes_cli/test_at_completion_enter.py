"""Tests for at_completion_to_accept_on_enter (Enter-selects @ completions).

The classic CLI's Enter handler submits; while an ``@`` file/folder completion
menu is open it must instead accept the highlighted row so several files can be
picked before sending. These tests pin the discriminator logic (accept only
``@``-prefixed rows; everything else keeps Enter=submit).
"""

from types import SimpleNamespace

from prompt_toolkit.completion import Completion

from hermes_cli.commands import at_completion_to_accept_on_enter


def _buffer(completions, current_index=None):
    """Minimal stand-in for prompt_toolkit's Buffer: complete_state is the only
    thing at_completion_to_accept_on_enter inspects."""
    state = SimpleNamespace(
        completions=list(completions),
        current_completion=(
            completions[current_index] if current_index is not None else None
        ),
    )
    return SimpleNamespace(complete_state=state)


class TestAtCompletionToAcceptOnEnter:
    def test_no_completion_state_returns_none(self):
        buf = SimpleNamespace(complete_state=None)
        assert at_completion_to_accept_on_enter(buf) is None

    def test_at_file_row_is_accepted(self):
        comp = Completion("@file:src/main.py", start_position=-2)
        assert at_completion_to_accept_on_enter(_buffer([comp], 0)) is comp

    def test_at_folder_row_is_accepted(self):
        comp = Completion("@folder:two-agent-slide/", start_position=-2)
        assert at_completion_to_accept_on_enter(_buffer([comp], 0)) is comp

    def test_static_refs_are_accepted(self):
        for text in ("@diff", "@staged", "@git:5", "@url:"):
            comp = Completion(text, start_position=-1)
            assert at_completion_to_accept_on_enter(_buffer([comp], 0)) is comp

    def test_slash_command_row_still_submits(self):
        comp = Completion("/help", start_position=-4)
        assert at_completion_to_accept_on_enter(_buffer([comp], 0)) is None

    def test_plain_text_row_still_submits(self):
        comp = Completion("hello", start_position=-5)
        assert at_completion_to_accept_on_enter(_buffer([comp], 0)) is None

    def test_nothing_highlighted_falls_back_to_first_row(self):
        first = Completion("@file:a.py", start_position=-2)
        second = Completion("/help", start_position=-4)
        assert at_completion_to_accept_on_enter(_buffer([first, second])) is first

    def test_nothing_highlighted_empty_list_returns_none(self):
        assert at_completion_to_accept_on_enter(_buffer([])) is None

    def test_highlighted_slash_among_at_rows_still_submits(self):
        # The user moved the cursor onto a slash row — respect that row, don't
        # sneak in the first @ row.
        at_row = Completion("@file:a.py", start_position=-2)
        slash_row = Completion("/help", start_position=-4)
        assert at_completion_to_accept_on_enter(_buffer([at_row, slash_row], 1)) is None
