"""SlashCommandAutoSuggest shows the pending next-prompt suggestion before history/slash ghost text."""

from prompt_toolkit.document import Document
from prompt_toolkit.auto_suggest import Suggestion

from hermes_cli.commands_completion import SlashCommandAutoSuggest


class _History:
    def get_suggestion(self, buffer, document):
        return Suggestion("-from-history") if document.text else None


def _ghost(text, pending):
    suggest = SlashCommandAutoSuggest(history_suggest=_History(), next_prompt_source=lambda: pending)
    result = suggest.get_suggestion(object(), Document(text=text))
    return result.text if result else None


def test_empty_buffer_shows_whole_suggestion():
    assert _ghost("", "run the full suite") == "run the full suite"


def test_prefix_shows_remainder():
    assert _ghost("run the", "run the full suite") == " full suite"


def test_divergence_falls_back_to_history():
    assert _ghost("fix", "run the full suite") == "-from-history"


def test_divergence_into_slash_falls_back_to_slash_ghost():
    assert _ghost("/he", "run the full suite") == "lp"


def test_fully_typed_suggestion_has_no_ghost_of_its_own():
    assert _ghost("run the full suite", "run the full suite") == "-from-history"


def test_no_pending_suggestion_keeps_old_behaviour():
    assert _ghost("", None) is None
    assert _ghost("/he", None) == "lp"
