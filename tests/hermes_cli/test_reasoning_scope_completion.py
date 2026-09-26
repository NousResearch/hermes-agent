"""RED first for #121129: /reasoning autocomplete must surface --global past the
first argument position (e.g. `/reasoning hide --<TAB>`)."""

from unittest.mock import MagicMock

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_plain_text

from hermes_cli.commands_completion import SlashCommandCompleter


def _texts(completer, text):
    doc = Document(text, cursor_position=len(text))
    return [to_plain_text(c.display) for c in completer.get_completions(doc, MagicMock())]


class TestReasoningScopeFlagCompletion:
    def test_first_arg_flag_still_offered(self):
        assert "--global" in _texts(SlashCommandCompleter(), "/reasoning --")

    def test_second_position_bare_space_offers_global(self):
        assert "--global" in _texts(SlashCommandCompleter(), "/reasoning hide ")

    def test_second_position_partial_flag_offers_global(self):
        assert "--global" in _texts(SlashCommandCompleter(), "/reasoning hide --g")

    def test_level_then_flag_offers_global(self):
        assert "--global" in _texts(SlashCommandCompleter(), "/reasoning high --")

    def test_global_not_repeated_once_typed(self):
        assert "--global" not in _texts(SlashCommandCompleter(), "/reasoning hide --global ")

    def test_fast_command_gets_same_flag_behavior(self):
        assert "--global" in _texts(SlashCommandCompleter(), "/fast auto --")
