"""Tests for shared slash command definitions and autocomplete."""

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from hermes_cli.commands import COMMANDS, SlashCommandCompleter


def _completions(completer: SlashCommandCompleter, text: str):
    return list(
        completer.get_completions(
            Document(text=text),
            CompleteEvent(completion_requested=True),
        )
    )


class TestCommands:
    def test_shared_commands_include_cli_specific_entries(self):
        assert COMMANDS["/paste"] == "Check clipboard for an image and attach it"
        assert COMMANDS["/reload-mcp"] == "Reload MCP servers from config.yaml"


class TestSlashCommandCompleter:
    def test_builtin_prefix_completion_uses_shared_registry(self):
        completions = _completions(SlashCommandCompleter(), "/re")
        texts = {item.text for item in completions}

        assert "reset" in texts
        assert "retry" in texts
        assert "reload-mcp" in texts

    def test_exact_match_completion_adds_trailing_space(self):
        completions = _completions(SlashCommandCompleter(), "/help")

        assert [item.text for item in completions] == ["help "]

    def test_skill_commands_are_completed_from_provider(self):
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: {
                "/gif-search": {"description": "Search for GIFs across providers"},
            }
        )

        completions = _completions(completer, "/gif")

        assert len(completions) == 1
        assert completions[0].text == "gif-search"
        assert str(completions[0].display) == "/gif-search"
        assert "⚡ Search for GIFs across providers" == str(completions[0].display_meta)
