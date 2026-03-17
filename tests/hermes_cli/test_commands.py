"""Tests for shared slash command definitions and autocomplete."""

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from hermes_cli.commands import COMMANDS, SlashCommandCompleter


# All commands that must be present in the shared COMMANDS dict.
EXPECTED_COMMANDS = {
    "/help", "/tools", "/toolsets", "/model", "/provider", "/prompt",
    "/personality", "/clear", "/history", "/new", "/reset", "/retry",
    "/undo", "/save", "/config", "/cron", "/skills", "/platforms",
    "/verbose", "/reasoning", "/compress", "/title", "/usage", "/insights", "/paste",
    "/reload-mcp", "/rollback", "/stop", "/background", "/bg", "/skin", "/voice", "/browser", "/quit",
    "/plugins",
}


def _completions(completer: SlashCommandCompleter, text: str):
    return list(
        completer.get_completions(
            Document(text=text),
            CompleteEvent(completion_requested=True),
        )
    )


class TestCommands:
    def test_shared_commands_include_cli_specific_entries(self):
        """Entries that previously only existed in cli.py are now in the shared dict."""
        assert COMMANDS["/paste"] == "Check clipboard for an image and attach it"
        assert COMMANDS["/reload-mcp"] == "Reload MCP servers from config.yaml"

    def test_all_expected_commands_present(self):
        """Regression guard — every known command must appear in the shared dict."""
        assert set(COMMANDS.keys()) == EXPECTED_COMMANDS

    def test_every_command_has_nonempty_description(self):
        for cmd, desc in COMMANDS.items():
            assert isinstance(desc, str) and len(desc) > 0, f"{cmd} has empty description"


class TestSlashCommandCompleter:
    # -- basic prefix completion -----------------------------------------

    def test_builtin_prefix_completion_uses_shared_registry(self):
        completions = _completions(SlashCommandCompleter(), "/re")
        texts = {item.text for item in completions}

        assert "reset" in texts
        assert "retry" in texts
        assert "reload-mcp" in texts

    def test_builtin_completion_display_meta_shows_description(self):
        completions = _completions(SlashCommandCompleter(), "/help")
        assert len(completions) == 1
        assert completions[0].display_meta_text == "Show this help message"

    # -- exact-match trailing space --------------------------------------

    def test_exact_match_completion_adds_trailing_space(self):
        completions = _completions(SlashCommandCompleter(), "/help")

        assert [item.text for item in completions] == ["help "]

    def test_partial_match_does_not_add_trailing_space(self):
        completions = _completions(SlashCommandCompleter(), "/hel")

        assert [item.text for item in completions] == ["help"]

    # -- non-slash input returns nothing ---------------------------------

    def test_no_completions_for_non_slash_input(self):
        assert _completions(SlashCommandCompleter(), "help") == []

    def test_no_completions_for_empty_input(self):
        assert _completions(SlashCommandCompleter(), "") == []

    # -- skill commands via provider ------------------------------------

    def test_skill_commands_are_completed_from_provider(self):
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: {
                "/gif-search": {"description": "Search for GIFs across providers"},
            }
        )

        completions = _completions(completer, "/gif")

        assert len(completions) == 1
        assert completions[0].text == "gif-search"
        assert completions[0].display_text == "/gif-search"
        assert completions[0].display_meta_text == "⚡ Search for GIFs across providers"

    def test_skill_exact_match_adds_trailing_space(self):
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: {
                "/gif-search": {"description": "Search for GIFs"},
            }
        )

        completions = _completions(completer, "/gif-search")

        assert len(completions) == 1
        assert completions[0].text == "gif-search "

    def test_no_skill_provider_means_no_skill_completions(self):
        """Default (None) provider should not blow up or add completions."""
        completer = SlashCommandCompleter()
        completions = _completions(completer, "/gif")
        # /gif doesn't match any builtin command
        assert completions == []

    def test_skill_provider_exception_is_swallowed(self):
        """A broken provider should not crash autocomplete."""
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # Should return builtin matches only, no crash
        completions = _completions(completer, "/he")
        texts = {item.text for item in completions}
        assert "help" in texts

    def test_skill_description_truncated_at_50_chars(self):
        long_desc = "A" * 80
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: {
                "/long-skill": {"description": long_desc},
            }
        )
        completions = _completions(completer, "/long")
        assert len(completions) == 1
        meta = completions[0].display_meta_text
        # "⚡ " prefix + 50 chars + "..."
        assert meta == f"⚡ {'A' * 50}..."

    def test_skill_missing_description_uses_fallback(self):
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: {
                "/no-desc": {},
            }
        )
        completions = _completions(completer, "/no-desc")
        assert len(completions) == 1
        assert "Skill command" in completions[0].display_meta_text
