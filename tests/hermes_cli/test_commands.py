"""Tests for the central command registry and autocomplete."""

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from hermes_cli.commands import (
    COMMAND_REGISTRY,
    COMMANDS,
    COMMANDS_BY_CATEGORY,
    CommandDef,
    GATEWAY_KNOWN_COMMANDS,
    SUBCOMMANDS,
    SlashCommandAutoSuggest,
    SlashCommandCompleter,
    gateway_help_lines,
    resolve_command,
    slack_subcommand_map,
    telegram_bot_commands,
)


def _completions(completer: SlashCommandCompleter, text: str):
    return list(
        completer.get_completions(
            Document(text=text),
            CompleteEvent(completion_requested=True),
        )
    )


# ---------------------------------------------------------------------------
# CommandDef registry tests
# ---------------------------------------------------------------------------

class TestCommandRegistry:
    def test_registry_is_nonempty(self):
        assert len(COMMAND_REGISTRY) > 30

    def test_every_entry_is_commanddef(self):
        for entry in COMMAND_REGISTRY:
            assert isinstance(entry, CommandDef), f"Unexpected type: {type(entry)}"

    def test_no_duplicate_canonical_names(self):
        names = [cmd.name for cmd in COMMAND_REGISTRY]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_no_alias_collides_with_canonical_name(self):
        """An alias must not shadow another command's canonical name."""
        canonical_names = {cmd.name for cmd in COMMAND_REGISTRY}
        for cmd in COMMAND_REGISTRY:
            for alias in cmd.aliases:
                if alias in canonical_names:
                    # reset -> new is intentional (reset IS an alias for new)
                    target = next(c for c in COMMAND_REGISTRY if c.name == alias)
                    # This should only happen if the alias points to the same entry
                    assert resolve_command(alias).name == cmd.name or alias == cmd.name, \
                        f"Alias '{alias}' of '{cmd.name}' shadows canonical '{target.name}'"

    def test_every_entry_has_valid_category(self):
        valid_categories = {"Session", "Configuration", "Tools & Skills", "Info", "Exit"}
        for cmd in COMMAND_REGISTRY:
            assert cmd.category in valid_categories, f"{cmd.name} has invalid category '{cmd.category}'"

    def test_cli_only_and_gateway_only_are_mutually_exclusive(self):
        for cmd in COMMAND_REGISTRY:
            assert not (cmd.cli_only and cmd.gateway_only), \
                f"{cmd.name} cannot be both cli_only and gateway_only"


# ---------------------------------------------------------------------------
# resolve_command tests
# ---------------------------------------------------------------------------

class TestResolveCommand:
    def test_canonical_name_resolves(self):
        assert resolve_command("help").name == "help"
        assert resolve_command("background").name == "background"

    def test_alias_resolves_to_canonical(self):
        assert resolve_command("bg").name == "background"
        assert resolve_command("reset").name == "new"
        assert resolve_command("q").name == "quit"
        assert resolve_command("exit").name == "quit"
        assert resolve_command("gateway").name == "platforms"
        assert resolve_command("set-home").name == "sethome"
        assert resolve_command("reload_mcp").name == "reload-mcp"

    def test_leading_slash_stripped(self):
        assert resolve_command("/help").name == "help"
        assert resolve_command("/bg").name == "background"

    def test_unknown_returns_none(self):
        assert resolve_command("nonexistent") is None
        assert resolve_command("") is None


# ---------------------------------------------------------------------------
# Derived dicts (backwards compat)
# ---------------------------------------------------------------------------

class TestDerivedDicts:
    def test_commands_dict_excludes_gateway_only(self):
        """gateway_only commands should NOT appear in the CLI COMMANDS dict."""
        for cmd in COMMAND_REGISTRY:
            if cmd.gateway_only:
                assert f"/{cmd.name}" not in COMMANDS, \
                    f"gateway_only command /{cmd.name} should not be in COMMANDS"

    def test_commands_dict_includes_all_cli_commands(self):
        for cmd in COMMAND_REGISTRY:
            if not cmd.gateway_only:
                assert f"/{cmd.name}" in COMMANDS, \
                    f"/{cmd.name} missing from COMMANDS dict"

    def test_commands_dict_includes_aliases(self):
        assert "/bg" in COMMANDS
        assert "/reset" in COMMANDS
        assert "/q" in COMMANDS
        assert "/exit" in COMMANDS
        assert "/reload_mcp" in COMMANDS
        assert "/gateway" in COMMANDS

    def test_commands_by_category_covers_all_categories(self):
        registry_categories = {cmd.category for cmd in COMMAND_REGISTRY if not cmd.gateway_only}
        assert set(COMMANDS_BY_CATEGORY.keys()) == registry_categories

    def test_every_command_has_nonempty_description(self):
        for cmd, desc in COMMANDS.items():
            assert isinstance(desc, str) and len(desc) > 0, f"{cmd} has empty description"


# ---------------------------------------------------------------------------
# Gateway helpers
# ---------------------------------------------------------------------------

class TestGatewayKnownCommands:
    def test_excludes_cli_only(self):
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only:
                assert cmd.name not in GATEWAY_KNOWN_COMMANDS, \
                    f"cli_only command '{cmd.name}' should not be in GATEWAY_KNOWN_COMMANDS"

    def test_includes_gateway_commands(self):
        for cmd in COMMAND_REGISTRY:
            if not cmd.cli_only:
                assert cmd.name in GATEWAY_KNOWN_COMMANDS
                for alias in cmd.aliases:
                    assert alias in GATEWAY_KNOWN_COMMANDS

    def test_bg_alias_in_gateway(self):
        assert "bg" in GATEWAY_KNOWN_COMMANDS
        assert "background" in GATEWAY_KNOWN_COMMANDS

    def test_is_frozenset(self):
        assert isinstance(GATEWAY_KNOWN_COMMANDS, frozenset)


class TestGatewayHelpLines:
    def test_returns_nonempty_list(self):
        lines = gateway_help_lines()
        assert len(lines) > 10

    def test_excludes_cli_only_commands(self):
        lines = gateway_help_lines()
        joined = "\n".join(lines)
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only:
                assert f"`/{cmd.name}" not in joined, \
                    f"cli_only command /{cmd.name} should not be in gateway help"

    def test_includes_alias_note_for_bg(self):
        lines = gateway_help_lines()
        bg_line = [l for l in lines if "/background" in l]
        assert len(bg_line) == 1
        assert "/bg" in bg_line[0]


class TestTelegramBotCommands:
    def test_returns_list_of_tuples(self):
        cmds = telegram_bot_commands()
        assert len(cmds) > 10
        for name, desc in cmds:
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_no_hyphens_in_command_names(self):
        """Telegram does not support hyphens in command names."""
        for name, _ in telegram_bot_commands():
            assert "-" not in name, f"Telegram command '{name}' contains a hyphen"

    def test_excludes_cli_only(self):
        names = {name for name, _ in telegram_bot_commands()}
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only:
                tg_name = cmd.name.replace("-", "_")
                assert tg_name not in names


class TestSlackSubcommandMap:
    def test_returns_dict(self):
        mapping = slack_subcommand_map()
        assert isinstance(mapping, dict)
        assert len(mapping) > 10

    def test_values_are_slash_prefixed(self):
        for key, val in slack_subcommand_map().items():
            assert val.startswith("/"), f"Slack mapping for '{key}' should start with /"

    def test_includes_aliases(self):
        mapping = slack_subcommand_map()
        assert "bg" in mapping
        assert "reset" in mapping

    def test_excludes_cli_only(self):
        mapping = slack_subcommand_map()
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only:
                assert cmd.name not in mapping


# ---------------------------------------------------------------------------
# Autocomplete (SlashCommandCompleter)
# ---------------------------------------------------------------------------

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
        assert completions[0].display_meta_text == "Show available commands"

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


# ── SUBCOMMANDS extraction ──────────────────────────────────────────────


class TestSubcommands:
    def test_explicit_subcommands_extracted(self):
        """Commands with explicit subcommands on CommandDef are extracted."""
        assert "/prompt" in SUBCOMMANDS
        assert "clear" in SUBCOMMANDS["/prompt"]

    def test_reasoning_has_subcommands(self):
        assert "/reasoning" in SUBCOMMANDS
        subs = SUBCOMMANDS["/reasoning"]
        assert "high" in subs
        assert "show" in subs
        assert "hide" in subs

    def test_voice_has_subcommands(self):
        assert "/voice" in SUBCOMMANDS
        assert "on" in SUBCOMMANDS["/voice"]
        assert "off" in SUBCOMMANDS["/voice"]

    def test_cron_has_subcommands(self):
        assert "/cron" in SUBCOMMANDS
        assert "list" in SUBCOMMANDS["/cron"]
        assert "add" in SUBCOMMANDS["/cron"]

    def test_commands_without_subcommands_not_in_dict(self):
        """Plain commands should not appear in SUBCOMMANDS."""
        assert "/help" not in SUBCOMMANDS
        assert "/quit" not in SUBCOMMANDS
        assert "/clear" not in SUBCOMMANDS


# ── Subcommand tab completion ───────────────────────────────────────────


class TestSubcommandCompletion:
    def test_subcommand_completion_after_space(self):
        """Typing '/reasoning ' then Tab should show subcommands."""
        completions = _completions(SlashCommandCompleter(), "/reasoning ")
        texts = {c.text for c in completions}
        assert "high" in texts
        assert "show" in texts

    def test_subcommand_prefix_filters(self):
        """Typing '/reasoning sh' should only show 'show'."""
        completions = _completions(SlashCommandCompleter(), "/reasoning sh")
        texts = {c.text for c in completions}
        assert texts == {"show"}

    def test_subcommand_exact_match_suppressed(self):
        """Typing the full subcommand shouldn't re-suggest it."""
        completions = _completions(SlashCommandCompleter(), "/reasoning show")
        texts = {c.text for c in completions}
        assert "show" not in texts

    def test_no_subcommands_for_plain_command(self):
        """Commands without subcommands yield nothing after space."""
        completions = _completions(SlashCommandCompleter(), "/help ")
        assert completions == []


# ── Ghost text (SlashCommandAutoSuggest) ────────────────────────────────


def _suggestion(text: str, completer=None) -> str | None:
    """Get ghost text suggestion for given input."""
    suggest = SlashCommandAutoSuggest(completer=completer)
    doc = Document(text=text)

    class FakeBuffer:
        pass

    result = suggest.get_suggestion(FakeBuffer(), doc)
    return result.text if result else None


class TestGhostText:
    def test_command_name_suggestion(self):
        """/he → 'lp'"""
        assert _suggestion("/he") == "lp"

    def test_command_name_suggestion_reasoning(self):
        """/rea → 'soning'"""
        assert _suggestion("/rea") == "soning"

    def test_no_suggestion_for_complete_command(self):
        assert _suggestion("/help") is None

    def test_subcommand_suggestion(self):
        """/reasoning h → 'igh'"""
        assert _suggestion("/reasoning h") == "igh"

    def test_subcommand_suggestion_show(self):
        """/reasoning sh → 'ow'"""
        assert _suggestion("/reasoning sh") == "ow"

    def test_no_suggestion_for_non_slash(self):
        assert _suggestion("hello") is None
