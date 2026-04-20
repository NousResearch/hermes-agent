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
    _CMD_NAME_LIMIT,
    _TG_NAME_LIMIT,
    _clamp_command_names,
    _clamp_telegram_names,
    _sanitize_telegram_name,
    discord_skill_commands,
    gateway_help_lines,
    resolve_command,
    slack_subcommand_map,
    telegram_bot_commands,
    telegram_menu_commands,
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

    def test_reasoning_subcommands_are_in_logical_order(self):
        reasoning = next(cmd for cmd in COMMAND_REGISTRY if cmd.name == "reasoning")
        assert reasoning.subcommands[:6] == (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        )

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
        assert resolve_command("copy").name == "copy"
        assert resolve_command("agents").name == "agents"

    def test_alias_resolves_to_canonical(self):
        assert resolve_command("bg").name == "background"
        assert resolve_command("reset").name == "new"
        assert resolve_command("q").name == "queue"
        assert resolve_command("exit").name == "quit"
        assert resolve_command("gateway").name == "platforms"
        assert resolve_command("set-home").name == "sethome"
        assert resolve_command("reload_mcp").name == "reload-mcp"
        assert resolve_command("tasks").name == "agents"

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
    def test_excludes_cli_only_without_config_gate(self):
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only and not cmd.gateway_config_gate:
                assert cmd.name not in GATEWAY_KNOWN_COMMANDS, \
                    f"cli_only command '{cmd.name}' should not be in GATEWAY_KNOWN_COMMANDS"

    def test_includes_config_gated_cli_only(self):
        """Commands with gateway_config_gate are always in GATEWAY_KNOWN_COMMANDS."""
        for cmd in COMMAND_REGISTRY:
            if cmd.gateway_config_gate:
                assert cmd.name in GATEWAY_KNOWN_COMMANDS, \
                    f"config-gated command '{cmd.name}' should be in GATEWAY_KNOWN_COMMANDS"

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

    def test_excludes_cli_only_commands_without_config_gate(self):
        lines = gateway_help_lines()
        joined = "\n".join(lines)
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only and not cmd.gateway_config_gate:
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

    def test_all_names_valid_telegram_chars(self):
        """Telegram requires: lowercase a-z, 0-9, underscores only."""
        import re
        tg_valid = re.compile(r"^[a-z0-9_]+$")
        for name, _ in telegram_bot_commands():
            assert tg_valid.match(name), f"Invalid Telegram command name: {name!r}"

    def test_excludes_cli_only_without_config_gate(self):
        names = {name for name, _ in telegram_bot_commands()}
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only and not cmd.gateway_config_gate:
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

    def test_excludes_cli_only_without_config_gate(self):
        mapping = slack_subcommand_map()
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only and not cmd.gateway_config_gate:
                assert cmd.name not in mapping


# ---------------------------------------------------------------------------
# Config-gated gateway commands
# ---------------------------------------------------------------------------

class TestGatewayConfigGate:
    """Tests for the gateway_config_gate mechanism on CommandDef."""

    def test_verbose_has_config_gate(self):
        cmd = resolve_command("verbose")
        assert cmd is not None
        assert cmd.cli_only is True
        assert cmd.gateway_config_gate == "display.tool_progress_command"

    def test_verbose_in_gateway_known_commands(self):
        """Config-gated commands are always recognized by the gateway."""
        assert "verbose" in GATEWAY_KNOWN_COMMANDS

    def test_config_gate_excluded_from_help_when_off(self, tmp_path, monkeypatch):
        """When the config gate is falsy, the command should not appear in help."""
        # Write a config with the gate off (default)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("display:\n  tool_progress_command: false\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        lines = gateway_help_lines()
        joined = "\n".join(lines)
        assert "`/verbose" not in joined

    def test_config_gate_included_in_help_when_on(self, tmp_path, monkeypatch):
        """When the config gate is truthy, the command should appear in help."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("display:\n  tool_progress_command: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        lines = gateway_help_lines()
        joined = "\n".join(lines)
        assert "`/verbose" in joined

    def test_config_gate_excluded_from_telegram_when_off(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("display:\n  tool_progress_command: false\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        names = {name for name, _ in telegram_bot_commands()}
        assert "verbose" not in names

    def test_config_gate_included_in_telegram_when_on(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("display:\n  tool_progress_command: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        names = {name for name, _ in telegram_bot_commands()}
        assert "verbose" in names

    def test_config_gate_excluded_from_slack_when_off(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("display:\n  tool_progress_command: false\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        mapping = slack_subcommand_map()
        assert "verbose" not in mapping

    def test_config_gate_included_in_slack_when_on(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("display:\n  tool_progress_command: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        mapping = slack_subcommand_map()
        assert "verbose" in mapping


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
        assert "/skills" in SUBCOMMANDS
        assert "install" in SUBCOMMANDS["/skills"]

    def test_reasoning_has_subcommands(self):
        assert "/reasoning" in SUBCOMMANDS
        subs = SUBCOMMANDS["/reasoning"]
        assert "high" in subs
        assert "show" in subs
        assert "hide" in subs

    def test_fast_has_subcommands(self):
        assert "/fast" in SUBCOMMANDS
        subs = SUBCOMMANDS["/fast"]
        assert "fast" in subs
        assert "normal" in subs
        assert "status" in subs

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

    def test_fast_subcommand_completion_after_space(self):
        completions = _completions(SlashCommandCompleter(), "/fast ")
        texts = {c.text for c in completions}
        assert "fast" in texts
        assert "normal" in texts

    def test_fast_command_filtered_out_when_unavailable(self):
        completions = _completions(
            SlashCommandCompleter(command_filter=lambda cmd: cmd != "/fast"),
            "/fa",
        )
        texts = {c.text for c in completions}
        assert "fast" not in texts

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

    def test_fast_subcommand_suggestion(self):
        assert _suggestion("/fast f") == "ast"

    def test_fast_subcommand_suggestion_hidden_when_filtered(self):
        completer = SlashCommandCompleter(command_filter=lambda cmd: cmd != "/fast")
        assert _suggestion("/fa", completer=completer) is None

    def test_no_suggestion_for_non_slash(self):
        assert _suggestion("hello") is None


# ---------------------------------------------------------------------------
# Telegram command name sanitization
# ---------------------------------------------------------------------------


class TestSanitizeTelegramName:
    """Tests for _sanitize_telegram_name() — Telegram requires [a-z0-9_] only."""

    def test_hyphens_replaced_with_underscores(self):
        assert _sanitize_telegram_name("my-skill-name") == "my_skill_name"

    def test_plus_sign_stripped(self):
        """Regression: skill name 'Jellyfin + Jellystat 24h Summary'."""
        assert _sanitize_telegram_name("jellyfin-+-jellystat-24h-summary") == "jellyfin_jellystat_24h_summary"

    def test_slash_stripped(self):
        """Regression: skill name 'Sonarr v3/v4 API Integration'."""
        assert _sanitize_telegram_name("sonarr-v3/v4-api-integration") == "sonarr_v3v4_api_integration"

    def test_uppercase_lowercased(self):
        assert _sanitize_telegram_name("MyCommand") == "mycommand"

    def test_dots_and_special_chars_stripped(self):
        assert _sanitize_telegram_name("skill.v2@beta!") == "skillv2beta"

    def test_consecutive_underscores_collapsed(self):
        assert _sanitize_telegram_name("a---b") == "a_b"
        assert _sanitize_telegram_name("a-+-b") == "a_b"

    def test_leading_trailing_underscores_stripped(self):
        assert _sanitize_telegram_name("-leading") == "leading"
        assert _sanitize_telegram_name("trailing-") == "trailing"
        assert _sanitize_telegram_name("-both-") == "both"

    def test_digits_preserved(self):
        assert _sanitize_telegram_name("skill-24h") == "skill_24h"

    def test_empty_after_sanitization(self):
        assert _sanitize_telegram_name("+++") == ""

    def test_spaces_only_becomes_empty(self):
        assert _sanitize_telegram_name("   ") == ""

    def test_already_valid(self):
        assert _sanitize_telegram_name("valid_name_123") == "valid_name_123"


# ---------------------------------------------------------------------------
# Telegram command name clamping (32-char limit)
# ---------------------------------------------------------------------------


class TestClampTelegramNames:
    """Tests for _clamp_telegram_names() — 32-char enforcement + collision."""

    def test_short_names_unchanged(self):
        entries = [("help", "Show help"), ("status", "Show status")]
        result = _clamp_telegram_names(entries, set())
        assert result == entries

    def test_long_name_truncated(self):
        long = "a" * 40
        result = _clamp_telegram_names([(long, "desc")], set())
        assert len(result) == 1
        assert result[0][0] == "a" * _TG_NAME_LIMIT
        assert result[0][1] == "desc"

    def test_collision_with_reserved_gets_digit_suffix(self):
        # The truncated form collides with a reserved name
        prefix = "x" * _TG_NAME_LIMIT
        long_name = "x" * 40
        result = _clamp_telegram_names([(long_name, "d")], reserved={prefix})
        assert len(result) == 1
        name = result[0][0]
        assert len(name) == _TG_NAME_LIMIT
        assert name == "x" * (_TG_NAME_LIMIT - 1) + "0"

    def test_collision_between_entries_gets_incrementing_digits(self):
        # Two long names that truncate to the same 32-char prefix
        base = "y" * 40
        entries = [(base + "_alpha", "d1"), (base + "_beta", "d2")]
        result = _clamp_telegram_names(entries, set())
        assert len(result) == 2
        assert result[0][0] == "y" * _TG_NAME_LIMIT
        assert result[1][0] == "y" * (_TG_NAME_LIMIT - 1) + "0"

    def test_collision_with_reserved_and_entries_skips_taken_digits(self):
        prefix = "z" * _TG_NAME_LIMIT
        digit0 = "z" * (_TG_NAME_LIMIT - 1) + "0"
        # Reserve both the plain truncation and digit-0
        reserved = {prefix, digit0}
        long_name = "z" * 50
        result = _clamp_telegram_names([(long_name, "d")], reserved)
        assert len(result) == 1
        assert result[0][0] == "z" * (_TG_NAME_LIMIT - 1) + "1"

    def test_all_digits_exhausted_drops_entry(self):
        prefix = "w" * _TG_NAME_LIMIT
        # Reserve the plain truncation + all 10 digit slots
        reserved = {prefix} | {"w" * (_TG_NAME_LIMIT - 1) + str(d) for d in range(10)}
        long_name = "w" * 50
        result = _clamp_telegram_names([(long_name, "d")], reserved)
        assert result == []

    def test_exact_32_chars_not_truncated(self):
        name = "a" * _TG_NAME_LIMIT
        result = _clamp_telegram_names([(name, "desc")], set())
        assert result[0][0] == name

    def test_duplicate_short_name_deduplicated(self):
        entries = [("foo", "d1"), ("foo", "d2")]
        result = _clamp_telegram_names(entries, set())
        assert len(result) == 1
        assert result[0] == ("foo", "d1")


class TestTelegramMenuCommands:
    """Integration: telegram_menu_commands enforces the 32-char limit."""

    def test_all_names_within_limit(self):
        menu, _ = telegram_menu_commands(max_commands=100)
        for name, _desc in menu:
            assert 1 <= len(name) <= _TG_NAME_LIMIT, (
                f"Command '{name}' is {len(name)} chars (limit {_TG_NAME_LIMIT})"
            )

    def test_includes_plugin_commands_via_lazy_discovery(self, tmp_path, monkeypatch):
        """Telegram menu generation should discover plugin slash commands on first access."""
        from unittest.mock import patch
        import hermes_cli.plugins as plugins_mod

        plugin_dir = tmp_path / "plugins" / "cmd-plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: cmd-plugin\nversion: 0.1.0\ndescription: Test plugin\n"
        )
        (plugin_dir / "__init__.py").write_text(
            "def register(ctx):\n"
            "    ctx.register_command('lcm', lambda args: 'ok', description='LCM status and diagnostics')\n"
        )
        # Opt-in: plugins are opt-in by default, so enable in config.yaml
        (tmp_path / "config.yaml").write_text(
            "plugins:\n  enabled:\n    - cmd-plugin\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch.object(plugins_mod, "_plugin_manager", None):
            menu, _ = telegram_menu_commands(max_commands=100)

        menu_names = {name for name, _ in menu}
        assert "lcm" in menu_names

    def test_excludes_telegram_disabled_skills(self, tmp_path, monkeypatch):
        """Skills disabled for telegram should not appear in the menu."""
        from unittest.mock import patch, MagicMock

        # Set up a config with a telegram-specific disabled list
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "skills:\n"
            "  platform_disabled:\n"
            "    telegram:\n"
            "      - my-disabled-skill\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Mock get_skill_commands to return two skills
        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/my-disabled-skill": {
                "name": "my-disabled-skill",
                "description": "Should be hidden",
                "skill_md_path": f"{fake_skills_dir}/my-disabled-skill/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/my-disabled-skill",
            },
            "/my-enabled-skill": {
                "name": "my-enabled-skill",
                "description": "Should be visible",
                "skill_md_path": f"{fake_skills_dir}/my-enabled-skill/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/my-enabled-skill",
            },
        }
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            (tmp_path / "skills").mkdir(exist_ok=True)
            menu, hidden = telegram_menu_commands(max_commands=100)

        menu_names = {n for n, _ in menu}
        assert "my_enabled_skill" in menu_names
        assert "my_disabled_skill" not in menu_names

    def test_special_chars_in_skill_names_sanitized(self, tmp_path, monkeypatch):
        """Skills with +, /, or other special chars produce valid Telegram names."""
        from unittest.mock import patch
        import re

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/jellyfin-+-jellystat-24h-summary": {
                "name": "Jellyfin + Jellystat 24h Summary",
                "description": "Test",
                "skill_md_path": f"{fake_skills_dir}/jellyfin/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/jellyfin",
            },
            "/sonarr-v3/v4-api": {
                "name": "Sonarr v3/v4 API",
                "description": "Test",
                "skill_md_path": f"{fake_skills_dir}/sonarr/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/sonarr",
            },
        }
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            (tmp_path / "skills").mkdir(exist_ok=True)
            menu, _ = telegram_menu_commands(max_commands=100)

        # Every name must match Telegram's [a-z0-9_] requirement
        tg_valid = re.compile(r"^[a-z0-9_]+$")
        for name, _ in menu:
            assert tg_valid.match(name), f"Invalid Telegram command name: {name!r}"

    def test_empty_sanitized_names_excluded(self, tmp_path, monkeypatch):
        """Skills whose names sanitize to empty string are silently dropped."""
        from unittest.mock import patch

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/+++": {
                "name": "+++",
                "description": "All special chars",
                "skill_md_path": f"{fake_skills_dir}/bad/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/bad",
            },
            "/valid-skill": {
                "name": "valid-skill",
                "description": "Normal skill",
                "skill_md_path": f"{fake_skills_dir}/valid/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/valid",
            },
        }
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            (tmp_path / "skills").mkdir(exist_ok=True)
            menu, _ = telegram_menu_commands(max_commands=100)

        menu_names = {n for n, _ in menu}
        # The valid skill should be present, the empty one should not
        assert "valid_skill" in menu_names
        # No empty string in menu names
        assert "" not in menu_names


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

class TestBackwardCompatAliases:
    """The renamed constants/functions still exist under the old names."""

    def test_tg_name_limit_alias(self):
        assert _TG_NAME_LIMIT == _CMD_NAME_LIMIT == 32

    def test_clamp_telegram_names_is_clamp_command_names(self):
        assert _clamp_telegram_names is _clamp_command_names


# ---------------------------------------------------------------------------
# Discord skill command registration
# ---------------------------------------------------------------------------

class TestDiscordSkillCommands:
    """Tests for discord_skill_commands() — centralized skill registration."""

    def test_returns_skill_entries(self, tmp_path, monkeypatch):
        """Skills under SKILLS_DIR (not .hub) should be returned."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/gif-search": {
                "name": "gif-search",
                "description": "Search for GIFs",
                "skill_md_path": f"{fake_skills_dir}/gif-search/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/gif-search",
            },
            "/code-review": {
                "name": "code-review",
                "description": "Review code changes",
                "skill_md_path": f"{fake_skills_dir}/code-review/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/code-review",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "skills").mkdir(exist_ok=True)
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            entries, hidden = discord_skill_commands(
                max_slots=50, reserved_names=set(),
            )

        names = {n for n, _d, _k in entries}
        assert "gif-search" in names
        assert "code-review" in names
        assert hidden == 0
        # Verify cmd_key is preserved for handler callbacks
        keys = {k for _n, _d, k in entries}
        assert "/gif-search" in keys
        assert "/code-review" in keys

    def test_names_allow_hyphens(self, tmp_path, monkeypatch):
        """Discord names should keep hyphens (unlike Telegram's _ sanitization)."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/my-cool-skill": {
                "name": "my-cool-skill",
                "description": "A cool skill",
                "skill_md_path": f"{fake_skills_dir}/my-cool-skill/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/my-cool-skill",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "skills").mkdir(exist_ok=True)
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            entries, _ = discord_skill_commands(
                max_slots=50, reserved_names=set(),
            )

        assert entries[0][0] == "my-cool-skill"  # hyphens preserved

    def test_cap_enforcement(self, tmp_path, monkeypatch):
        """Entries beyond max_slots should be hidden."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            f"/skill-{i:03d}": {
                "name": f"skill-{i:03d}",
                "description": f"Skill {i}",
                "skill_md_path": f"{fake_skills_dir}/skill-{i:03d}/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/skill-{i:03d}",
            }
            for i in range(20)
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "skills").mkdir(exist_ok=True)
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            entries, hidden = discord_skill_commands(
                max_slots=5, reserved_names=set(),
            )

        assert len(entries) == 5
        assert hidden == 15

    def test_excludes_discord_disabled_skills(self, tmp_path, monkeypatch):
        """Skills disabled for discord should not appear."""
        from unittest.mock import patch

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "skills:\n"
            "  platform_disabled:\n"
            "    discord:\n"
            "      - secret-skill\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/secret-skill": {
                "name": "secret-skill",
                "description": "Should not appear",
                "skill_md_path": f"{fake_skills_dir}/secret-skill/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/secret-skill",
            },
            "/public-skill": {
                "name": "public-skill",
                "description": "Should appear",
                "skill_md_path": f"{fake_skills_dir}/public-skill/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/public-skill",
            },
        }
        (tmp_path / "skills").mkdir(exist_ok=True)
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            entries, _ = discord_skill_commands(
                max_slots=50, reserved_names=set(),
            )

        names = {n for n, _d, _k in entries}
        assert "secret-skill" not in names
        assert "public-skill" in names

    def test_reserved_names_not_overwritten(self, tmp_path, monkeypatch):
        """Skills whose names collide with built-in commands should be skipped."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/status": {
                "name": "status",
                "description": "Skill that collides with built-in",
                "skill_md_path": f"{fake_skills_dir}/status/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/status",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "skills").mkdir(exist_ok=True)
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            entries, _ = discord_skill_commands(
                max_slots=50, reserved_names={"status"},
            )

        names = {n for n, _d, _k in entries}
        assert "status" not in names

    def test_description_truncated_at_100_chars(self, tmp_path, monkeypatch):
        """Descriptions exceeding 100 chars should be truncated."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        long_desc = "x" * 150
        fake_cmds = {
            "/verbose-skill": {
                "name": "verbose-skill",
                "description": long_desc,
                "skill_md_path": f"{fake_skills_dir}/verbose-skill/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/verbose-skill",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "skills").mkdir(exist_ok=True)
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            entries, _ = discord_skill_commands(
                max_slots=50, reserved_names=set(),
            )

        assert len(entries[0][1]) == 100
        assert entries[0][1].endswith("...")

    def test_all_names_within_32_chars(self, tmp_path, monkeypatch):
        """All returned names must respect the 32-char Discord limit."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        long_name = "a" * 50
        fake_cmds = {
            f"/{long_name}": {
                "name": long_name,
                "description": "Long name skill",
                "skill_md_path": f"{fake_skills_dir}/{long_name}/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/{long_name}",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "skills").mkdir(exist_ok=True)
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            entries, _ = discord_skill_commands(
                max_slots=50, reserved_names=set(),
            )

        for name, _d, _k in entries:
            assert len(name) <= _CMD_NAME_LIMIT, (
                f"Name '{name}' is {len(name)} chars (limit {_CMD_NAME_LIMIT})"
            )


# ---------------------------------------------------------------------------
# Discord skill commands grouped by category
# ---------------------------------------------------------------------------

from hermes_cli.commands import discord_skill_commands_by_category  # noqa: E402


class TestDiscordSkillCommandsByCategory:
    """Tests for discord_skill_commands_by_category() — /skill group registration."""

    def test_groups_skills_by_category(self, tmp_path, monkeypatch):
        """Skills nested 2+ levels deep should be grouped by top-level category."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        # Create the directory structure so resolve() works
        for p in [
            "skills/creative/ascii-art",
            "skills/creative/excalidraw",
            "skills/media/gif-search",
        ]:
            (tmp_path / p).mkdir(parents=True, exist_ok=True)
            (tmp_path / p / "SKILL.md").write_text("---\nname: test\n---\n")

        fake_cmds = {
            "/ascii-art": {
                "name": "ascii-art",
                "description": "Generate ASCII art",
                "skill_md_path": f"{fake_skills_dir}/creative/ascii-art/SKILL.md",
            },
            "/excalidraw": {
                "name": "excalidraw",
                "description": "Hand-drawn diagrams",
                "skill_md_path": f"{fake_skills_dir}/creative/excalidraw/SKILL.md",
            },
            "/gif-search": {
                "name": "gif-search",
                "description": "Search for GIFs",
                "skill_md_path": f"{fake_skills_dir}/media/gif-search/SKILL.md",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            categories, uncategorized, hidden = discord_skill_commands_by_category(
                reserved_names=set(),
            )

        assert "creative" in categories
        assert "media" in categories
        assert len(categories["creative"]) == 2
        assert len(categories["media"]) == 1
        assert uncategorized == []
        assert hidden == 0

    def test_root_level_skills_are_uncategorized(self, tmp_path, monkeypatch):
        """Skills directly under SKILLS_DIR (only 1 path component) → uncategorized."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        (tmp_path / "skills" / "dogfood").mkdir(parents=True, exist_ok=True)
        (tmp_path / "skills" / "dogfood" / "SKILL.md").write_text("")

        fake_cmds = {
            "/dogfood": {
                "name": "dogfood",
                "description": "QA testing",
                "skill_md_path": f"{fake_skills_dir}/dogfood/SKILL.md",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            categories, uncategorized, hidden = discord_skill_commands_by_category(
                reserved_names=set(),
            )

        assert categories == {}
        assert len(uncategorized) == 1
        assert uncategorized[0][0] == "dogfood"

    def test_hub_skills_excluded(self, tmp_path, monkeypatch):
        """Skills under .hub should be excluded."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        (tmp_path / "skills" / ".hub" / "some-skill").mkdir(parents=True, exist_ok=True)
        (tmp_path / "skills" / ".hub" / "some-skill" / "SKILL.md").write_text("")

        fake_cmds = {
            "/some-skill": {
                "name": "some-skill",
                "description": "Hub skill",
                "skill_md_path": f"{fake_skills_dir}/.hub/some-skill/SKILL.md",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            categories, uncategorized, hidden = discord_skill_commands_by_category(
                reserved_names=set(),
            )

        assert categories == {}
        assert uncategorized == []

    def test_deep_nested_skills_use_top_category(self, tmp_path, monkeypatch):
        """Skills like mlops/training/axolotl should group under 'mlops'."""
        from unittest.mock import patch

        fake_skills_dir = str(tmp_path / "skills")
        (tmp_path / "skills" / "mlops" / "training" / "axolotl").mkdir(parents=True, exist_ok=True)
        (tmp_path / "skills" / "mlops" / "training" / "axolotl" / "SKILL.md").write_text("")
        (tmp_path / "skills" / "mlops" / "inference" / "vllm").mkdir(parents=True, exist_ok=True)
        (tmp_path / "skills" / "mlops" / "inference" / "vllm" / "SKILL.md").write_text("")

        fake_cmds = {
            "/axolotl": {
                "name": "axolotl",
                "description": "Fine-tuning with Axolotl",
                "skill_md_path": f"{fake_skills_dir}/mlops/training/axolotl/SKILL.md",
            },
            "/vllm": {
                "name": "vllm",
                "description": "vLLM inference",
                "skill_md_path": f"{fake_skills_dir}/mlops/inference/vllm/SKILL.md",
            },
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            categories, uncategorized, hidden = discord_skill_commands_by_category(
                reserved_names=set(),
            )

        # Both should be under 'mlops' regardless of sub-category
        assert "mlops" in categories
        names = {n for n, _d, _k in categories["mlops"]}
        assert "axolotl" in names
        assert "vllm" in names
        assert len(uncategorized) == 0
