"""Canonical command identity and metadata contracts."""
from commands import COMMAND_REGISTRY, CommandDef, GATEWAY_KNOWN_COMMANDS, infer_argument_mode, resolve_command

class TestCommandRegistry:


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


    def test_argument_mode_infers_text_from_any_args_hint(self):
        assert infer_argument_mode(CommandDef("demo", "Demo", "Session", args_hint="<prompt>")) == "text"
        assert infer_argument_mode(CommandDef("ask", "Ask", "Session", args_hint="<query>")) == "text"
        assert infer_argument_mode(CommandDef("note", "Note", "Session", args_hint="[message]")) == "text"


class TestResolveCommandAliases:
    """One-letter aliases resolve to their command, never a longer canonical
    (exact lookup treats the alias as a full name — /s is not a /sessions prefix)."""

    def test_q_resolves_to_queue(self):
        cmd = resolve_command("q")
        assert cmd is not None and cmd.name == "queue"

    def test_s_resolves_to_steer(self):
        cmd = resolve_command("s")
        assert cmd is not None and cmd.name == "steer"

    def test_exact_names_still_win_over_the_alias(self):
        cmd = resolve_command("sessions")
        assert cmd is not None and cmd.name == "sessions"
        cmd = resolve_command("steer")
        assert cmd is not None and cmd.name == "steer"


class TestGatewayKnownCommands:

    def test_includes_config_gated_cli_only(self):
        """Commands with gateway_config_gate are always in GATEWAY_KNOWN_COMMANDS."""
        for cmd in COMMAND_REGISTRY:
            if cmd.gateway_config_gate:
                assert cmd.name in GATEWAY_KNOWN_COMMANDS, \
                    f"config-gated command '{cmd.name}' should be in GATEWAY_KNOWN_COMMANDS"
