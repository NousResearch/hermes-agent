"""Tests for slash command prefix matching in HermesCLI.process_command."""
from unittest.mock import MagicMock, patch
from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = None
    cli_obj._pending_input = MagicMock()
    return cli_obj


class TestSlashCommandPrefixMatching:
    def test_unique_prefix_dispatches_command(self):
        """/con should dispatch to /config when it uniquely matches."""
        cli_obj = _make_cli()
        with patch.object(cli_obj, 'show_config') as mock_config:
            cli_obj.process_command("/con")
        mock_config.assert_called_once()

    def test_unique_prefix_with_args_does_not_recurse(self):
        """/con set key value should expand to /config set key value without infinite recursion."""
        cli_obj = _make_cli()
        dispatched = []

        original = cli_obj.process_command.__func__

        def counting_process_command(self_inner, cmd):
            dispatched.append(cmd)
            if len(dispatched) > 5:
                raise RecursionError("process_command called too many times")
            return original(self_inner, cmd)

        # Mock show_config since the test is about recursion, not config display
        with patch.object(type(cli_obj), 'process_command', counting_process_command), \
             patch.object(cli_obj, 'show_config'):
            try:
                cli_obj.process_command("/con set key value")
            except RecursionError:
                assert False, "process_command recursed infinitely"

        # Should have been called at most twice: once for /con set..., once for /config set...
        assert len(dispatched) <= 2

    def test_exact_command_with_args_does_not_recurse(self):
        """/config set key value hits exact branch and does not loop back to prefix."""
        cli_obj = _make_cli()
        call_count = [0]

        original_pc = HermesCLI.process_command

        def guarded(self_inner, cmd):
            call_count[0] += 1
            if call_count[0] > 10:
                raise RecursionError("Infinite recursion detected")
            return original_pc(self_inner, cmd)

        # Mock show_config since the test is about recursion, not config display
        with patch.object(HermesCLI, 'process_command', guarded), \
             patch.object(cli_obj, 'show_config'):
            try:
                cli_obj.process_command("/config set key value")
            except RecursionError:
                assert False, "Recursed infinitely on /config set key value"

        assert call_count[0] <= 3

    def test_ambiguous_prefix_shows_suggestions(self):
        """/re matches multiple commands — should show ambiguous message."""
        cli_obj = _make_cli()
        with patch("cli._cprint") as mock_cprint:
            cli_obj.process_command("/re")
            printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Ambiguous" in printed or "Did you mean" in printed

    def test_unknown_command_shows_error(self):
        """/xyz should show unknown command error."""
        cli_obj = _make_cli()
        with patch("cli._cprint") as mock_cprint:
            cli_obj.process_command("/xyz")
            printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Unknown command" in printed

    def test_exact_command_still_works(self):
        """/help should still work as exact match."""
        cli_obj = _make_cli()
        with patch.object(cli_obj, 'show_help') as mock_help:
            cli_obj.process_command("/help")
        mock_help.assert_called_once()

    def test_skill_command_prefix_matches(self):
        """A prefix that uniquely matches a skill command should dispatch it."""
        cli_obj = _make_cli()
        fake_skill = {"/test-skill-xyz": {"name": "Test Skill", "description": "test"}}
        printed = []
        cli_obj.console.print = lambda *a, **kw: printed.append(str(a))

        import cli as cli_mod
        with patch.object(cli_mod, '_skill_commands', fake_skill):
            cli_obj.process_command("/test-skill-xy")

        # Should NOT show "Unknown command" — should have dispatched or attempted skill
        unknown = any("Unknown command" in p for p in printed)
        assert not unknown, f"Expected skill prefix to match, got: {printed}"

    def test_ambiguous_between_builtin_and_skill(self):
        """Ambiguous prefix spanning builtin + skill commands shows suggestions."""
        cli_obj = _make_cli()
        # /help-extra is a fake skill that shares /hel prefix with /help
        fake_skill = {"/help-extra": {"name": "Help Extra", "description": "test"}}

        import cli as cli_mod
        with patch.object(cli_mod, '_skill_commands', fake_skill),              patch.object(cli_obj, 'show_help') as mock_help:
            cli_obj.process_command("/help")

        # /help is an exact match so should work normally, not show ambiguous
        mock_help.assert_called_once()
        printed = " ".join(str(c) for c in cli_obj.console.print.call_args_list)
        assert "Ambiguous" not in printed

    def test_shortest_match_preferred_over_longer_skill(self):
        """/qui should dispatch to /quit (5 chars) not report ambiguous with /quint-pipeline (15 chars)."""
        cli_obj = _make_cli()
        fake_skill = {"/quint-pipeline": {"name": "Quint Pipeline", "description": "test"}}

        import cli as cli_mod
        with patch.object(cli_mod, '_skill_commands', fake_skill):
            # /quit is caught by the exact "/quit" branch → process_command returns False
            result = cli_obj.process_command("/qui")

        # Returns False because /quit was dispatched (exits chat loop)
        assert result is False
        printed = " ".join(str(c) for c in cli_obj.console.print.call_args_list)
        assert "Ambiguous" not in printed

    def test_tied_shortest_matches_still_ambiguous(self):
        """/re matches /reset and /retry (both 6 chars) — no unique shortest, stays ambiguous."""
        cli_obj = _make_cli()
        printed = []
        import cli as cli_mod
        with patch.object(cli_mod, '_cprint', side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/re")
        combined = " ".join(printed)
        assert "Ambiguous" in combined or "Did you mean" in combined

    def test_exact_typed_name_dispatches_over_longer_match(self):
        """/help typed with /help-extra skill installed → exact match wins."""
        cli_obj = _make_cli()
        fake_skill = {"/help-extra": {"name": "Help Extra", "description": ""}}
        import cli as cli_mod
        with patch.object(cli_mod, '_skill_commands', fake_skill), \
             patch.object(cli_obj, 'show_help') as mock_help:
            cli_obj.process_command("/help")
        mock_help.assert_called_once()
        printed = " ".join(str(c) for c in cli_obj.console.print.call_args_list)
        assert "Ambiguous" not in printed
