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



    def test_ambiguous_prefix_shows_suggestions(self):
        """/re matches multiple commands — should show ambiguous message."""
        cli_obj = _make_cli()
        with patch("cli._cprint") as mock_cprint:
            cli_obj.process_command("/re")
            printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Ambiguous" in printed or "Did you mean" in printed



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

    def test_prefix_matching_ignores_blocked_bundles_for_selection(self):
        """/research-p should not select a blocked bundle under protected governance."""
        cli_obj = _make_cli()
        printed = []
        import cli as cli_mod

        with (
            patch.object(cli_mod, "_skill_commands", {}),
            patch.object(cli_mod, "get_skill_bundles", return_value={"/research-pack": {"name": "Research Pack"}}),
            patch.object(cli_mod, "get_discoverable_skill_bundles", return_value={}),
            patch.object(cli_mod, "_cprint", side_effect=lambda text: printed.append(str(text))),
        ):
            cli_obj.process_command("/research-p")

        combined = " ".join(printed)
        assert "Unknown command" in combined
        assert "research-pack" not in combined

    def test_exact_blocked_bundle_is_not_selected_for_invocation(self):
        """Typing the full blocked bundle name must not select or load it."""
        cli_obj = _make_cli()
        import cli as cli_mod
        printed = []

        with (
            patch.object(cli_mod, "_skill_commands", {}),
            patch.object(cli_mod, "get_skill_bundles", return_value={"/research-pack": {"name": "Research Pack"}}),
            patch.object(cli_mod, "get_discoverable_skill_bundles", return_value={}),
            patch.object(cli_mod, "_cprint", side_effect=lambda text: printed.append(str(text))),
            patch.object(cli_mod, "build_bundle_invocation_message") as mock_build,
        ):
            cli_obj.process_command("/research-pack investigate")

        combined = " ".join(printed)
        assert "Unknown command" in combined
        mock_build.assert_not_called()
        cli_obj._pending_input.put.assert_not_called()

    def test_prefix_matching_preserves_unprotected_bundle_behavior(self):
        """/research-p should still expand to a visible bundle when unprotected."""
        cli_obj = _make_cli()
        import cli as cli_mod

        with (
            patch.object(cli_mod, "_skill_commands", {}),
            patch.object(cli_mod, "get_skill_bundles", return_value={"/research-pack": {"name": "Research Pack"}}),
            patch.object(cli_mod, "get_discoverable_skill_bundles", return_value={"/research-pack": {"name": "Research Pack"}}),
            patch.object(
                cli_mod,
                "build_bundle_invocation_message",
                return_value=("bundle payload", ["alpha"], []),
            ) as mock_build,
            patch("builtins.print"),
        ):
            cli_obj.process_command("/research-p investigate")

        mock_build.assert_called_once_with(
            "/research-pack",
            "investigate",
            task_id=None,
        )

    def test_exact_unprotected_bundle_invocation_still_loads(self):
        """Typing the full visible bundle name should still load it."""
        cli_obj = _make_cli()
        import cli as cli_mod

        with (
            patch.object(cli_mod, "_skill_commands", {}),
            patch.object(cli_mod, "get_skill_bundles", return_value={"/research-pack": {"name": "Research Pack"}}),
            patch.object(cli_mod, "get_discoverable_skill_bundles", return_value={"/research-pack": {"name": "Research Pack"}}),
            patch.object(
                cli_mod,
                "build_bundle_invocation_message",
                return_value=("bundle payload", ["alpha"], []),
            ) as mock_build,
            patch("builtins.print"),
        ):
            cli_obj.process_command("/research-pack investigate")

        mock_build.assert_called_once_with(
            "/research-pack",
            "investigate",
            task_id=None,
        )
        cli_obj._pending_input.put.assert_called_once_with("bundle payload")
