"""Tests for OpenClaw migration integration in the setup wizard."""

from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock, patch

from hermes_cli import setup as setup_mod


# ---------------------------------------------------------------------------
# _offer_openclaw_migration — unit tests
# ---------------------------------------------------------------------------


class TestOfferOpenclawMigration:
    """Test the _offer_openclaw_migration helper in isolation."""

    def test_skips_when_no_openclaw_dir(self, tmp_path):
        """Should return False immediately when ~/.openclaw does not exist."""
        with patch("hermes_cli.setup.Path.home", return_value=tmp_path):
            assert setup_mod._offer_openclaw_migration(tmp_path / ".hermes") is False

    def test_skips_when_migration_script_missing(self, tmp_path):
        """Should return False when the migration script file is absent."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", tmp_path / "nonexistent.py"),
        ):
            assert setup_mod._offer_openclaw_migration(tmp_path / ".hermes") is False

    def test_skips_when_user_declines(self, tmp_path):
        """Should return False when user declines the migration prompt."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")
        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            patch.object(setup_mod, "prompt_yes_no", return_value=False),
        ):
            assert setup_mod._offer_openclaw_migration(tmp_path / ".hermes") is False

    def test_runs_migration_when_user_accepts(self, tmp_path):
        """Should dynamically load the script and run the Migrator."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()

        # Create a fake hermes home with config
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  max_turns: 90\n")

        # Build a fake migration module
        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value={"soul", "memory"})
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 3, "skipped": 1, "conflict": 0, "error": 0},
            "output_dir": str(hermes_home / "migration"),
        }
        fake_mod.Migrator = MagicMock(return_value=fake_migrator)

        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")

        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            patch.object(setup_mod, "prompt_yes_no", return_value=True),
            patch.object(setup_mod, "get_config_path", return_value=config_path),
            patch("importlib.util.spec_from_file_location") as mock_spec_fn,
        ):
            # Wire up the fake module loading
            mock_spec = MagicMock()
            mock_spec.loader = MagicMock()
            mock_spec_fn.return_value = mock_spec

            def exec_module(mod):
                mod.resolve_selected_options = fake_mod.resolve_selected_options
                mod.Migrator = fake_mod.Migrator

            mock_spec.loader.exec_module = exec_module

            result = setup_mod._offer_openclaw_migration(hermes_home)

        assert result is True
        fake_mod.resolve_selected_options.assert_called_once_with(
            None, None, preset="full"
        )
        fake_mod.Migrator.assert_called_once()
        call_kwargs = fake_mod.Migrator.call_args[1]
        assert call_kwargs["execute"] is True
        assert call_kwargs["overwrite"] is False
        assert call_kwargs["migrate_secrets"] is True
        assert call_kwargs["preset_name"] == "full"
        fake_migrator.migrate.assert_called_once()

    def test_handles_migration_error_gracefully(self, tmp_path):
        """Should catch exceptions and return False."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("")

        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")

        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            patch.object(setup_mod, "prompt_yes_no", return_value=True),
            patch.object(setup_mod, "get_config_path", return_value=config_path),
            patch(
                "importlib.util.spec_from_file_location",
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = setup_mod._offer_openclaw_migration(hermes_home)

        assert result is False

    def test_creates_config_if_missing(self, tmp_path):
        """Should bootstrap config.yaml before running migration."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        # config does NOT exist yet

        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")

        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            patch.object(setup_mod, "prompt_yes_no", return_value=True),
            patch.object(setup_mod, "get_config_path", return_value=config_path),
            patch.object(setup_mod, "load_config", return_value={"agent": {}}),
            patch.object(setup_mod, "save_config") as mock_save,
            patch(
                "importlib.util.spec_from_file_location",
                side_effect=RuntimeError("stop early"),
            ),
        ):
            setup_mod._offer_openclaw_migration(hermes_home)

        # save_config should have been called to bootstrap the file
        mock_save.assert_called_once_with({"agent": {}})


# ---------------------------------------------------------------------------
# Integration with run_setup_wizard — first-time flow
# ---------------------------------------------------------------------------


def _first_time_args() -> Namespace:
    return Namespace(
        section=None,
        non_interactive=False,
        reset=False,
    )


class TestSetupWizardOpenclawIntegration:
    """Verify _offer_openclaw_migration is called during first-time setup."""

    def test_migration_offered_during_first_time_setup(self, tmp_path):
        """On first-time setup, _offer_openclaw_migration should be called."""
        args = _first_time_args()

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value={}),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "get_env_value", return_value=""),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            # User presses Enter to start
            patch("builtins.input", return_value=""),
            # Mock the migration offer
            patch.object(
                setup_mod, "_offer_openclaw_migration", return_value=False
            ) as mock_migration,
            # Mock the actual setup sections so they don't run
            patch.object(setup_mod, "setup_model_provider"),
            patch.object(setup_mod, "setup_terminal_backend"),
            patch.object(setup_mod, "setup_agent_settings"),
            patch.object(setup_mod, "setup_gateway"),
            patch.object(setup_mod, "setup_tools"),
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        mock_migration.assert_called_once_with(tmp_path)

    def test_migration_reloads_config_on_success(self, tmp_path):
        """When migration returns True, config should be reloaded."""
        args = _first_time_args()
        call_order = []

        def tracking_load_config():
            call_order.append("load_config")
            return {}

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", side_effect=tracking_load_config),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "get_env_value", return_value=""),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch("builtins.input", return_value=""),
            patch.object(setup_mod, "_offer_openclaw_migration", return_value=True),
            patch.object(setup_mod, "setup_model_provider"),
            patch.object(setup_mod, "setup_terminal_backend"),
            patch.object(setup_mod, "setup_agent_settings"),
            patch.object(setup_mod, "setup_gateway"),
            patch.object(setup_mod, "setup_tools"),
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        # load_config called twice: once at start, once after migration
        assert call_order.count("load_config") == 2

    def test_reloaded_config_flows_into_remaining_setup_sections(self, tmp_path):
        args = _first_time_args()
        initial_config = {}
        reloaded_config = {"model": {"provider": "openrouter"}}

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(
                setup_mod,
                "load_config",
                side_effect=[initial_config, reloaded_config],
            ),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "get_env_value", return_value=""),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch("builtins.input", return_value=""),
            patch.object(setup_mod, "_offer_openclaw_migration", return_value=True),
            patch.object(setup_mod, "setup_model_provider") as setup_model_provider,
            patch.object(setup_mod, "setup_terminal_backend"),
            patch.object(setup_mod, "setup_agent_settings"),
            patch.object(setup_mod, "setup_gateway"),
            patch.object(setup_mod, "setup_tools"),
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        setup_model_provider.assert_called_once_with(reloaded_config)

    def test_migration_not_offered_for_existing_install(self, tmp_path):
        """Returning users should not see the migration prompt."""
        args = _first_time_args()

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value={}),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(
                setup_mod,
                "get_env_value",
                side_effect=lambda k: "sk-xxx" if k == "OPENROUTER_API_KEY" else "",
            ),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            # Returning user picks "Exit"
            patch.object(setup_mod, "prompt_choice", return_value=9),
            patch.object(
                setup_mod, "_offer_openclaw_migration", return_value=False
            ) as mock_migration,
        ):
            setup_mod.run_setup_wizard(args)

        mock_migration.assert_not_called()
