from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from hermes_cli import setup as setup_cli


def _first_time_args() -> Namespace:
    return Namespace(
        section=None,
        migrate_from=None,
        non_interactive=False,
        reset=False,
        skip_migration_prompt=False,
    )


def test_run_setup_wizard_runs_explicit_openclaw_migration(tmp_path: Path):
    args = Namespace(
        section=None,
        migrate_from="openclaw",
        non_interactive=False,
        reset=False,
        skip_migration_prompt=False,
    )

    with (
        patch.object(setup_cli, "ensure_hermes_home"),
        patch.object(setup_cli, "load_config", return_value={}),
        patch.object(setup_cli, "get_hermes_home", return_value=tmp_path),
        patch.object(setup_cli, "get_env_value", return_value=""),
        patch("hermes_cli.auth.get_active_provider", return_value=None),
        patch(
            "hermes_cli.setup.perform_openclaw_migration",
            return_value={"summary": {"migrated": 1}},
            create=True,
        ) as run_migration,
        patch.object(setup_cli, "setup_model_provider"),
        patch.object(setup_cli, "setup_terminal_backend"),
        patch.object(setup_cli, "setup_agent_settings"),
        patch.object(setup_cli, "setup_gateway"),
        patch.object(setup_cli, "setup_tools"),
        patch.object(setup_cli, "save_config"),
        patch.object(setup_cli, "_print_setup_summary"),
        patch("builtins.input", return_value=""),
    ):
        setup_cli.run_setup_wizard(args)

    run_migration.assert_called_once()


def test_run_setup_wizard_offers_detected_openclaw_migration_on_first_run(
    tmp_path: Path,
):
    args = _first_time_args()

    with (
        patch.object(setup_cli, "ensure_hermes_home"),
        patch.object(setup_cli, "load_config", return_value={}),
        patch.object(setup_cli, "get_hermes_home", return_value=tmp_path),
        patch.object(setup_cli, "get_env_value", return_value=""),
        patch("hermes_cli.auth.get_active_provider", return_value=None),
        patch(
            "hermes_cli.setup.detect_migration_sources",
            return_value=["openclaw"],
            create=True,
        ),
        patch.object(setup_cli, "prompt_yes_no", return_value=True) as prompt_yes_no,
        patch(
            "hermes_cli.setup.perform_openclaw_migration",
            return_value={"summary": {"migrated": 2}},
            create=True,
        ) as run_migration,
        patch.object(setup_cli, "setup_model_provider"),
        patch.object(setup_cli, "setup_terminal_backend"),
        patch.object(setup_cli, "setup_agent_settings"),
        patch.object(setup_cli, "setup_gateway"),
        patch.object(setup_cli, "setup_tools"),
        patch.object(setup_cli, "save_config"),
        patch.object(setup_cli, "_print_setup_summary"),
        patch("builtins.input", return_value=""),
    ):
        setup_cli.run_setup_wizard(args)

    prompt_yes_no.assert_any_call("Import settings from OpenClaw now?", True)
    run_migration.assert_called_once()


def test_run_setup_wizard_skips_migration_prompt_when_no_supported_source(
    tmp_path: Path,
):
    args = _first_time_args()

    with (
        patch.object(setup_cli, "ensure_hermes_home"),
        patch.object(setup_cli, "load_config", return_value={}),
        patch.object(setup_cli, "get_hermes_home", return_value=tmp_path),
        patch.object(setup_cli, "get_env_value", return_value=""),
        patch("hermes_cli.auth.get_active_provider", return_value=None),
        patch(
            "hermes_cli.setup.detect_migration_sources", return_value=[], create=True
        ),
        patch.object(setup_cli, "prompt_yes_no") as prompt_yes_no,
        patch("hermes_cli.setup.perform_openclaw_migration", create=True) as run_migration,
        patch.object(setup_cli, "setup_model_provider"),
        patch.object(setup_cli, "setup_terminal_backend"),
        patch.object(setup_cli, "setup_agent_settings"),
        patch.object(setup_cli, "setup_gateway"),
        patch.object(setup_cli, "setup_tools"),
        patch.object(setup_cli, "save_config"),
        patch.object(setup_cli, "_print_setup_summary"),
        patch("builtins.input", return_value=""),
    ):
        setup_cli.run_setup_wizard(args)

    prompt_yes_no.assert_not_called()
    run_migration.assert_not_called()


def test_run_setup_wizard_runs_explicit_openclaw_migration_for_existing_install(
    tmp_path: Path,
):
    args = Namespace(
        section=None,
        migrate_from="openclaw",
        non_interactive=False,
        reset=False,
        skip_migration_prompt=False,
    )

    with (
        patch.object(setup_cli, "ensure_hermes_home"),
        patch.object(setup_cli, "load_config", return_value={}),
        patch.object(setup_cli, "get_hermes_home", return_value=tmp_path),
        patch.object(
            setup_cli,
            "get_env_value",
            side_effect=lambda key: "configured" if key == "OPENROUTER_API_KEY" else "",
        ),
        patch("hermes_cli.auth.get_active_provider", return_value=None),
        patch(
            "hermes_cli.setup.perform_openclaw_migration",
            return_value={"summary": {"migrated": 1}},
        ) as run_migration,
        patch.object(setup_cli, "prompt_choice", return_value=9),
    ):
        setup_cli.run_setup_wizard(args)

    run_migration.assert_called_once_with(tmp_path)


def test_run_setup_wizard_skips_detected_prompt_after_installer_decline(
    tmp_path: Path,
):
    args = Namespace(
        section=None,
        migrate_from=None,
        non_interactive=False,
        reset=False,
        skip_migration_prompt=True,
    )

    with (
        patch.object(setup_cli, "ensure_hermes_home"),
        patch.object(setup_cli, "load_config", return_value={}),
        patch.object(setup_cli, "get_hermes_home", return_value=tmp_path),
        patch.object(setup_cli, "get_env_value", return_value=""),
        patch("hermes_cli.auth.get_active_provider", return_value=None),
        patch(
            "hermes_cli.setup.detect_migration_sources",
            return_value=["openclaw"],
        ),
        patch.object(setup_cli, "prompt_yes_no") as prompt_yes_no,
        patch("hermes_cli.setup.perform_openclaw_migration") as run_migration,
        patch.object(setup_cli, "setup_model_provider"),
        patch.object(setup_cli, "setup_terminal_backend"),
        patch.object(setup_cli, "setup_agent_settings"),
        patch.object(setup_cli, "setup_gateway"),
        patch.object(setup_cli, "setup_tools"),
        patch.object(setup_cli, "save_config"),
        patch.object(setup_cli, "_print_setup_summary"),
        patch("builtins.input", return_value=""),
    ):
        setup_cli.run_setup_wizard(args)

    prompt_yes_no.assert_not_called()
    run_migration.assert_not_called()


def test_perform_openclaw_migration_creates_config_before_import(tmp_path: Path):
    source_root = tmp_path / ".openclaw"
    source_root.mkdir()
    hermes_home = tmp_path / ".hermes"
    config_path = hermes_home / "config.yaml"

    with (
        patch(
            "hermes_cli.openclaw_migration.get_openclaw_source_root",
            return_value=source_root,
        ),
        patch.object(setup_cli, "get_config_path", return_value=config_path),
        patch.object(
            setup_cli,
            "load_config",
            return_value={"agent": {"max_turns": 90}},
        ),
        patch.object(setup_cli, "save_config") as save_config,
        patch(
            "hermes_cli.openclaw_migration.run_openclaw_migration",
            return_value={"summary": {"migrated": 1}, "output_dir": None},
        ) as run_migration,
    ):
        setup_cli.perform_openclaw_migration(hermes_home)

    save_config.assert_called_once_with({"agent": {"max_turns": 90}})
    run_migration.assert_called_once()
