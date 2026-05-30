import os
from unittest.mock import patch


def test_default_config_raises_main_and_delegation_iteration_budgets():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["max_turns"] == 240
    assert DEFAULT_CONFIG["delegation"]["max_iterations"] == 180


def test_load_cli_config_uses_raised_iteration_budgets_when_no_user_override(tmp_path):
    import cli as cli_mod
    from cli import load_cli_config

    with (
        patch.object(cli_mod, "_hermes_home", tmp_path),
        patch.dict(os.environ, {"LLM_MODEL": ""}, clear=False),
    ):
        config = load_cli_config()

    assert config["agent"]["max_turns"] == 240
    assert config["delegation"]["max_iterations"] == 180
