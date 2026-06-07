from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.context_cmd import (
    ONE_M_CONTEXT_LENGTH,
    get_model_context_length_override,
    parse_context_length_action,
    run_context_config_command,
    set_model_context_length_override,
)
from hermes_cli.config import read_raw_config


def test_parse_context_length_action_accepts_1m_and_auto():
    assert parse_context_length_action("1m") == ("set", ONE_M_CONTEXT_LENGTH)
    assert parse_context_length_action("on") == ("set", ONE_M_CONTEXT_LENGTH)
    assert parse_context_length_action("auto") == ("auto", None)
    assert parse_context_length_action("status") == ("status", None)


def test_set_context_length_upgrades_bare_model_string():
    config = {"model": "test/model"}

    set_model_context_length_override(config, ONE_M_CONTEXT_LENGTH)

    assert config["model"]["default"] == "test/model"
    assert config["model"]["context_length"] == ONE_M_CONTEXT_LENGTH
    assert get_model_context_length_override(config) == ONE_M_CONTEXT_LENGTH


def test_clear_context_length_does_not_rewrite_bare_model_string():
    config = {"model": "test/model"}

    set_model_context_length_override(config, None)

    assert config["model"] == "test/model"


def test_run_context_command_saves_and_clears_override(tmp_path):
    with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}):
        set_msg = run_context_config_command("1m")
        raw = read_raw_config()

        assert "1,000,000" in set_msg
        assert raw["model"]["context_length"] == ONE_M_CONTEXT_LENGTH

        status_msg = run_context_config_command("status")
        assert "1,000,000" in status_msg

        clear_msg = run_context_config_command("auto")
        raw = read_raw_config()

        assert "auto-detect" in clear_msg
        assert "context_length" not in raw["model"]


def test_run_context_command_updates_live_agent(tmp_path):
    compressor = SimpleNamespace(update_model=MagicMock())
    agent = SimpleNamespace(
        model="test-model",
        base_url="",
        api_key="",
        provider="openrouter",
        api_mode="",
        context_compressor=compressor,
    )

    with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}):
        msg = run_context_config_command("1m", agent=agent)

    assert "Current session context window is now 1,000,000 tokens" in msg
    assert agent._config_context_length == ONE_M_CONTEXT_LENGTH
    compressor.update_model.assert_called_once_with(
        "test-model",
        ONE_M_CONTEXT_LENGTH,
        base_url="",
        api_key="",
        provider="openrouter",
        api_mode="",
    )
