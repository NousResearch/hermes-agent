"""Tests for the CLI /routing command."""

from unittest.mock import patch

import cli as cli_mod


def _make_cli(enabled=True):
    obj = object.__new__(cli_mod.HermesCLI)
    obj._smart_model_routing = {
        "enabled": enabled,
        "max_simple_chars": 160,
        "max_simple_words": 28,
        "cheap_model": {
            "provider": "nous",
            "model": "google/gemini-3-flash-preview",
        },
    }
    obj._smart_model_routing_override_enabled = None
    return obj


def _printed_text(mock_cprint):
    return "\n".join(str(call.args[0]) for call in mock_cprint.call_args_list if call.args)


def test_routing_status_shows_global_and_session_state():
    cli_obj = _make_cli(enabled=True)

    with patch("cli._cprint") as mock_cprint:
        cli_obj._handle_routing_command("/routing")

    printed = _printed_text(mock_cprint)
    assert "Smart routing (global): ON" in printed
    assert "Smart routing (session):" in printed
    assert "inherit global" in printed
    assert "google/gemini-3-flash-preview via nous" in printed


def test_routing_off_sets_session_override_only():
    cli_obj = _make_cli(enabled=True)

    with patch("cli._cprint"):
        cli_obj._handle_routing_command("/routing off")

    assert cli_obj._smart_model_routing_override_enabled is False
    assert cli_obj._smart_model_routing["enabled"] is True


def test_routing_default_clears_session_override():
    cli_obj = _make_cli(enabled=True)
    cli_obj._smart_model_routing_override_enabled = False

    with patch("cli._cprint"):
        cli_obj._handle_routing_command("/routing default")

    assert cli_obj._smart_model_routing_override_enabled is None


def test_routing_global_updates_config_and_clears_override():
    cli_obj = _make_cli(enabled=True)
    cli_obj._smart_model_routing_override_enabled = False

    with patch("cli._cprint"), patch("cli.save_config_value", return_value=True) as mock_save:
        cli_obj._handle_routing_command("/routing off --global")

    assert cli_obj._smart_model_routing_override_enabled is None
    assert cli_obj._smart_model_routing["enabled"] is False
    mock_save.assert_called_once_with("smart_model_routing.enabled", False)


def test_routing_global_save_failure_falls_back_to_session_override():
    cli_obj = _make_cli(enabled=True)

    with patch("cli._cprint"), patch("cli.save_config_value", return_value=False) as mock_save:
        cli_obj._handle_routing_command("/routing off --global")

    assert cli_obj._smart_model_routing["enabled"] is True
    assert cli_obj._smart_model_routing_override_enabled is False
    mock_save.assert_called_once_with("smart_model_routing.enabled", False)
