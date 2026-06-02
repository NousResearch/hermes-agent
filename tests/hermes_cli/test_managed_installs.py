from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.config import (
    format_managed_message,
    is_config_managed,
    get_managed_system,
    recommended_update_command,
)
from hermes_cli.main import cmd_postinstall, cmd_uninstall, cmd_update
from tools.skills_hub import OptionalSkillSource


def test_get_managed_system_homebrew(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "homebrew")

    assert get_managed_system() == "Homebrew"
    assert is_config_managed() is False
    assert recommended_update_command() == "brew upgrade hermes-agent"


def test_get_managed_system_nixos_is_config_managed(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "nixos")

    assert get_managed_system() == "NixOS"
    assert is_config_managed() is True


def test_get_managed_system_snap(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "snap")

    assert get_managed_system() == "Snap"
    assert is_config_managed() is False
    assert recommended_update_command() == "snap refresh hermes-agent"


def test_format_managed_message_homebrew(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "homebrew")

    message = format_managed_message("update Hermes Agent")

    assert "managed by Homebrew" in message
    assert "brew upgrade hermes-agent" in message


def test_format_managed_message_snap(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "snap")

    message = format_managed_message("update Hermes Agent")

    assert "managed by Snap" in message
    assert "snap refresh hermes-agent" in message


def test_recommended_update_command_defaults_to_hermes_update(monkeypatch):
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    # Also short-circuit the .managed marker path — CI runners may have an
    # ambient ~/.hermes/.managed if a prior test left HERMES_HOME pointing
    # somewhere with that marker, which would make get_managed_update_command()
    # return "Update your Nix flake input ..." instead of falling through to
    # detect_install_method().
    with patch("hermes_cli.config.get_managed_update_command", return_value=None), \
         patch("hermes_cli.config.detect_install_method", return_value="git"):
        assert recommended_update_command() == "hermes update"


def test_cmd_update_blocks_managed_homebrew(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_MANAGED", "homebrew")

    with patch("hermes_cli.main.subprocess.run") as mock_run:
        cmd_update(SimpleNamespace())

    assert not mock_run.called
    captured = capsys.readouterr()
    assert "managed by Homebrew" in captured.err
    assert "brew upgrade hermes-agent" in captured.err


def test_cmd_update_blocks_managed_snap(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_MANAGED", "snap")

    with patch("hermes_cli.main.subprocess.run") as mock_run:
        cmd_update(SimpleNamespace())

    assert not mock_run.called
    captured = capsys.readouterr()
    assert "managed by Snap" in captured.err
    assert "snap refresh hermes-agent" in captured.err


def test_cmd_postinstall_snap_prints_guidance(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_MANAGED", "snap")

    with patch("hermes_cli.main.cmd_setup") as mock_setup:
        cmd_postinstall(SimpleNamespace())

    assert not mock_setup.called
    captured = capsys.readouterr()
    assert "installed as a Snap" in captured.out
    assert "hermes setup" in captured.out
    assert "snap refresh hermes-agent" in captured.out


def test_cmd_uninstall_snap_prints_guidance(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_MANAGED", "snap")

    with patch("hermes_cli.main._require_tty"), \
         patch("hermes_cli.uninstall.run_uninstall") as mock_uninstall:
        cmd_uninstall(SimpleNamespace())

    assert not mock_uninstall.called
    captured = capsys.readouterr()
    assert "installed as a Snap" in captured.out
    assert "snap remove hermes-agent" in captured.out


def test_optional_skill_source_honors_env_override(monkeypatch, tmp_path):
    optional_dir = tmp_path / "optional-skills"
    optional_dir.mkdir()
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(optional_dir))

    source = OptionalSkillSource()

    assert source._optional_dir == optional_dir
