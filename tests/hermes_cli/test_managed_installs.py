from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.config import (
    format_managed_message,
    get_managed_system,
    recommended_update_command,
)
from hermes_cli.main import cmd_update
from tools.skills_hub import OptionalSkillSource


def test_get_managed_system_homebrew(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "homebrew")

    assert get_managed_system() == "Homebrew"
    assert recommended_update_command() == "brew upgrade hermes-agent"


def test_format_managed_message_homebrew(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "homebrew")

    message = format_managed_message("update Hermes Agent")

    assert "managed by Homebrew" in message
    assert "brew upgrade hermes-agent" in message


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


def test_optional_skill_source_honors_env_override(monkeypatch, tmp_path):
    optional_dir = tmp_path / "optional-skills"
    optional_dir.mkdir()
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(optional_dir))

    source = OptionalSkillSource()

    assert source._optional_dir == optional_dir


# ── Home Manager detection tests ──────────────────────────────────────────────


def test_get_managed_system_home_manager_env(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "home-manager")

    assert get_managed_system() == "Home Manager"
    assert recommended_update_command() == "home-manager switch"


def test_get_managed_system_home_manager_nixos_env(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "home-manager-nixos")

    assert get_managed_system() == "Home Manager (NixOS)"
    assert recommended_update_command() == "sudo nixos-rebuild switch"


def test_get_managed_system_home_manager_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".managed").write_text("home-manager")

    assert get_managed_system() == "Home Manager"


def test_get_managed_system_home_manager_nixos_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".managed").write_text("home-manager-nixos")

    assert get_managed_system() == "Home Manager (NixOS)"


def test_get_managed_system_empty_marker_still_nixos(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".managed").write_text("")

    assert get_managed_system() == "NixOS"


def test_get_managed_system_nixos_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".managed").write_text("nixos")

    assert get_managed_system() == "NixOS"


def test_format_managed_message_home_manager(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "home-manager")

    message = format_managed_message("update Hermes Agent")

    assert "managed by Home Manager" in message
    assert "home-manager switch" in message
    assert "programs.hermes-agent.settings" in message


def test_format_managed_message_home_manager_nixos(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "home-manager-nixos")

    message = format_managed_message("update Hermes Agent")

    assert "managed by Home Manager" in message
    assert "via NixOS module" in message
    assert "sudo nixos-rebuild switch" in message
    assert "programs.hermes-agent.settings" in message


def test_format_managed_message_nixos_unchanged(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "true")

    message = format_managed_message("update Hermes Agent")

    assert "managed by NixOS" in message
    assert "sudo nixos-rebuild switch" in message
    assert "services.hermes-agent.settings" in message


def test_cmd_update_blocks_managed_home_manager(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_MANAGED", "home-manager")

    with patch("hermes_cli.main.subprocess.run") as mock_run:
        cmd_update(SimpleNamespace())

    assert not mock_run.called
    captured = capsys.readouterr()
    assert "managed by Home Manager" in captured.err
    assert "home-manager switch" in captured.err
