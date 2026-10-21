"""Tests for hermes_cli/setup_terminal.py backend wizards."""

import pytest

from hermes_cli.config import save_env_value, get_env_value
from hermes_cli import setup as setup_mod
from hermes_cli import setup_terminal


@pytest.fixture
def ssh_wizard(tmp_path, monkeypatch):
    """Drive ``_setup_backend_ssh`` with scripted answers, SSH test declined."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TERMINAL_SSH_PORT", raising=False)

    def run(answers):
        it = iter(answers)
        monkeypatch.setattr(setup_mod, "prompt", lambda label, default="": next(it))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda label, default=True: False)
        setup_terminal._setup_backend_ssh({})

    return run


def test_ssh_port_reset_to_22_removes_saved_port(ssh_wizard):
    # A stored non-default port must be removable: answering "22" restores the
    # default, it does not silently leave the old value in .env.
    save_env_value("TERMINAL_SSH_PORT", "2222")
    ssh_wizard(["host.example.com", "me", "22", "/tmp/key"])
    assert get_env_value("TERMINAL_SSH_PORT") is None


def test_ssh_port_non_default_still_saves(ssh_wizard):
    # Control: the deliberate skip is only for the default; a real port persists.
    ssh_wizard(["host.example.com", "me", "2222", "/tmp/key"])
    assert get_env_value("TERMINAL_SSH_PORT") == "2222"


def test_docker_wizard_reports_the_resolved_podman_runtime(monkeypatch, capsys):
    """Podman satisfies the docker backend, so the wizard must name it instead of warning."""
    monkeypatch.setattr(setup_terminal, "find_docker", lambda: "/usr/bin/podman")
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda label, default=True: False)

    setup_terminal._setup_backend_docker({"terminal": {}})

    out = capsys.readouterr().out
    assert "Podman found: /usr/bin/podman" in out
    assert "not found in PATH" not in out
