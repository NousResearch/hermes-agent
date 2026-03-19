"""Tests for Morph terminal backend CLI/config surfaces."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hermes_cli.config import get_env_path, load_config
from hermes_cli.setup import setup_terminal_backend
from hermes_cli.status import show_status


@pytest.fixture()
def isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    return tmp_path


def test_setup_terminal_backend_morph_updates_config_and_env(
    isolated_hermes_home, monkeypatch
):
    monkeypatch.setitem(sys.modules, "morphcloud", ModuleType("morphcloud"))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.delenv("MORPH_API_KEY", raising=False)
    monkeypatch.delenv("TERMINAL_MORPH_IMAGE_ID", raising=False)

    config = load_config()

    def _pick_morph(_question, choices, _default):
        return next(
            index for index, choice in enumerate(choices) if choice.startswith("Morph -")
        )

    prompt_values = iter(
        [
            "morph-api-key",
            "morphvm-minimal",
            "yes",
            "2",
            "6144",
            "20480",
        ]
    )

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", _pick_morph)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(prompt_values))

    setup_terminal_backend(config)

    env_text = get_env_path().read_text(encoding="utf-8")
    assert config["terminal"]["backend"] == "morph"
    assert config["terminal"]["morph_image_id"] == "morphvm-minimal"
    assert config["terminal"]["container_cpu"] == 2.0
    assert config["terminal"]["container_memory"] == 6144
    assert config["terminal"]["container_disk"] == 20480
    assert "MORPH_API_KEY=morph-api-key" in env_text
    assert "TERMINAL_MORPH_IMAGE_ID=morphvm-minimal" in env_text
    assert "TERMINAL_ENV=morph" in env_text


def test_show_status_includes_morph_backend_details(monkeypatch, capsys):
    monkeypatch.setenv("TERMINAL_ENV", "morph")
    monkeypatch.setenv("MORPH_API_KEY", "morph-key")
    monkeypatch.setenv("TERMINAL_MORPH_IMAGE_ID", "morphvm-minimal")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    show_status(SimpleNamespace(all=False, deep=False))
    output = capsys.readouterr().out

    assert "Backend:      morph" in output
    assert "Morph Image: morphvm-minimal" in output
    assert "Morph API Key:" in output


def test_show_status_uses_default_morph_image_when_env_missing(
    isolated_hermes_home, monkeypatch, capsys
):
    monkeypatch.setenv("TERMINAL_ENV", "morph")
    monkeypatch.setenv("MORPH_API_KEY", "morph-key")
    monkeypatch.delenv("TERMINAL_MORPH_IMAGE_ID", raising=False)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    show_status(SimpleNamespace(all=False, deep=False))
    output = capsys.readouterr().out

    assert "Morph Image: morphvm-minimal" in output


def test_doctor_reports_morph_configuration(monkeypatch, isolated_hermes_home, capsys):
    monkeypatch.setenv("TERMINAL_ENV", "morph")
    monkeypatch.setenv("MORPH_API_KEY", "morph-key")
    monkeypatch.setenv("TERMINAL_MORPH_IMAGE_ID", "morphvm-minimal")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "morphcloud", ModuleType("morphcloud"))
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    import hermes_cli.doctor as doctor_module

    doctor_module = importlib.reload(doctor_module)
    doctor_module.run_doctor(SimpleNamespace(fix=False))
    output = capsys.readouterr().out

    assert "Morph API key" in output
    assert "Morph base image" in output
    assert "morphcloud SDK" in output


def test_doctor_uses_default_morph_image_when_env_missing(
    monkeypatch, isolated_hermes_home, capsys
):
    monkeypatch.setenv("TERMINAL_ENV", "morph")
    monkeypatch.setenv("MORPH_API_KEY", "morph-key")
    monkeypatch.delenv("TERMINAL_MORPH_IMAGE_ID", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "morphcloud", ModuleType("morphcloud"))
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    import hermes_cli.doctor as doctor_module

    doctor_module = importlib.reload(doctor_module)
    doctor_module.run_doctor(SimpleNamespace(fix=False))
    output = capsys.readouterr().out

    assert "Morph base image" in output
    assert "morphvm-minimal" in output
