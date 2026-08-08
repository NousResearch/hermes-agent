"""Tests for the Python dependency input-hash cache in ``hermes update``."""

from __future__ import annotations

import pytest

from hermes_cli import update_cmd
from hermes_constants import venv_python_path


def _setup_project(tmp_path, *, pyproject: str = "", uv_lock: str = "") -> None:
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(uv_lock, encoding="utf-8")


def _setup_venv(tmp_path, monkeypatch) -> None:
    venv_python = venv_python_path(tmp_path / "venv", windows=False)
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(update_cmd._m(), "_is_windows", lambda: False)


@pytest.fixture
def _healthy_venv(monkeypatch):
    """Patch the venv health probe to report a healthy install."""
    monkeypatch.setattr(
        update_cmd, "_venv_core_imports_healthy", lambda: (True, "")
    )


def test_python_dependencies_changed_true_when_no_cache(
    monkeypatch, tmp_path, _healthy_venv
):
    hermes_root = tmp_path / "hermes_home"
    hermes_root.mkdir()
    _setup_project(tmp_path, pyproject='[project]\nname = "hermes"\n')
    _setup_venv(tmp_path, monkeypatch)
    monkeypatch.setattr(update_cmd, "get_default_hermes_root", lambda: hermes_root)

    assert update_cmd._python_dependencies_changed(hermes_root, "all") is True


def test_record_then_skip_on_unchanged_inputs(
    monkeypatch, tmp_path, _healthy_venv
):
    hermes_root = tmp_path / "hermes_home"
    hermes_root.mkdir()
    _setup_project(tmp_path, pyproject='[project]\nname = "hermes"\n')
    _setup_venv(tmp_path, monkeypatch)
    monkeypatch.setattr(update_cmd, "get_default_hermes_root", lambda: hermes_root)

    update_cmd._record_python_dependencies_hash(hermes_root, "all")
    assert update_cmd._python_dependencies_changed(hermes_root, "all") is False


def test_python_dependencies_changed_true_when_inputs_change(
    monkeypatch, tmp_path, _healthy_venv
):
    hermes_root = tmp_path / "hermes_home"
    hermes_root.mkdir()
    _setup_project(tmp_path, pyproject='[project]\nname = "hermes"\n')
    _setup_venv(tmp_path, monkeypatch)
    monkeypatch.setattr(update_cmd, "get_default_hermes_root", lambda: hermes_root)

    update_cmd._record_python_dependencies_hash(hermes_root, "all")
    assert update_cmd._python_dependencies_changed(hermes_root, "all") is False

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "hermes"\nversion = "0.20.1"\n', encoding="utf-8"
    )
    assert update_cmd._python_dependencies_changed(hermes_root, "all") is True


def test_python_dependencies_changed_true_when_unhealthy(
    monkeypatch, tmp_path
):
    hermes_root = tmp_path / "hermes_home"
    hermes_root.mkdir()
    _setup_project(tmp_path, pyproject='[project]\nname = "hermes"\n')
    _setup_venv(tmp_path, monkeypatch)
    monkeypatch.setattr(update_cmd, "get_default_hermes_root", lambda: hermes_root)
    monkeypatch.setattr(update_cmd, "_venv_core_imports_healthy", lambda: (False, "broken"))

    update_cmd._record_python_dependencies_hash(hermes_root, "all")
    assert update_cmd._python_dependencies_changed(hermes_root, "all") is True


def test_python_dependencies_changed_true_when_no_pyproject(
    monkeypatch, tmp_path, _healthy_venv
):
    hermes_root = tmp_path / "hermes_home"
    hermes_root.mkdir()
    _setup_venv(tmp_path, monkeypatch)
    monkeypatch.setattr(update_cmd, "get_default_hermes_root", lambda: hermes_root)

    assert update_cmd._python_dependencies_changed(hermes_root, "all") is True
