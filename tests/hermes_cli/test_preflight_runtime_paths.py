"""Tests for _preflight_runtime_paths() in hermes_cli.main."""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest


def _import_preflight():
    from hermes_cli.main import _preflight_runtime_paths

    return _preflight_runtime_paths


def test_all_paths_writable(tmp_path, monkeypatch):
    """No warnings when all runtime paths are writable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _preflight = _import_preflight()
    warnings = _preflight()
    assert warnings == []


def test_log_dir_not_writable(tmp_path, monkeypatch):
    """Warning when logs/ directory is not writable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove write permission
    log_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    try:
        _preflight = _import_preflight()
        warnings = _preflight()
        assert len(warnings) == 1
        assert "log directory" in warnings[0]
        assert str(log_dir) in warnings[0]
        assert "PermissionError" in warnings[0]
    finally:
        # Restore permissions so tmp_path cleanup works
        log_dir.chmod(stat.S_IRWXU)


def test_history_file_not_writable(tmp_path, monkeypatch):
    """Warning when .hermes_history file is not writable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    history_file = tmp_path / ".hermes_history"
    history_file.write_text("old content")

    # Remove write permission from the file
    history_file.chmod(stat.S_IRUSR)

    try:
        _preflight = _import_preflight()
        warnings = _preflight()
        assert len(warnings) == 1
        assert "history file" in warnings[0]
    finally:
        history_file.chmod(stat.S_IRWXU)


def test_pastes_dir_not_writable(tmp_path, monkeypatch):
    """Warning when pastes/ directory is not writable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pastes_dir = tmp_path / "pastes"
    pastes_dir.mkdir(parents=True, exist_ok=True)

    # Remove write permission
    pastes_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    try:
        _preflight = _import_preflight()
        warnings = _preflight()
        assert len(warnings) == 1
        assert "pastes directory" in warnings[0]
    finally:
        pastes_dir.chmod(stat.S_IRWXU)


def test_multiple_paths_not_writable(tmp_path, monkeypatch):
    """Multiple warnings when multiple paths are not writable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pastes_dir = tmp_path / "pastes"
    pastes_dir.mkdir(parents=True, exist_ok=True)

    log_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    pastes_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    try:
        _preflight = _import_preflight()
        warnings = _preflight()
        assert len(warnings) == 2
        labels = [w for w in warnings]
        assert any("log directory" in w for w in labels)
        assert any("pastes directory" in w for w in labels)
    finally:
        log_dir.chmod(stat.S_IRWXU)
        pastes_dir.chmod(stat.S_IRWXU)


def test_history_file_does_not_exist_is_ok(tmp_path, monkeypatch):
    """No warning when .hermes_history does not exist (will be created on demand)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert not (tmp_path / ".hermes_history").exists()

    _preflight = _import_preflight()
    warnings = _preflight()
    assert warnings == []


def test_home_resolution_failure_is_silent(monkeypatch):
    """Returns empty list when get_hermes_home() fails."""
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: (_ for _ in ()).throw(RuntimeError("no home")),
    )
    _preflight = _import_preflight()
    warnings = _preflight()
    assert warnings == []


def test_fix_hint_includes_chown(tmp_path, monkeypatch):
    """Warning message includes a sudo chown fix hint."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    try:
        _preflight = _import_preflight()
        warnings = _preflight()
        assert len(warnings) == 1
        assert "chown" in warnings[0]
    finally:
        log_dir.chmod(stat.S_IRWXU)
