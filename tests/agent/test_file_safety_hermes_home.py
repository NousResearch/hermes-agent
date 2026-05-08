"""Tests that file_safety respects HERMES_HOME environment variable."""

import pytest

from agent.file_safety import _hermes_home_path, build_write_denied_paths, get_read_block_error


def test_hermes_home_path_respects_hermes_home(tmp_path, monkeypatch):
    """_hermes_home_path() should use HERMES_HOME when set."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    result = _hermes_home_path()
    assert str(result).startswith(str(tmp_path))


def test_build_write_denied_paths_respects_hermes_home(tmp_path, monkeypatch):
    """build_write_denied_paths should include paths under HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    home = str(tmp_path)
    paths = build_write_denied_paths(home)
    hermes_env_path = str(tmp_path / ".hermes" / ".env")
    assert any(p == hermes_env_path for p in paths)


def test_get_read_block_error_respects_hermes_home(tmp_path, monkeypatch):
    """get_read_block_error should block paths under HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    blocked_path = str(tmp_path / ".hermes" / "skills" / ".hub" / "index-cache" / "test.json")
    error = get_read_block_error(blocked_path)
    assert error is not None
    assert "Access denied" in error