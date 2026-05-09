"""Tests for hermes_constants.get_config_path() — config.yaml under HERMES_HOME."""
from pathlib import Path

from hermes_constants import get_config_path


class TestGetConfigPath:
    def test_returns_path_under_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = get_config_path()
        assert result == tmp_path / "config.yaml"

    def test_uses_default_home_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        result = get_config_path()
        assert result == tmp_path / ".hermes" / "config.yaml"

    def test_filename_is_always_config_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert get_config_path().name == "config.yaml"

    def test_returns_a_path_object(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert isinstance(get_config_path(), Path)

    def test_changes_when_hermes_home_changes(self, tmp_path, monkeypatch):
        a = tmp_path / "a"
        b = tmp_path / "b"
        monkeypatch.setenv("HERMES_HOME", str(a))
        first = get_config_path()
        monkeypatch.setenv("HERMES_HOME", str(b))
        second = get_config_path()
        assert first != second
        assert first.parent == a
        assert second.parent == b
