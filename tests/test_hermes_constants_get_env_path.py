"""Tests for hermes_constants.get_env_path() — .env under HERMES_HOME."""
from pathlib import Path

from hermes_constants import get_env_path


class TestGetEnvPath:
    def test_returns_dotenv_under_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert get_env_path() == tmp_path / ".env"

    def test_uses_default_home_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        result = get_env_path()
        assert result == tmp_path / ".hermes" / ".env"

    def test_filename_is_always_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert get_env_path().name == ".env"

    def test_returns_a_path_object(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert isinstance(get_env_path(), Path)

    def test_changes_when_hermes_home_changes(self, tmp_path, monkeypatch):
        a = tmp_path / "a"
        b = tmp_path / "b"
        monkeypatch.setenv("HERMES_HOME", str(a))
        first = get_env_path()
        monkeypatch.setenv("HERMES_HOME", str(b))
        second = get_env_path()
        assert first.parent == a
        assert second.parent == b
