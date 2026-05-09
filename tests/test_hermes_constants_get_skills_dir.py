"""Tests for hermes_constants.get_skills_dir() — skills/ under HERMES_HOME."""
from pathlib import Path

from hermes_constants import get_skills_dir


class TestGetSkillsDir:
    def test_returns_skills_under_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert get_skills_dir() == tmp_path / "skills"

    def test_uses_default_home_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        result = get_skills_dir()
        assert result.name == "skills"
        assert result.parent.name == ".hermes"

    def test_directory_name_is_always_skills(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert get_skills_dir().name == "skills"

    def test_returns_a_path_object(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert isinstance(get_skills_dir(), Path)

    def test_changes_when_hermes_home_changes(self, tmp_path, monkeypatch):
        a = tmp_path / "a"
        b = tmp_path / "b"
        monkeypatch.setenv("HERMES_HOME", str(a))
        first = get_skills_dir()
        monkeypatch.setenv("HERMES_HOME", str(b))
        second = get_skills_dir()
        assert first.parent == a
        assert second.parent == b
