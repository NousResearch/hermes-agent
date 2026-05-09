"""Tests for hermes_constants.display_hermes_home() — user-facing path string."""
from pathlib import Path

from hermes_constants import display_hermes_home


class TestDisplayHermesHome:
    def test_uses_tilde_shorthand_for_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_HOME", raising=False)
        result = display_hermes_home()
        assert result.startswith("~/")
        assert ".hermes" in result

    def test_uses_tilde_for_path_under_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "custom-hermes"))
        assert display_hermes_home() == "~/custom-hermes"

    def test_returns_absolute_when_outside_home(self, tmp_path, monkeypatch):
        outside = tmp_path / "elsewhere" / "hermes"
        outside.mkdir(parents=True)
        home = tmp_path / "user-home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("HERMES_HOME", str(outside))
        result = display_hermes_home()
        assert result == str(outside)
        assert "~" not in result

    def test_returns_string_type(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert isinstance(display_hermes_home(), str)

    def test_profile_path_under_home_uses_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        profile = tmp_path / ".hermes" / "profiles" / "coder"
        monkeypatch.setenv("HERMES_HOME", str(profile))
        result = display_hermes_home()
        assert result.startswith("~/.hermes/profiles/")
        assert result.endswith("coder")
