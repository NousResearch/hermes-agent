"""Tests for agent/skill_utils.py — skill config resolution and path expansion."""

import os

from agent.skill_utils import resolve_skill_config_values


class TestResolveSkillConfigValuesExpandUser:
    """Skill config ~/... expansion should use subprocess HOME, not Python process HOME.

    Hermes maintains a separate per-profile HOME ({HERMES_HOME}/home/) for
    terminal and background subprocesses.  Skill config defaults expressed as
    ~/... paths (e.g. ~/wiki) are meant for the subprocess environment, so the
    expansion must honour get_subprocess_home() when that directory exists.
    """

    def test_tilde_path_expands_to_subprocess_home_when_available(self, tmp_path, monkeypatch):
        """When {HERMES_HOME}/home/ exists, ~/wiki must resolve inside it."""
        # conftest._hermetic_environment already creates tmp_path/hermes_test and sets
        # HERMES_HOME to it.  We only need to add the "home/" subdirectory.
        hermes_home = tmp_path / "hermes_test"
        (hermes_home / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_vars = [
            {
                "key": "wiki.path",
                "description": "Path to wiki",
                "default": "~/wiki",
            },
        ]

        result = resolve_skill_config_values(config_vars)

        expected = str(hermes_home / "home" / "wiki")
        assert result["wiki.path"] == expected

    def test_bare_tilde_expands_to_subprocess_home_when_available(self, tmp_path, monkeypatch):
        """When {HERMES_HOME}/home/ exists, ~ alone must resolve to it."""
        hermes_home = tmp_path / "hermes_test"
        (hermes_home / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_vars = [
            {
                "key": "base.path",
                "description": "Base directory",
                "default": "~",
            },
        ]

        result = resolve_skill_config_values(config_vars)

        assert result["base.path"] == str(hermes_home / "home")

    def test_no_subprocess_home_falls_back_to_expanduser(self, tmp_path, monkeypatch):
        """When {HERMES_HOME}/home/ does NOT exist, fall back to os.path.expanduser."""
        hermes_home = tmp_path / "hermes_test"
        # No "home" subdirectory — get_subprocess_home() returns None.
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        original_expanduser = os.path.expanduser

        def mock_expanduser(path):
            if path == "~/wiki":
                return "/real/home/wiki"
            return original_expanduser(path)

        monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

        config_vars = [
            {
                "key": "wiki.path",
                "description": "Path to wiki",
                "default": "~/wiki",
            },
        ]

        result = resolve_skill_config_values(config_vars)

        assert result["wiki.path"] == "/real/home/wiki"

    def test_env_var_expansion_still_works(self, tmp_path, monkeypatch):
        """${VARIABLE} references are still expanded regardless of subprocess HOME."""
        hermes_home = tmp_path / "hermes_test"
        (hermes_home / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("MY_WIKI_ROOT", "/custom/wiki")

        config_vars = [
            {
                "key": "wiki.path",
                "description": "Path to wiki",
                "default": "${MY_WIKI_ROOT}/notes",
            },
        ]

        result = resolve_skill_config_values(config_vars)

        assert result["wiki.path"] == "/custom/wiki/notes"

    def test_non_tilde_value_unchanged(self, tmp_path, monkeypatch):
        """Values without ~ or ${ are returned as-is."""
        hermes_home = tmp_path / "hermes_test"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_vars = [
            {
                "key": "wiki.path",
                "description": "Path to wiki",
                "default": "/absolute/path/to/wiki",
            },
        ]

        result = resolve_skill_config_values(config_vars)

        assert result["wiki.path"] == "/absolute/path/to/wiki"

    def test_stored_config_value_overrides_default(self, tmp_path, monkeypatch):
        """When a value is already set in config.yaml, default is not used."""
        hermes_home = tmp_path / "hermes_test"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_vars = [
            {
                "key": "wiki.path",
                "description": "Path to wiki",
                "default": "~/wiki",
            },
        ]

        result = resolve_skill_config_values(config_vars)

        # Default should be expanded because no value is stored in config.yaml.
        assert "wiki.path" in result
