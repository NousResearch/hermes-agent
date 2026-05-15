"""Integration tests: ``agent_home_scope`` actually isolates profile data.

These tests don't spin up a full AIAgent (that has dozens of
dependencies that would bloat the test).  Instead, they exercise the
profile-aware path functions that AIAgent uses internally — memory,
skills, soul — and verify each one resolves to the *active agent's*
home directory when wrapped in ``agent_home_scope``.

If a future refactor introduces a path resolver that bypasses
``get_hermes_home()``, these tests will fail — that's the design
intent.  ``get_hermes_home()`` is the single chokepoint that makes the
multi-agent gateway possible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.agent_context import agent_home_scope, reset_agent_home
from hermes_constants import (
    get_hermes_home,
    get_config_path,
    get_env_path,
    get_skills_dir,
)


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Lay out two fully-populated agent homes side by side."""
    root = tmp_path / ".hermes"
    coder = root / "profiles" / "coder"
    sci = root / "profiles" / "data-sci"
    for home in (root, coder, sci):
        home.mkdir(parents=True, exist_ok=True)
        (home / "memories").mkdir()
        (home / "skills").mkdir()
        (home / "SOUL.md").write_text(f"# Soul of {home.name}\n", encoding="utf-8")
        (home / "memories" / "MEMORY.md").write_text(
            f"memory_for_{home.name}\n", encoding="utf-8"
        )
        (home / "config.yaml").write_text(
            f"# config for {home.name}\n", encoding="utf-8"
        )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    reset_agent_home()
    yield root, coder, sci
    reset_agent_home()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPathFunctionsRespectScope:
    def test_get_hermes_home_default(self, two_profiles):
        root, _coder, _sci = two_profiles
        assert get_hermes_home() == root

    def test_get_hermes_home_inside_scope(self, two_profiles):
        _root, coder, _sci = two_profiles
        with agent_home_scope(coder):
            assert get_hermes_home() == coder

    def test_get_config_path_inside_scope(self, two_profiles):
        _root, coder, _sci = two_profiles
        with agent_home_scope(coder):
            assert get_config_path() == coder / "config.yaml"

    def test_get_env_path_inside_scope(self, two_profiles):
        _root, coder, _sci = two_profiles
        with agent_home_scope(coder):
            assert get_env_path() == coder / ".env"

    def test_get_skills_dir_inside_scope(self, two_profiles):
        _root, coder, _sci = two_profiles
        with agent_home_scope(coder):
            assert get_skills_dir() == coder / "skills"

    def test_switching_between_scopes(self, two_profiles):
        _root, coder, sci = two_profiles
        with agent_home_scope(coder):
            assert get_hermes_home() == coder
            with agent_home_scope(sci):
                assert get_hermes_home() == sci
            assert get_hermes_home() == coder


# ---------------------------------------------------------------------------
# Memory isolation
# ---------------------------------------------------------------------------


class TestMemoryIsolation:
    """tools.memory_tool reads ``get_hermes_home() / 'memories'``."""

    def test_memory_dir_inside_scope(self, two_profiles):
        _root, coder, sci = two_profiles
        from tools.memory_tool import get_memory_dir

        with agent_home_scope(coder):
            assert get_memory_dir() == coder / "memories"
        with agent_home_scope(sci):
            assert get_memory_dir() == sci / "memories"

    def test_memory_content_isolation(self, two_profiles):
        """Two agents see distinct MEMORY.md content."""
        _root, coder, sci = two_profiles
        from tools.memory_tool import get_memory_dir

        with agent_home_scope(coder):
            coder_mem = (get_memory_dir() / "MEMORY.md").read_text(encoding="utf-8")
        with agent_home_scope(sci):
            sci_mem = (get_memory_dir() / "MEMORY.md").read_text(encoding="utf-8")
        assert "coder" in coder_mem
        assert "data-sci" in sci_mem
        assert coder_mem != sci_mem


# ---------------------------------------------------------------------------
# Soul isolation
# ---------------------------------------------------------------------------


class TestSoulIsolation:
    """The agent's SOUL.md is read from get_hermes_home() / 'SOUL.md'.

    See ``agent/prompt_builder.py``.  We do the read directly rather
    than invoking the full prompt builder so the test stays focused
    on the path resolution.
    """

    def _read_soul(self) -> str:
        return (get_hermes_home() / "SOUL.md").read_text(encoding="utf-8")

    def test_soul_inside_scope(self, two_profiles):
        _root, coder, sci = two_profiles
        with agent_home_scope(coder):
            assert "coder" in self._read_soul()
        with agent_home_scope(sci):
            assert "data-sci" in self._read_soul()


# ---------------------------------------------------------------------------
# Skills isolation
# ---------------------------------------------------------------------------


class TestSkillsIsolation:
    """tools/skill_usage.py and similar callers read ``get_hermes_home() / 'skills'``."""

    def test_skills_dir_inside_scope(self, two_profiles):
        _root, coder, sci = two_profiles
        with agent_home_scope(coder):
            assert get_skills_dir() == coder / "skills"
        with agent_home_scope(sci):
            assert get_skills_dir() == sci / "skills"

    def test_skills_content_isolation(self, two_profiles):
        """Each profile sees only its own installed skills."""
        _root, coder, sci = two_profiles
        # Plant a SKILL.md in coder but NOT in data-sci
        (coder / "skills" / "coder-only").mkdir()
        (coder / "skills" / "coder-only" / "SKILL.md").write_text(
            "---\nname: coder-only\n---\n", encoding="utf-8"
        )

        with agent_home_scope(coder):
            coder_skills = sorted(p.name for p in get_skills_dir().iterdir())
        with agent_home_scope(sci):
            sci_skills = sorted(p.name for p in get_skills_dir().iterdir())

        assert "coder-only" in coder_skills
        assert "coder-only" not in sci_skills


# ---------------------------------------------------------------------------
# Negative: env var alone (no contextvar) still resolves to the env-var path
# so single-profile gateways behave exactly as before.
# ---------------------------------------------------------------------------


class TestEnvVarBackwardCompat:
    def test_env_var_path_when_no_scope(self, two_profiles, monkeypatch):
        _root, coder, _sci = two_profiles
        monkeypatch.setenv("HERMES_HOME", str(coder))
        assert get_hermes_home() == coder

    def test_scope_wins_over_env_var(self, two_profiles, monkeypatch):
        _root, coder, sci = two_profiles
        monkeypatch.setenv("HERMES_HOME", str(coder))
        with agent_home_scope(sci):
            assert get_hermes_home() == sci
        # Restored to env-var path
        assert get_hermes_home() == coder
