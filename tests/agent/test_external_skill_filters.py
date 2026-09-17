"""Tests for ``skills.external_include`` / ``skills.external_exclude`` (prompt-index narrowing).

The filter applies to the ``skills.external_dirs`` tier only: the profile's own ``skills/``
directory is never filtered, and an empty ``include`` keeps every external skill — the
behaviour every pre-existing config already had.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from agent.prompt_builder import (
    _get_external_skill_filters,
    _skill_matches_filter,
    build_skills_system_prompt,
    clear_skills_system_prompt_cache,
)
from agent.skill_utils import _external_dirs_cache_clear, get_external_skills_dirs


def _skill(root: Path, rel: str, name: str, description: str = "d") -> None:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n",
                                encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    _external_dirs_cache_clear()
    yield
    clear_skills_system_prompt_cache(clear_snapshot=True)
    _external_dirs_cache_clear()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated profile home + external skills tree + a local profile skill."""
    home = tmp_path / ".hermes"
    (home / "skills" / "local-tool").mkdir(parents=True)
    (home / "skills" / "local-tool" / "SKILL.md").write_text(
        "---\nname: local-tool\ndescription: Profile-local skill\n---\n\nBody.\n", encoding="utf-8")

    ext = tmp_path / "ext-skills"
    _skill(ext, "workflow/release-notes", "release-notes")
    _skill(ext, "workflow/deep/nested/deep-skill", "deep-skill")
    _skill(ext, "workflow/vendor/vendor-tool", "vendor-tool")
    _skill(ext, "devops/docker-ops", "docker-ops")
    _skill(ext, "github", "gh-skill")

    monkeypatch.setenv("HERMES_HOME", str(home))

    def write_config(extra: str = "") -> Path:
        cfg = home / "config.yaml"
        cfg.write_text(f"skills:\n  external_dirs:\n    - {ext}\n{extra}", encoding="utf-8")
        return cfg

    return {"home": home, "ext": ext, "write_config": write_config}


def _index(env) -> str:
    assert get_external_skills_dirs(), "external dir must resolve, else the test proves nothing"
    return build_skills_system_prompt()


class TestSkillMatchesFilter:
    def test_empty_include_keeps_everything(self):
        assert _skill_matches_filter("workflow/release-notes", "release-notes", [], []) is True

    def test_include_selects_and_drops(self):
        inc = ["workflow/*"]
        assert _skill_matches_filter("workflow/release-notes", "release-notes", inc, []) is True
        assert _skill_matches_filter("devops/docker-ops", "docker-ops", inc, []) is False

    def test_exclude_wins_over_include(self):
        assert _skill_matches_filter("workflow/vendor/vendor-tool", "vendor-tool",
                                     ["workflow/*"], ["workflow/vendor/*"]) is False

    def test_exclude_alone_narrows_nothing_else(self):
        assert _skill_matches_filter("devops/docker-ops", "docker-ops", [], ["workflow/*"]) is True
        assert _skill_matches_filter("workflow/release-notes", "release-notes", [], ["workflow/*"]) is False

    def test_path_prefix_keeps_nested_skills(self):
        assert _skill_matches_filter("workflow/deep/nested/deep-skill", "deep-skill", ["workflow/*"], []) is True

    def test_bare_pattern_matches_a_top_level_category(self):
        assert _skill_matches_filter("devops/docker-ops", "docker-ops", ["devops"], []) is True
        assert _skill_matches_filter("workflow/release-notes", "release-notes", ["devops"], []) is False

    def test_pattern_matches_the_skill_name(self):
        assert _skill_matches_filter("workflow/release-notes", "release-notes", ["release-*"], []) is True

    def test_windows_separators_are_normalised(self):
        assert _skill_matches_filter("workflow" + os.sep + "release-notes", "release-notes", ["workflow/*"], []) is True


class TestGetExternalSkillFilters:
    def test_absent_keys_are_empty(self, env):
        env["write_config"]()
        assert _get_external_skill_filters() == ([], [])

    def test_reads_lists_and_drops_blanks(self, env):
        env["write_config"]("  external_include: ['workflow/*', '  ']\n  external_exclude: []\n")
        assert _get_external_skill_filters() == (["workflow/*"], [])

    def test_scalar_or_junk_fails_open(self, env):
        env["write_config"]("  external_include: 'workflow/*'\n  external_exclude: 7\n")
        assert _get_external_skill_filters() == ([], [])


class TestExternalFiltersInIndex:
    def test_no_filters_keeps_current_behaviour(self, env):
        env["write_config"]()
        result = _index(env)
        for name in ("release-notes", "deep-skill", "vendor-tool", "docker-ops", "gh-skill", "local-tool"):
            assert name in result

    def test_include_narrows_to_the_pattern(self, env):
        env["write_config"]("  external_include: ['workflow/*']\n")
        result = _index(env)
        assert "release-notes" in result
        assert "deep-skill" in result          # nested under the kept category
        assert "vendor-tool" in result         # include-only: still present
        assert "docker-ops" not in result
        assert "gh-skill" not in result

    def test_exclude_removes_a_kept_subtree(self, env):
        env["write_config"]("  external_include: ['workflow/*']\n  external_exclude: ['workflow/vendor/*']\n")
        result = _index(env)
        assert "release-notes" in result
        assert "vendor-tool" not in result

    def test_profile_skills_are_never_filtered(self, env):
        env["write_config"]("  external_include: ['github']\n")
        result = _index(env)
        assert "local-tool" in result
        assert "gh-skill" in result
        assert "release-notes" not in result

    def test_distinct_patterns_do_not_share_a_cache_entry(self, env):
        """Several profiles share one skills root and differ only by these patterns: the prompt
        cache key must separate them, or the first build's index gets served to the next one."""
        env["write_config"]("  external_include: ['workflow/*']\n")
        first = _index(env)
        assert "release-notes" in first and "docker-ops" not in first

        config = env["write_config"]("  external_include: ['devops/*']\n")  # same dirs, new patterns
        st = config.stat()
        os.utime(config, (st.st_atime + 10, st.st_atime + 10))

        second = build_skills_system_prompt()  # no cache clearing: the key itself must differ
        assert "docker-ops" in second
        assert "release-notes" not in second


def test_missing_external_dir_logs_a_warning(tmp_path, monkeypatch, caplog):
    """A glob or a typo in external_dirs is no longer a silent no-op."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {tmp_path / 'nope-*'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    _external_dirs_cache_clear()
    with caplog.at_level(logging.WARNING, logger="agent.skill_utils"):
        assert get_external_skills_dirs() == []
    assert any(r.levelno == logging.WARNING and "does not exist" in r.message for r in caplog.records)
