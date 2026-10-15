"""Tests for the trusted-project skill allowlist (<root>/.hermes/skill-allowlist)."""

from pathlib import Path

import pytest

import agent.skill_utils as su
from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache


def _skill(root: Path, rel: str, name: str) -> None:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name} procedure\n---\nbody\n")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """HERMES_HOME with three local skills + a git repo that has one project skill."""
    home = tmp_path / ".hermes"
    for name in ("pytest-flow", "vitest-flow", "audio-mixing"):
        _skill(home / "skills", f"dev/{name}", name)
    config = home / "config.yaml"
    config.write_text("skills:\n  external_dirs: []\n")
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    _skill(repo / ".hermes" / "skills", "repo-only", "repo-only")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(repo)
    su._external_dirs_cache_clear()
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield {"config": config, "repo": repo}
    su._external_dirs_cache_clear()
    clear_skills_system_prompt_cache(clear_snapshot=True)


def _trust(env) -> None:
    env["config"].write_text(f"skills:\n  external_dirs: []\n  trusted_project_dirs: ['{env['repo']}']\n")
    su._external_dirs_cache_clear()
    clear_skills_system_prompt_cache(clear_snapshot=True)


def _write_allowlist(env, text: str) -> None:
    (env["repo"] / ".hermes" / "skill-allowlist").write_text(text)
    clear_skills_system_prompt_cache(clear_snapshot=True)


def test_no_allowlist_lists_every_skill(env):
    _trust(env)
    assert su.get_project_skill_allowlist() is None
    index = build_skills_system_prompt()
    for name in ("pytest-flow", "vitest-flow", "audio-mixing", "repo-only"):
        assert name in index


def test_allowlist_narrows_index_and_keeps_project_skills(env):
    _trust(env)
    _write_allowlist(env, "# skills this repo needs\nvitest-flow\n\n")
    assert su.get_project_skill_allowlist() == frozenset({"vitest-flow"} | su.ESSENTIAL_SKILLS)
    index = build_skills_system_prompt()
    assert "vitest-flow" in index
    assert "repo-only" in index  # the project's own skills are never filtered out
    assert "pytest-flow" not in index
    assert "audio-mixing" not in index


def test_untrusted_repo_allowlist_is_ignored(env):
    _write_allowlist(env, "vitest-flow\n")
    assert su.get_project_skill_allowlist() is None
    index = build_skills_system_prompt()
    assert "pytest-flow" in index and "audio-mixing" in index


@pytest.mark.parametrize("prefix", ["", "\ufeff"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_allowlist_preserves_first_skill_with_optional_bom(env, prefix, newline):
    _trust(env)
    _write_allowlist(env, prefix + newline.join(["vitest-flow", "pytest-flow", ""]))
    assert su.get_project_skill_allowlist() == frozenset(
        {"vitest-flow", "pytest-flow"} | set(su.ESSENTIAL_SKILLS)
    )
    prompt = build_skills_system_prompt()
    assert "vitest-flow" in prompt
    assert "pytest-flow" in prompt
    assert "audio-mixing" not in prompt


@pytest.mark.parametrize("content", ["# only comments\n\n", "\ufeff", "\ufeff# only comments\r\n\r\n"])
def test_empty_allowlist_means_no_narrowing(env, content):
    _trust(env)
    _write_allowlist(env, content)
    assert su.get_project_skill_allowlist() is None
    assert "audio-mixing" in build_skills_system_prompt()


def test_allowlist_change_invalidates_cached_index(env):
    _trust(env)
    _write_allowlist(env, "vitest-flow\n")
    assert "pytest-flow" not in build_skills_system_prompt()
    (env["repo"] / ".hermes" / "skill-allowlist").write_text("pytest-flow\n")
    # No explicit cache clear: the allowlist is part of the cache key.
    assert "pytest-flow" in build_skills_system_prompt()
