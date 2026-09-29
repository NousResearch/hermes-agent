"""Tests for the ``hermes prompt-size`` diagnostic (issue #34667)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.prompt_size import (
    _SKILLS_BLOCK_RE,
    _build_inspection_agent,
    _compute_skills_breakdown,
    compute_prompt_breakdown,
    render_breakdown,
)


def _seed_memory(hermes_home, memory_text="", user_text=""):
    mem_dir = hermes_home / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    if memory_text:
        (mem_dir / "MEMORY.md").write_text(memory_text, encoding="utf-8")
    if user_text:
        (mem_dir / "USER.md").write_text(user_text, encoding="utf-8")


def _seed_skill(hermes_home, name, description):
    skill_dir = hermes_home / "skills" / "demo" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\nbody\n",
        encoding="utf-8",
    )


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.chdir(tmp_path)  # avoid picking up the repo's AGENTS.md
    return hermes_home




def test_runs_offline_without_credentials(isolated_home, monkeypatch):
    """No provider credentials configured → still produces a breakdown."""
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "NOUS_API_KEY",
                "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    data = compute_prompt_breakdown("cli")
    assert data["system_prompt"]["bytes"] > 0












def test_employee_prompt_size_excludes_installed_skills(isolated_home):
    _seed_skill(isolated_home, "small-skill", "short desc")
    _seed_skill(isolated_home, "big-skill", "a much longer description " * 20)
    data = compute_prompt_breakdown("cli")
    assert data["skills_breakdown"] == []
    assert data["skills_index"]["bytes"] == 0


def test_skills_breakdown_attributes_demoted_category_shared_line(isolated_home):
    """A real posture-demoted category retains every skill in the breakdown."""
    from agent.prompt_builder import build_skills_system_prompt

    _seed_skill(isolated_home, "alpha-skill", "alpha description")
    _seed_skill(isolated_home, "beta-skill", "beta description")
    prompt = build_skills_system_prompt(compact_categories=frozenset({"demo"}))
    skills_match = _SKILLS_BLOCK_RE.search(prompt)
    assert skills_match is not None
    skills_block = skills_match.group(0)
    shared_line = next(
        line for line in skills_block.splitlines() if "demo [names only]" in line
    )

    entries = _compute_skills_breakdown(skills_block)
    by_name = {entry["name"]: entry for entry in entries}
    assert set(by_name) == {"alpha-skill", "beta-skill"}

    shared_line_bytes = len(shared_line.encode("utf-8"))
    assert sum(entry["index_line_bytes"] for entry in entries) == shared_line_bytes
    for entry in entries:
        assert entry["index_line_total_bytes"] == shared_line_bytes
        assert entry["index_line_shared_bytes"] > 0
        assert entry["index_line_skill_count"] == 2






@pytest.fixture(autouse=True)
def _retained_native_contract(native_skills):
    """Exercise the retained native implementation, not employee surface policy."""
