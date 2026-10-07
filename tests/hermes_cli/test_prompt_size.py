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












def test_skills_breakdown_shape_sorted_and_attributed(isolated_home):
    """Per-skill breakdown reports index-line + on-disk SKILL.md bytes.

    Seeded before the first build (skills prompt is cached per-process).
    """
    _seed_skill(isolated_home, "small-skill", "short desc")
    _seed_skill(isolated_home, "big-skill", "a much longer description " * 20)
    data = compute_prompt_breakdown("cli")
    skills = data["skills_breakdown"]
    names = {s["name"] for s in skills}
    assert {"small-skill", "big-skill"} <= names
    for s in skills:
        assert set(s) >= {"name", "index_line_bytes", "skill_md_bytes", "path"}
        assert s["index_line_bytes"] > 0
    # Sorted largest-first by on-disk SKILL.md size.
    md_sizes = [s["skill_md_bytes"] or 0 for s in skills]
    assert md_sizes == sorted(md_sizes, reverse=True)
    # On-disk bytes match the real file; big-skill's SKILL.md is the larger.
    by_name = {s["name"]: s for s in skills}
    big = by_name["big-skill"]
    assert big["path"] and Path(big["path"]).stat().st_size == big["skill_md_bytes"]
    assert big["skill_md_bytes"] > by_name["small-skill"]["skill_md_bytes"]
    # Per-skill index lines are a subset of the whole <available_skills> block,
    # so they never exceed it (on-disk SKILL.md bytes are separate and don't).
    assert sum(s["index_line_bytes"] for s in skills) <= data["skills_index"]["bytes"]


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




