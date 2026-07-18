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


def test_breakdown_keys_and_shape(isolated_home):
    """The breakdown exposes every documented key with int byte/char counts."""
    data = compute_prompt_breakdown("cli")
    assert set(data) >= {
        "platform",
        "model",
        "system_prompt",
        "skills_index",
        "memory",
        "user_profile",
        "tools",
        "sections",
    }
    assert data["platform"] == "cli"
    for key in ("system_prompt", "skills_index", "memory", "user_profile"):
        assert data[key]["bytes"] >= 0
        assert data[key]["chars"] >= 0
    assert data["tools"]["count"] >= 0
    assert data["tools"]["json_bytes"] >= 0
    # System prompt is non-trivial even with empty home (identity + guidance).
    assert data["system_prompt"]["bytes"] > 0


def test_runs_offline_without_credentials(isolated_home, monkeypatch):
    """No provider credentials configured → still produces a breakdown."""
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "NOUS_API_KEY",
                "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    data = compute_prompt_breakdown("cli")
    assert data["system_prompt"]["bytes"] > 0


def test_inspection_agent_uses_resolved_platform_toolsets(monkeypatch):
    """Inspection must match real CLI tool resolution, including disables."""
    captured = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    cfg = {
        "model": {"default": "test/model"},
        "agent": {"disabled_toolsets": ["memory"]},
    }

    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        SimpleNamespace(AIAgent=FakeAIAgent),
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda passed_cfg, platform: {"terminal", "file"},
    )

    _build_inspection_agent("cli")

    assert captured["model"] == "test/model"
    assert captured["platform"] == "cli"
    assert captured["enabled_toolsets"] == ["file", "terminal"]
    assert captured["disabled_toolsets"] == ["memory"]


def test_blank_slate_prompt_size_counts_only_minimal_tools(isolated_home):
    """Blank Slate prompt-size should report file + terminal schemas only."""
    from hermes_cli.config import save_config
    from hermes_cli.setup import (
        _blank_slate_minimal_toolsets,
        _blank_slate_minimize_config,
    )

    cfg = {"model": {"default": "MiniMax-M2.7"}}
    _blank_slate_minimal_toolsets(cfg)
    _blank_slate_minimize_config(cfg)
    save_config(cfg)

    data = compute_prompt_breakdown("cli")

    assert data["tools"]["count"] == 6


def test_skills_index_reflects_installed_skills(isolated_home):
    """Installing a skill makes the skills-index block non-empty.

    Note: the skills prompt is cached per-process (in-process LRU + disk
    snapshot), so we seed the skill BEFORE the first build rather than
    comparing before/after within one process.
    """
    _seed_skill(isolated_home, "hello", "a demo skill for size testing")
    data = compute_prompt_breakdown("cli")
    assert data["skills_index"]["bytes"] > 0


def test_memory_and_profile_are_attributed(isolated_home):
    """Memory and user-profile blocks are measured separately."""
    _seed_memory(
        isolated_home,
        memory_text="Project uses pytest.\n",
        user_text="User is a developer.\n",
    )
    data = compute_prompt_breakdown("cli")
    assert data["memory"]["bytes"] > 0
    assert data["user_profile"]["bytes"] > 0


def test_skills_block_regex_matches_tagged_block():
    text = "preamble\n<available_skills>\n  cat:\n    - a: b\n</available_skills>\ntail"
    m = _SKILLS_BLOCK_RE.search(text)
    assert m is not None
    assert m.group(0).startswith("<available_skills>")
    assert m.group(0).endswith("</available_skills>")


def test_toolsets_breakdown_reconciles_and_sorted(isolated_home):
    """Per-toolset schema bytes attribute every tool exactly once.

    Each resolved tool belongs to one registry toolset, so the grand total of
    per-toolset json bytes equals the whole-array total minus JSON framing
    (``2 * count`` bytes: brackets + ``", "`` separators between items).
    """
    data = compute_prompt_breakdown("cli")
    toolsets = data["toolsets_breakdown"]
    assert toolsets  # CLI always resolves at least terminal + file
    for ts in toolsets:
        assert set(ts) >= {"toolset", "tool_count", "json_bytes"}
        assert ts["tool_count"] >= 1
        assert ts["json_bytes"] > 0
    # Sorted largest-first.
    byte_sizes = [ts["json_bytes"] for ts in toolsets]
    assert byte_sizes == sorted(byte_sizes, reverse=True)
    # Every tool attributed to exactly one toolset.
    assert sum(ts["tool_count"] for ts in toolsets) == data["tools"]["count"]
    # Bytes reconcile to the existing whole-array total.
    grand = sum(ts["json_bytes"] for ts in toolsets)
    assert grand == data["tools"]["json_bytes"] - 2 * data["tools"]["count"]


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


def test_skills_breakdown_unmapped_name_is_none():
    """A skill line with no matching SKILL.md on disk reports None, not a crash."""
    block = (
        "<available_skills>\n"
        "  demo:\n"
        "    - phantom-skill: not on disk\n"
        "</available_skills>\n"
    )
    entries = _compute_skills_breakdown(block)
    assert len(entries) == 1
    assert entries[0]["name"] == "phantom-skill"
    assert entries[0]["skill_md_bytes"] is None
    assert entries[0]["path"] == ""
    assert entries[0]["index_line_bytes"] > 0


def test_skills_breakdown_parses_namespaced_names():
    """Namespaced names (``ns:skill``) survive the ``name: desc`` split."""
    block = (
        "<available_skills>\n"
        "  plugins:\n"
        "    - codex:rescue: rescue helper\n"
        "</available_skills>\n"
    )
    entries = _compute_skills_breakdown(block)
    assert [e["name"] for e in entries] == ["codex:rescue"]


def test_render_includes_per_component_tables(isolated_home):
    """The rendered report gains the two new sorted tables (additive)."""
    _seed_skill(isolated_home, "demo-skill", "a demo skill")
    data = compute_prompt_breakdown("cli")
    out = render_breakdown(data)
    assert "Toolsets by size" in out
    assert "Skills by size" in out


def test_render_breakdown_is_plain_text(isolated_home):
    data = compute_prompt_breakdown("cli")
    out = render_breakdown(data)
    assert "System prompt total" in out
    assert "skills index" in out
    assert "Tool schemas" in out
    # Plain text — no JSON braces leaking in.
    assert not out.strip().startswith("{")


def test_json_serializable(isolated_home):
    data = compute_prompt_breakdown("cli")
    # Round-trips cleanly for ``--json`` output.
    assert json.loads(json.dumps(data)) == json.loads(json.dumps(data))
