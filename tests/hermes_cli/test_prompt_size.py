"""Tests for the ``hermes prompt-size`` diagnostic (issue #34667)."""

import json

import pytest

from hermes_cli.prompt_size import (
    _SKILLS_BLOCK_RE,
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


def test_tool_schemas_respect_platform_toolsets(isolated_home):
    """The inspection agent must mirror gateway/cron toolset resolution.

    Regression: _build_inspection_agent used to omit enabled_toolsets, so the
    reported tool schemas were the FULL default set regardless of the
    platform_toolsets config — overstating the wire payload (androux-y7ufv).
    """
    (isolated_home / "config.yaml").write_text(
        "platform_toolsets:\n"
        "  telegram:\n"
        "  - clarify\n"
        "  - todo\n"
        "  - no_mcp\n",
        encoding="utf-8",
    )
    scoped = compute_prompt_breakdown("telegram")
    unscoped = compute_prompt_breakdown("cli")  # cli has no explicit config here

    assert scoped["tools"]["count"] < unscoped["tools"]["count"], (
        "explicitly scoped platform should ship fewer tool schemas than an "
        "unconfigured one"
    )
    # The scoped platform's tools must be a small set — clarify + todo only
    # (plus nothing smuggled in by plugins/MCP defaults).
    assert scoped["tools"]["count"] <= 4


def test_skills_index_platform_allowlist(isolated_home, monkeypatch):
    """skills.platform_enabled.<platform> gates the <available_skills> index.

    Machine platforms (cron) should carry a ~10-entry index, not 300+ —
    androux-0svm6 A1. The allowlist matches categories or individual skill
    names; absent config means no filtering.
    """
    from agent.prompt_builder import build_skills_system_prompt

    _seed_skill(isolated_home, "keeper", "stays in the index")
    other_dir = isolated_home / "skills" / "noise" / "dropped"
    other_dir.mkdir(parents=True)
    (other_dir / "SKILL.md").write_text(
        "---\nname: dropped\ndescription: should vanish\n---\n# dropped\n",
        encoding="utf-8",
    )
    (isolated_home / "config.yaml").write_text(
        "skills:\n  platform_enabled:\n    cron:\n    - demo\n",
        encoding="utf-8",
    )

    # Explicit platform arg — the path real cron agents use (cron never
    # seeds the session-platform env; see build_skills_system_prompt).
    scoped = build_skills_system_prompt(platform="cron")
    assert "keeper" in scoped
    assert "dropped" not in scoped

    monkeypatch.setenv("HERMES_PLATFORM", "telegram")  # no allowlist for it
    unscoped = build_skills_system_prompt()
    assert "keeper" in unscoped and "dropped" in unscoped
