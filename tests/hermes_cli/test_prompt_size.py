"""Tests for the ``hermes prompt-size`` diagnostic (issue #34667)."""

import json

import pytest

from hermes_cli.prompt_size import (
    _SKILLS_BLOCK_RE,
    build_full_prompt_text,
    cmd_prompt_dump,
    cmd_prompt_size,
    compute_prompt_breakdown,
    render_breakdown,
)


class _Args:
    """Minimal argparse-like namespace for driving cmd_prompt_size."""

    def __init__(self, platform="cli", json=False):
        self.platform = platform
        self.json = json


class _DumpArgs:
    """Minimal argparse-like namespace for driving cmd_prompt_dump."""

    def __init__(self, platform="cli", json=False):
        self.platform = platform
        self.json = json


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


def test_build_full_prompt_text_non_empty_with_marker(isolated_home):
    """prompt-dump builds the full assembled prompt with a stable-tier marker."""
    prompt = build_full_prompt_text("cli")
    assert prompt.strip()
    # Stable identity tier always opens with the Hermes Agent identity line.
    assert "You are Hermes Agent" in prompt


def test_cmd_dump_prints_full_prompt(isolated_home, capsys):
    """`prompt-dump` prints the raw prompt (no size table/banner)."""
    cmd_prompt_dump(_DumpArgs())
    out = capsys.readouterr().out
    assert "You are Hermes Agent" in out
    # Size-report labels must NOT appear in dump mode.
    assert "System prompt total" not in out
    assert "Tool schemas" not in out


def test_cmd_dump_json_emits_prompt_key(isolated_home, capsys):
    """`prompt-dump --json` wraps the prompt in a {"prompt": ...} object."""
    cmd_prompt_dump(_DumpArgs(json=True))
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "prompt" in payload
    assert "You are Hermes Agent" in payload["prompt"]


def test_cmd_prompt_size_unchanged_prints_size_report(isolated_home, capsys):
    """prompt-size still prints the size breakdown, never the raw prompt."""
    cmd_prompt_size(_Args())
    out = capsys.readouterr().out
    assert "System prompt total" in out
    assert "You are Hermes Agent" not in out
