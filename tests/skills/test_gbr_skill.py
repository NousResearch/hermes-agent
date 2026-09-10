"""Tests for the optional gbr phone-pairing skill.

Stdlib + pytest + unittest.mock only. No live network.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL_MD = REPO / "optional-skills" / "mcp" / "gbr" / "SKILL.md"
DESCRIPTION_MD = REPO / "optional-skills" / "mcp" / "DESCRIPTION.md"

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

FORBIDDEN = (
    "mailbox key",
    "mailbox keys",
    "x-gbr-key",
    "device.json",
)

MARKETING = re.compile(
    r"\b(powerful|comprehensive|seamless|advanced)\b",
    re.I,
)


def _skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_skill_lives_under_optional_mcp_not_bundled_or_plugins():
    assert SKILL_MD.is_file()
    rel = SKILL_MD.relative_to(REPO).as_posix()
    assert rel == "optional-skills/mcp/gbr/SKILL.md"
    assert not (REPO / "skills" / "mcp" / "gbr" / "SKILL.md").exists()
    assert not (REPO / "plugins" / "gbr").exists()


def test_description_hardline_single_line():
    text = _skill_text()
    m = re.search(r"^description: (.*)$", text, re.MULTILINE)
    assert m, "description must be a single YAML line, not a folded block"
    desc = m.group(1)
    assert len(desc) <= 60, (desc, len(desc))
    assert desc.endswith(".")
    assert not desc.startswith(">")
    assert "gbr" not in desc.lower()
    assert not MARKETING.search(desc)


def test_frontmatter_author_credits_human_first():
    text = _skill_text()
    m = re.search(r"^author: (.*)$", text, re.MULTILINE)
    assert m, "author field missing"
    author = m.group(1).strip()
    lead = author.split(",")[0].strip()
    assert lead != "community"
    assert lead != "Hermes Agent"
    assert "David Rad" in lead
    assert "LinespottingPrivate" in lead


def test_required_sections_in_order():
    text = _skill_text()
    assert re.search(r"^# GBR Skill\s*$", text, re.MULTILINE)
    positions = []
    for heading in REQUIRED_SECTIONS:
        idx = text.find(heading)
        assert idx != -1, f"missing section {heading}"
        positions.append(idx)
    assert positions == sorted(positions)


def test_how_to_run_uses_terminal_tool():
    text = _skill_text()
    how = text.split("## How to Run", 1)[1].split("## ", 1)[0]
    assert "`terminal`" in how
    assert "gbr-agent pair" in how
    assert "gbr-agent run" in how
    assert "hermes mcp add gbr --command node --args" in how
    assert "hermes mcp add gbr --url" not in how


def test_verification_is_one_gbr_agent_command():
    text = _skill_text()
    verify = text.split("## Verification", 1)[1]
    assert "gbr-agent doctor" in verify
    assert "curl" not in verify.lower()


def test_no_secrets_or_host_key_material():
    blob = _skill_text().lower()
    for needle in FORBIDDEN:
        assert needle not in blob, f"forbidden {needle!r} in SKILL.md"


def test_no_curl_grep_cat_prose_and_no_folded_description():
    text = _skill_text()
    assert "description: >" not in text
    assert re.search(r"\bcurl\b", text, re.I) is None
    # Word-level: do not instruct the agent to grep/cat host files.
    assert re.search(r"(?i)(?<!/)(?<!\w)grep(?!\w)", text) is None
    assert re.search(r"(?i)(?<!/)(?<!\w)cat(?!\w)", text) is None


def test_attach_only_loopback_or_stdio_and_spectator_role():
    text = _skill_text()
    assert "http://127.0.0.1:8788" in text
    assert "gbr-mcp" in text
    assert "spectator" in text.lower()
    assert "not orchestrator" in text.lower()
    # Real CLI: hermes mcp add NAME --command/--args. :8788 is Bot API REST, not MCP HTTP.
    assert "hermes mcp add gbr --command node --args mcp/gbr-mcp/bin/gbr-mcp.js" in text
    assert "hermes mcp add gbr --url http://127.0.0.1:8788" not in text
    assert "hermes mcp add gbr -- " not in text
    assert "Bot API REST" in text


def test_description_index_one_liner():
    index = DESCRIPTION_MD.read_text(encoding="utf-8")
    assert "`gbr`:" in index
    gbr_lines = [ln for ln in index.splitlines() if "`gbr`" in ln]
    assert len(gbr_lines) == 1
