"""Tests for the handoff skill: SKILL.md contract and the HANDOFF.md checker."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import hermes_yaml as yaml

from agent.skill_utils import SKILL_PROMPT_DESC_LIMIT

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "productivity" / "handoff"
SKILL_MD = SKILL_DIR / "SKILL.md"
CHECKER = SKILL_DIR / "scripts" / "check_handoff.py"

REQUIRED_SECTIONS = (
    "Current Task",
    "Done",
    "Blockers",
    "Next Steps",
    "Landmines",
    "Suggested Skills",
)

MODERN_ORDER = (
    "When to Use",
    "Prerequisites",
    "How to Run",
    "Quick Reference",
    "Procedure",
    "Pitfalls",
    "Verification",
)

MARKETING = ("powerful", "comprehensive", "seamless", "revolutionary", "cutting-edge")

VALID = """# HANDOFF

## Current Task
Ship the chat auto-scroll fix.

## Done
- [x] follow-scroll landed — `web/src/pages/ChatPage.tsx:1381`, commit `abc1234`

## Blockers
none

## Next Steps
1. Run the web suite — acceptance: the suite is green on a fresh checkout

## Landmines
- Trap: scrolling right after `term.write()`. Root cause: xterm applies writes
  asynchronously, so the viewport is stale. Fix: scroll inside the write callback.

## Suggested Skills
- `pdf` — only if the report has to be rendered

▶ Next session: read HANDOFF.md first
"""


def _load_checker():
    spec = importlib.util.spec_from_file_location("handoff_checker", CHECKER)
    assert spec and spec.loader, f"cannot load {CHECKER}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frontmatter() -> tuple[dict, str]:
    content = SKILL_MD.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must start with ---"
    end = content.index("\n---", 3)
    return yaml.safe_load(content[3:end]), content[end + 4 :]


def _headings(body: str) -> list[str]:
    return [
        line[3:].strip()
        for line in body.splitlines()
        if line.startswith("## ")
    ]


def _run(tmp_path: Path, text: str) -> subprocess.CompletedProcess:
    target = tmp_path / "HANDOFF.md"
    target.write_text(text, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(CHECKER), str(target)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_skill_directory_is_where_the_docs_generator_expects_it():
    assert SKILL_MD.is_file()
    assert CHECKER.is_file()


def test_frontmatter_satisfies_authoring_standards():
    fm, _ = _frontmatter()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == SKILL_DIR.name
    description = str(fm["description"])
    assert len(description) <= SKILL_PROMPT_DESC_LIMIT, (
        f"description is {len(description)} chars (limit {SKILL_PROMPT_DESC_LIMIT})"
    )
    assert description.rstrip().endswith(".")
    lowered = description.lower()
    assert not [word for word in MARKETING if word in lowered]
    assert "handoff" not in lowered, "the index already prints the skill name"
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    assert hermes.get("tags")


def test_body_follows_the_modern_section_order():
    _, body = _frontmatter()
    assert body.lstrip().startswith("# "), "body must open with an H1 title"
    headings = _headings(body)
    positions = []
    for name in MODERN_ORDER:
        assert name in headings, f"missing section: ## {name}"
        positions.append(headings.index(name))
    assert positions == sorted(positions), f"sections out of order: {headings}"


def test_slash_invocation_is_the_registered_one():
    _, body = _frontmatter()
    assert "/skill handoff" in body
    assert "/load handoff" not in body
    assert "`/handoff`" not in body or "built-in" in body


def test_template_covers_every_required_section():
    _, body = _frontmatter()
    fenced = body.split("```markdown", 1)[1].split("```", 1)[0]
    for name in REQUIRED_SECTIONS:
        assert f"## {name}" in fenced, f"template omits ## {name}"
    assert "Next session: read HANDOFF.md first" in fenced


def test_valid_handoff_passes(tmp_path):
    result = _run(tmp_path, VALID)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6/6 sections" in result.stdout


def test_missing_section_is_named(tmp_path):
    text = VALID.replace("## Landmines", "## Notes")
    result = _run(tmp_path, text)
    assert result.returncode == 1
    assert "missing section: ## Landmines" in result.stdout


def test_placeholder_only_section_fails(tmp_path):
    text = VALID.replace(
        "- [x] follow-scroll landed — `web/src/pages/ChatPage.tsx:1381`, commit `abc1234`",
        "- TODO",
    )
    result = _run(tmp_path, text)
    assert result.returncode == 1
    assert "## Done" in result.stdout
    assert "placeholder" in result.stdout


def test_explicit_none_is_an_answer(tmp_path):
    result = _run(tmp_path, VALID)
    assert result.returncode == 0
    assert "## Blockers" not in result.stdout


def test_secret_is_reported_without_being_echoed(tmp_path):
    token = "sk-live-abcdefghijklmnopqrstuvwx"
    text = VALID.replace(
        "none", f"- waiting on the vendor, key {token}", 1
    )
    result = _run(tmp_path, text)
    assert result.returncode == 1
    assert "credential" in result.stdout
    assert token not in result.stdout + result.stderr


def test_closing_line_is_required(tmp_path):
    text = VALID.replace("▶ Next session: read HANDOFF.md first", "")
    result = _run(tmp_path, text)
    assert result.returncode == 1
    assert "closing line" in result.stdout


def test_missing_file_exits_two(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(tmp_path / "absent.md")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "cannot read" in result.stderr


def test_fenced_template_placeholders_are_ignored(tmp_path):
    text = VALID.replace(
        "## Suggested Skills",
        "```markdown\n## Suggested Skills\n- skill-name: why\n```\n\n## Suggested Skills",
    )
    result = _run(tmp_path, text)
    assert result.returncode == 0, result.stdout


def test_checker_api_reports_problems_as_data():
    checker = _load_checker()
    problems, warnings = checker.check(VALID)
    assert problems == []
    assert warnings == []
    problems, _ = checker.check("# HANDOFF\n\n## Done\n- x\n")
    assert any("missing section" in problem for problem in problems)
    assert len(problems) >= len(checker.REQUIRED) - 1
