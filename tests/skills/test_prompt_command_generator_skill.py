"""Safety contract for the prompt/command generator skill."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT / "skills" / "productivity" / "prompt-command-generator" / "SKILL.md"
)


def _skill() -> tuple[str, str]:
    assert SKILL_PATH.is_file(), (
        "prompt-command-generator should ship as a bundled skill"
    )
    text = SKILL_PATH.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None, "skill must have YAML frontmatter"
    return match.group(1), text


def _section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"skill must contain a {heading!r} section"
    return match.group(1).lower()


def test_skill_declares_a_cross_platform_prompt_and_command_capability() -> None:
    frontmatter, text = _skill()

    assert re.search(
        r"^name:\s*prompt-command-generator\s*$", frontmatter, re.MULTILINE
    )
    assert re.search(
        r"^platforms:\s*\[linux,\s*macos,\s*windows\]\s*$",
        frontmatter,
        re.MULTILINE,
    )
    assert re.search(r"\bprompt\b", text, re.IGNORECASE)
    assert re.search(r"\bshell\b", text, re.IGNORECASE)


def test_generated_commands_are_drafts_not_executed_actions() -> None:
    _frontmatter, text = _skill()
    intro = text.split("\n## When to Use", 1)[0].lower()
    procedure = _section(text, "Procedure")
    pitfalls = _section(text, "Pitfalls")

    assert re.search(r"\bcommands?\b.{0,80}\bdrafts?\b", intro)
    assert re.search(
        r"\b(?:does not|do not|never)\s+(?:execute|run)\b", intro + procedure
    )
    assert re.search(
        r"\bseparat\w*\b.{0,60}\b(?:asks?|requests?)\b|"
        r"\b(?:asks?|requests?)\b.{0,60}\bseparat\w*\b",
        procedure,
    )
    assert re.search(r"\buser\b", procedure)
    assert re.search(
        r"\b(?:never|do not|must not|avoid)\b.{0,80}"
        r"\b(?:secrets?|tokens?|passwords?|private keys?)\b",
        procedure + pitfalls,
    )


def test_command_risk_and_verification_are_explicit() -> None:
    _frontmatter, text = _skill()
    procedure = _section(text, "Procedure")

    assert re.search(r"\bread[\s-]?only\b", procedure)
    assert re.search(r"\b(?:write|change)\w*\b.{0,40}\bstate\b", procedure)
    assert re.search(r"\bdestructive\b", procedure)
    assert re.search(r"^## verification\s*$", text, re.MULTILINE | re.IGNORECASE)
