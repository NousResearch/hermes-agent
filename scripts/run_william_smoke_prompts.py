#!/usr/bin/env python3
"""Offline validator/renderer for William-specific Hermes smoke prompts.

This script intentionally does not call an LLM or external APIs. It verifies that
smoke prompt definitions are complete, checks that referenced skills exist in the
local skill inventory, and writes a Markdown review artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SUITE = Path(__file__).resolve().parents[1] / "ops" / "william_smoke_prompts.json"
DEFAULT_OUT = Path.home() / ".hermes" / "reports" / "william_smoke_prompts.md"
SKILL_ROOTS = [Path.home() / ".hermes" / "skills", Path(__file__).resolve().parents[1] / "skills"]


def _skill_exists(name: str) -> bool:
    for root in SKILL_ROOTS:
        if not root.exists():
            continue
        for candidate in root.rglob("SKILL.md"):
            if candidate.parent.name == name:
                return True
    return False


def validate(suite: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("suite.cases must be a non-empty list")
        return errors, warnings
    seen: set[str] = set()
    for i, case in enumerate(cases, 1):
        cid = case.get("id")
        if not cid:
            errors.append(f"case {i}: missing id")
        elif cid in seen:
            errors.append(f"case {i}: duplicate id {cid}")
        seen.add(cid)
        if not case.get("prompt"):
            errors.append(f"case {cid or i}: missing prompt")
        behaviors = case.get("required_behaviors")
        if not isinstance(behaviors, list) or len(behaviors) < 2:
            errors.append(f"case {cid or i}: expected at least two required_behaviors")
        for skill in case.get("expected_skills", []):
            if not _skill_exists(skill):
                warnings.append(f"case {cid}: referenced skill not found locally: {skill}")
    return errors, warnings


def render_markdown(suite: dict, errors: list[str], warnings: list[str]) -> str:
    lines = [
        "# William Hermes Smoke Prompt Suite",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Suite: `{suite.get('suite', 'unknown')}` v{suite.get('version', '?')}",
        f"Owner: {suite.get('owner', 'unknown')}",
        "",
        suite.get("purpose", ""),
        "",
        "## Validation",
        "",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
    ]
    if errors:
        lines += ["", "### Errors", ""] + [f"- {e}" for e in errors]
    if warnings:
        lines += ["", "### Warnings", ""] + [f"- {w}" for w in warnings]
    lines += ["", "## Cases", ""]
    for case in suite.get("cases", []):
        lines += [
            f"### {case['id']}",
            "",
            f"Prompt: {case['prompt']}",
            "",
            "Expected skills: " + (", ".join(f"`{s}`" for s in case.get("expected_skills", [])) or "none"),
            "",
            "Required behaviors:",
        ]
        lines += [f"- {b}" for b in case.get("required_behaviors", [])]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    errors, warnings = validate(suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(suite, errors, warnings), encoding="utf-8")
    print(json.dumps({"ok": not errors, "errors": errors, "warnings": warnings, "output": str(args.out)}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
