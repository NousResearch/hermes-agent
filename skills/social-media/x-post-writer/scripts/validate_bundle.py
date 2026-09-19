#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "SKILL.md",
    "README.md",
    "references/formats.md",
    "references/source-routing.md",
    "references/claim-verification.md",
    "references/voice-customization.md",
    "references/algorithm-notes.md",
    "references/x-articles-scope-note.md",
    "examples/routing-examples.md",
    "tests/cases.json",
]

FORBIDDEN = [
    "Tony's Private",
    "TRT",
    "tony-content-bundle",
    "write-like-tony",
    "trt-x-post",
    "x-claim-verification",
    "private-voice-profile",
    "x-threads",
    "x-draft-scoring",
    "x-post-slate-builder",
    "C:\\Users\\asimo",
    "tonyreviewsthings",
    "highest-value signal",
]

REQUIRED_CASES = {
    "single-post-default",
    "explicit-thread",
    "quote-post",
    "reply-route",
    "personal-story",
    "user-draft-preservation",
    "fresh-angle-repurpose",
    "x-article-counter-trigger",
    "claim-verification",
    "no-invented-experience",
    "draft-only-delivery",
    "algorithm-uncertainty",
    "sensitive-data-gate",
    "no-auto-publish",
    "sparse-notes-no-gap-filling",
    "thread-labels",
}


def validate() -> list[str]:
    errors: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    skill_path = ROOT / "SKILL.md"
    if skill_path.is_file():
        skill = skill_path.read_text(encoding="utf-8")
        if not skill.startswith("---\n"):
            errors.append("SKILL.md must start with frontmatter")
        for marker in (
            "name: x-post-writer",
            "version: 1.0.0",
            "license: Apache-2.0",
            "## Overview",
            "## When to Use",
            "## Source Lock",
            "## Workflow",
            "## Common Pitfalls",
            "## Verification Checklist",
        ):
            if marker not in skill:
                errors.append(f"SKILL.md missing marker: {marker}")
        if len(skill.splitlines()) > 260:
            errors.append("SKILL.md exceeds 260 lines")

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".json"}
    )
    for phrase in FORBIDDEN:
        if phrase.casefold() in combined.casefold():
            errors.append(f"forbidden private or stale phrase found: {phrase}")

    cases_path = ROOT / "tests" / "cases.json"
    if cases_path.is_file():
        try:
            data = json.loads(cases_path.read_text(encoding="utf-8"))
            cases = data.get("cases", [])
            ids = [case.get("id") for case in cases]
            if len(ids) != len(set(ids)):
                errors.append("behavior case IDs must be unique")
            missing = REQUIRED_CASES - set(ids)
            if missing:
                errors.append(f"missing behavior cases: {sorted(missing)}")
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"invalid tests/cases.json: {exc}")

    return errors


if __name__ == "__main__":
    failures = validate()
    if failures:
        print(f"Validation failed with {len(failures)} error(s):")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)
    print("Validation passed: x-post-writer 1.0.0 bundle is structurally clean.")
