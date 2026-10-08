#!/usr/bin/env python3
"""Validate Hermes skill frontmatter without loading private skill content.

Spec
----
A skill package is only useful if future agents can discover and load it
reliably. This helper scans one or more roots for ``SKILL.md`` files and checks
small, durable contracts that are easy to regress when hand-authoring skills:

* frontmatter starts at byte zero with ``---`` and has a closing ``---`` marker
* YAML frontmatter parses as a mapping
* ``name`` and ``description`` are present and non-empty strings
* ``description`` stays below the 1024-character tool-schema budget
* the body after frontmatter is non-empty

The checker is read-only. It prints paths and issue codes, never skill bodies or
secret-adjacent content. Use ``--json`` for cron/CI aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

MAX_DESCRIPTION_CHARS = 1024


@dataclass(frozen=True)
class Issue:
    code: str
    message: str


@dataclass(frozen=True)
class SkillResult:
    path: str
    status: str
    issues: list[Issue]


def _frontmatter_error(code: str) -> Issue:
    messages = {
        "missing-frontmatter": "SKILL.md must start with YAML frontmatter delimited by ---.",
        "frontmatter-not-byte-zero": "Frontmatter delimiter must be the first bytes in the file.",
        "unclosed-frontmatter": "Opening frontmatter delimiter has no closing --- marker.",
        "invalid-frontmatter-yaml": "Frontmatter must be valid YAML.",
        "frontmatter-not-mapping": "Frontmatter must parse to a YAML mapping/object.",
        "missing-name": "Frontmatter requires a non-empty string name.",
        "missing-description": "Frontmatter requires a non-empty string description.",
        "description-too-long": f"Description must be <= {MAX_DESCRIPTION_CHARS} characters.",
        "empty-body": "SKILL.md body after frontmatter must be non-empty.",
    }
    return Issue(code=code, message=messages[code])


def find_skill_files(roots: Iterable[Path]) -> list[Path]:
    """Return sorted SKILL.md files under roots, accepting direct file paths."""

    skill_files: set[Path] = set()
    for root in roots:
        if root.is_file():
            if root.name == "SKILL.md":
                skill_files.add(root)
            continue
        if root.is_dir():
            skill_files.update(root.rglob("SKILL.md"))
    return sorted(skill_files, key=lambda path: str(path))


def _split_frontmatter(text: str) -> tuple[str | None, str | None, list[Issue]]:
    if text.startswith("---\n") or text.startswith("---\r\n"):
        lines = text.splitlines(keepends=True)
        body_start = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                body_start = index + 1
                break
        if body_start is None:
            return None, None, [_frontmatter_error("unclosed-frontmatter")]
        frontmatter = "".join(lines[1 : body_start - 1])
        body = "".join(lines[body_start:])
        return frontmatter, body, []

    stripped = text.lstrip("\ufeff\n\r\t ")
    if stripped.startswith("---"):
        return None, None, [_frontmatter_error("frontmatter-not-byte-zero")]
    return None, None, [_frontmatter_error("missing-frontmatter")]


def validate_skill_file(path: Path) -> SkillResult:
    """Validate one SKILL.md and return issue metadata without body excerpts."""

    issues: list[Issue] = []
    text = path.read_text(encoding="utf-8")
    frontmatter, body, split_issues = _split_frontmatter(text)
    issues.extend(split_issues)

    metadata: object = None
    if frontmatter is not None:
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            issues.append(_frontmatter_error("invalid-frontmatter-yaml"))
            metadata = None

        if metadata is not None and not isinstance(metadata, dict):
            issues.append(_frontmatter_error("frontmatter-not-mapping"))
            metadata = None

        if isinstance(metadata, dict):
            name = metadata.get("name")
            if not isinstance(name, str) or not name.strip():
                issues.append(_frontmatter_error("missing-name"))

            description = metadata.get("description")
            if not isinstance(description, str) or not description.strip():
                issues.append(_frontmatter_error("missing-description"))
            elif len(description) > MAX_DESCRIPTION_CHARS:
                issues.append(_frontmatter_error("description-too-long"))

        if body is not None and not body.strip():
            issues.append(_frontmatter_error("empty-body"))

    status = "fail" if issues else "ok"
    return SkillResult(path=str(path), status=status, issues=issues)


def scan_roots(roots: Sequence[Path]) -> list[SkillResult]:
    return [validate_skill_file(path) for path in find_skill_files(roots)]


def _summary(results: Sequence[SkillResult]) -> dict[str, int]:
    ok = sum(1 for result in results if result.status == "ok")
    fail = sum(1 for result in results if result.status == "fail")
    return {"ok": ok, "fail": fail}


def _json_payload(results: Sequence[SkillResult]) -> str:
    payload = {"summary": _summary(results), "results": [asdict(result) for result in results]}
    return json.dumps(payload, indent=2, sort_keys=True)


def _text_report(results: Sequence[SkillResult]) -> str:
    summary = _summary(results)
    lines = [f"Skill frontmatter doctor: {summary['ok']} ok, {summary['fail']} fail"]
    for result in results:
        if result.status == "ok":
            continue
        lines.append(f"\n{result.path}")
        for issue in result.issues:
            lines.append(f"  - {issue.code}: {issue.message}")
    return "\n".join(lines)


def _default_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    return [repo_root / "skills", repo_root / "optional-skills"]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Hermes SKILL.md frontmatter.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Skill roots or SKILL.md files to scan. Defaults to repo skills/ and optional-skills/.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    roots = args.paths or _default_roots()
    results = scan_roots(roots)

    if args.json:
        print(_json_payload(results))
    else:
        print(_text_report(results))

    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
