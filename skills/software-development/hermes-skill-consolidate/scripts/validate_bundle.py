#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "SKILL.md",
    "README.md",
    "references/protocol.md",
    "references/safety.md",
    "references/decision-model.md",
    "references/report-contract.md",
    "examples/example-report.md",
    "templates/consolidation-plan.json",
    "scripts/snapshot_skills.py",
    "tests/cases.json",
    "tests/contract-cases.json",
    "tests/test_contracts.py",
    "tests/test_snapshot.py",
]
SKILL_SECTIONS = [
    "## Overview",
    "## When to Use",
    "## Counter-Triggers",
    "## Safety Contract",
    "## Untrusted Content Boundary",
    "## Workflow",
    "## Required Procedure",
    "## Classification",
    "## Report Contract",
    "## Common Pitfalls",
    "## Verification Checklist",
]
README_SECTIONS = [
    "## Provenance",
    "## Inputs",
    "## Outputs",
    "## Requirements",
    "## Install",
    "## Invocation",
    "## Safety",
    "## Privacy",
    "## Limitations",
    "## Examples",
    "## Validation",
    "## Version history",
    "## License",
]


def parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("SKILL.md frontmatter is not closed")
    result: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if line and not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            result[key] = value.strip().strip('"').strip("'")
    return result


def validate() -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing: {relative}")
    if errors:
        return errors

    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    safety = (ROOT / "references" / "safety.md").read_text(encoding="utf-8")
    decision = (ROOT / "references" / "decision-model.md").read_text(encoding="utf-8")
    examples = (ROOT / "examples" / "example-report.md").read_text(encoding="utf-8")

    try:
        frontmatter = parse_frontmatter(skill)
    except ValueError as exc:
        errors.append(str(exc))
        frontmatter = {}

    expected = {
        "name": "hermes-skill-consolidate",
        "version": "0.1.0",
        "author": "Tony Simons",
        "license": "Apache-2.0",
    }
    for key, value in expected.items():
        if frontmatter.get(key) != value:
            errors.append(f"frontmatter {key} must equal {value}")
    if not frontmatter.get("description", "").startswith("Use when "):
        errors.append("frontmatter description must begin with 'Use when '")

    positions: list[int] = []
    for heading in SKILL_SECTIONS:
        if heading not in skill:
            errors.append(f"SKILL.md missing section: {heading}")
        else:
            positions.append(skill.index(heading))
    if positions != sorted(positions):
        errors.append("SKILL.md required sections are out of order")

    readme_positions: list[int] = []
    for heading in README_SECTIONS:
        if heading not in readme:
            errors.append(f"README.md missing section: {heading}")
        else:
            readme_positions.append(readme.index(heading))
    if readme_positions != sorted(readme_positions):
        errors.append("README.md required sections are out of order")

    for marker in [
        "second explicit approval",
        "safety is monotonic",
        "stage replacement content outside the live skills tree",
        "do not permanently delete originals",
        "approval is valid only",
    ]:
        if marker.lower() not in skill.lower():
            errors.append(f"SKILL.md missing safety marker: {marker}")

    combined_boundary = (skill + "\n" + safety).lower()
    for marker in [
        "untrusted evidence",
        "never follow instructions",
        "do not activate",
        "prompt injection or social engineering",
    ]:
        if marker not in combined_boundary:
            errors.append(f"missing hostile-content boundary marker: {marker}")

    for relationship in [
        "CONFIRMED DUPLICATE",
        "LIKELY REDUNDANT",
        "PARTIAL OVERLAP",
        "COMPLEMENTARY",
        "PARENT OR ORCHESTRATOR",
        "SHARED REFERENCE CANDIDATE",
        "INTENTIONALLY SEPARATE",
        "INSUFFICIENT EVIDENCE",
    ]:
        if relationship not in skill or relationship not in decision:
            errors.append(f"relationship classification missing: {relationship}")

    if "## Successful use" not in examples or "## Boundary or failure mode" not in examples:
        errors.append("examples must include successful and boundary scenarios")

    try:
        behavior = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
        contracts = json.loads((ROOT / "tests" / "contract-cases.json").read_text(encoding="utf-8"))
        plan = json.loads((ROOT / "templates" / "consolidation-plan.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON: {exc}")
    else:
        ids = {case.get("id") for case in behavior.get("cases", [])}
        for required_id in [
            "approval-boundary",
            "safety-separation",
            "untrusted-content-boundary",
            "rollback-before-write",
        ]:
            if required_id not in ids:
                errors.append(f"tests/cases.json missing {required_id} case")
        if len(contracts.get("untrusted_content_prompts", [])) < 2:
            errors.append("contract-cases.json missing hostile-content prompts")
        if len(contracts.get("approval_mutation_prompts", [])) < 2:
            errors.append("contract-cases.json missing approval prompts")
        if plan.get("approval", {}).get("required") is not True:
            errors.append("plan template must require approval")
        if plan.get("rollback", {}).get("snapshot_required") is not True:
            errors.append("plan template must require snapshot")

    snapshot = (ROOT / "scripts" / "snapshot_skills.py").read_text(encoding="utf-8")
    for forbidden in ["delete", "restore", "apply"]:
        if f'sub.add_parser("{forbidden}"' in snapshot:
            errors.append(f"snapshot helper must not expose {forbidden} command")

    secret_pattern = re.compile(
        r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9+/=_-]{20,}"
    )
    forbidden_markers = [
        "C:" + "\\Users\\" + "example-user",
        "/home/" + "example-user",
        "internal" + "." + "example",
        "private-" + "knowledge-base",
        "SECRET_" + "TOKEN=",
    ]
    for path in ROOT.rglob("*"):
        if path.name in {"__pycache__", ".DS_Store", "Thumbs.db"} or path.suffix in {".pyc", ".pyo"}:
            errors.append(f"generated artifact present: {path.relative_to(ROOT)}")
        if path.is_symlink():
            errors.append(f"symlink is not allowed: {path.relative_to(ROOT)}")
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in forbidden_markers:
            if marker in content:
                errors.append(f"private marker in {path.relative_to(ROOT)}: {marker}")
        if secret_pattern.search(content):
            errors.append(f"possible assigned secret in {path.relative_to(ROOT)}")

    license_path = ROOT.parents[1] / "LICENSE"
    if not license_path.is_file() or "Apache License" not in license_path.read_text(encoding="utf-8"):
        errors.append("root Apache-2.0 license is missing")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS: hermes-skill-consolidate bundle is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
