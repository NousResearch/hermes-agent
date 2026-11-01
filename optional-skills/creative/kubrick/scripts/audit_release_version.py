#!/usr/bin/env python3
"""Audit Kubrick release-version declarations against the VERSION manifest."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from manifest_contract import load_manifest

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
RELEASE_LINE = ".".join(EXPECTED_VERSION.split(".")[:2])
RELEASE_NOTES = f"docs/RELEASE-NOTES-v{RELEASE_LINE}.md"
RELEASE_ROADMAP = f"docs/ROADMAP-v{RELEASE_LINE}.md"
RELEASE_CHECKLIST = f"docs/RELEASE-CHECKLIST-v{RELEASE_LINE}.md"

TARGETS = {
    "SKILL.md": re.compile(r"^version:\s*([^\s]+)", re.MULTILINE),
    "README.md": re.compile(r"<em>(0\.\d+\.\d+)\s+—"),
    "CHANGELOG.md": re.compile(r"^## \[(0\.\d+\.\d+)\]", re.MULTILINE),
    RELEASE_NOTES: re.compile(r"^# Kubrick v(0\.\d+\.\d+) Release Notes", re.MULTILINE),
}

REQUIRED_CURRENT_REFERENCES = {
    "README.md": [RELEASE_ROADMAP, RELEASE_NOTES],
    "SKILL.md": [RELEASE_ROADMAP, RELEASE_NOTES],
    RELEASE_CHECKLIST: [f"v{EXPECTED_VERSION}"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    expected = EXPECTED_VERSION
    declarations: dict[str, str | None] = {}
    mismatches: list[dict[str, object]] = []

    manifest_version = load_manifest()["version"]
    declarations["kubrick.manifest.yaml"] = manifest_version
    if manifest_version != expected:
        mismatches.append(
            {
                "file": "kubrick.manifest.yaml",
                "expected": expected,
                "observed": manifest_version,
                "reason": "version mismatch",
            }
        )

    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_section = pyproject_text.split("[project]", 1)[1].split("[", 1)[0]
    pyproject_match = re.search(r'^version\s*=\s*"([^"]+)"', project_section, re.MULTILINE)
    pyproject_version = pyproject_match.group(1) if pyproject_match else None
    declarations["pyproject.toml"] = pyproject_version
    if pyproject_version != expected:
        mismatches.append(
            {
                "file": "pyproject.toml",
                "expected": expected,
                "observed": pyproject_version,
                "reason": "version mismatch",
            }
        )

    for relative, pattern in TARGETS.items():
        path = ROOT / relative
        if not path.exists():
            declarations[relative] = None
            mismatches.append({"file": relative, "expected": expected, "observed": None, "reason": "missing file"})
            continue
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        observed = match.group(1) if match else None
        declarations[relative] = observed
        if observed != expected:
            mismatches.append({"file": relative, "expected": expected, "observed": observed, "reason": "version mismatch"})

    reference_checks: dict[str, dict[str, bool]] = {}
    for relative, required_values in REQUIRED_CURRENT_REFERENCES.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        checks = {value: value in text for value in required_values}
        reference_checks[relative] = checks
        for value, present in checks.items():
            if not present:
                mismatches.append(
                    {
                        "file": relative,
                        "expected": value,
                        "observed": None,
                        "reason": "missing current release reference",
                    }
                )

    report = {
        "status": "READY" if not mismatches else "NOT_READY",
        "expected_version": expected,
        "declarations": declarations,
        "reference_checks": reference_checks,
        "mismatches": mismatches,
        "tag_allowed": not mismatches,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(1 if args.strict and mismatches else 0)


if __name__ == "__main__":
    main()
