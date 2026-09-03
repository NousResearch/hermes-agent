#!/usr/bin/env python3
"""Pre-release gate for Neon Genie (stdlib only).

Validates:
  - VERSION is semver
  - VERSION == SKILL.md frontmatter == manifest.json
  - CHANGELOG has a section for this version
  - distribution spine is clean
  - optional: tag name matches VERSION when RELEASE_TAG is set

Usage:
  python scripts/release_check.py
  python scripts/release_check.py --tag v3.23.0
  python scripts/release_check.py --skip-dist
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
PY = sys.executable
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def frontmatter_version(text: str) -> str | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    for line in text[4:end].splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Neon Genie pre-release gate")
    parser.add_argument("--tag", default="", help="Git tag (e.g. v3.23.0) must match VERSION")
    parser.add_argument("--skip-dist", action="store_true", help="Skip distribution_spine verify")
    parser.add_argument("--skip-audit", action="store_true", help="Skip audit_release_version")
    args = parser.parse_args(argv)

    errors: list[str] = []
    version_path = SKILL_ROOT / "VERSION"
    if not version_path.is_file():
        errors.append("NG-VERSION-001: VERSION missing")
        version = ""
    else:
        version = version_path.read_text(encoding="utf-8").strip()
        if not SEMVER.fullmatch(version):
            errors.append(f"NG-VERSION-002: VERSION not semver: {version!r}")

    skill_v = frontmatter_version((SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8"))
    if skill_v != version:
        errors.append(f"NG-VERSION-003: SKILL.md version {skill_v!r} != VERSION {version!r}")

    try:
        man = json.loads((SKILL_ROOT / "manifest.json").read_text(encoding="utf-8"))
        man_v = str(man.get("version", ""))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"NG-VERSION-004: manifest.json unreadable: {exc}")
        man_v = ""
    if man_v != version:
        errors.append(f"NG-VERSION-005: manifest version {man_v!r} != VERSION {version!r}")

    changelog = (SKILL_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if version and f"## [{version}]" not in changelog and f"## {version}" not in changelog:
        errors.append(f"NG-VERSION-006: CHANGELOG.md missing section for {version}")

    if args.tag:
        tag = args.tag.strip()
        expect = f"v{version}"
        if tag.lstrip("v") != version and tag != expect:
            errors.append(f"NG-VERSION-007: tag {tag!r} does not match VERSION {version!r}")

    if not args.skip_audit:
        r = subprocess.run(
            [PY, str(SCRIPT_DIR / "audit_release_version.py"), "--strict"],
            cwd=SKILL_ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            errors.append(f"NG-VERSION-008: audit_release_version failed:\n{r.stderr or r.stdout}")

    if not args.skip_dist:
        r = subprocess.run(
            [PY, str(SCRIPT_DIR / "distribution_spine.py"), "verify"],
            cwd=SKILL_ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            errors.append(f"NG-PKG-030: distribution spine dirty:\n{r.stderr or r.stdout}")

    if errors:
        print("FAIL: release check", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"PASS: release check (v{version})")
    print("  version alignment: ok")
    print("  changelog: ok")
    if not args.skip_dist:
        print("  distribution spine: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
