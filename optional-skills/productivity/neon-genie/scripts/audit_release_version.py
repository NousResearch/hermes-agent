#!/usr/bin/env python3
"""Audit VERSION alignment across Neon Genie packaging files (stdlib only).

Usage:
  python scripts/audit_release_version.py
  python scripts/audit_release_version.py --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent


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
    parser = argparse.ArgumentParser(description="Audit Neon Genie release version alignment")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any mismatch (default: fail on core mismatches always)",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import paths as ng_paths  # noqa: E402

    try:
        version_path = ng_paths.version_path()
        version = version_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            errors.append(f"VERSION not semver X.Y.Z: {version!r}")
    except (OSError, FileNotFoundError):
        errors.append("VERSION missing (also tried references/VERSION)")
        version = ""

    skill_v = frontmatter_version((SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8"))
    if skill_v != version:
        errors.append(f"SKILL.md version {skill_v!r} != VERSION {version!r}")

    try:
        manifest = json.loads(ng_paths.manifest_path().read_text(encoding="utf-8"))
        man_v = str(manifest.get("version", ""))
    except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
        errors.append(f"manifest.json unreadable: {exc}")
        man_v = ""
    if man_v != version:
        errors.append(f"manifest.json version {man_v!r} != VERSION {version!r}")

    # Soft checks: README badge / version table mention
    readme = (SKILL_ROOT / "README.md").read_text(encoding="utf-8") if (SKILL_ROOT / "README.md").is_file() else ""
    if version and version not in readme:
        msg = f"README.md does not mention VERSION {version}"
        (errors if args.strict else warnings).append(msg)

    changelog = (
        (SKILL_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        if (SKILL_ROOT / "CHANGELOG.md").is_file()
        else ""
    )
    if version and f"[{version}]" not in changelog and f"## [{version}]" not in changelog:
        msg = f"CHANGELOG.md missing section for {version}"
        (errors if args.strict else warnings).append(msg)

    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        print("FAIL: release version audit", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"PASS: release version audit ({version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
