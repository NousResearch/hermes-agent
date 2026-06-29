#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# ///
"""quick_validate — structural lint for a skill's SKILL.md frontmatter.

Checks the few things a structural error makes obvious: the frontmatter parses,
it carries only allowed keys, the name is hyphen-case and within length, and the
description is present, within bounds, and free of angle brackets (which break
the router). The allowed-key set is configurable, never baked to one provider:
pass --allow-key to extend it or --allow-keys to replace it.

Exit code is 0 when every check passes and 1 when any check fails, so a build or
CI step can gate on it. Findings print as one JSON object on stdout.

Usage:
  quick_validate.py <skill-dir-or-SKILL.md>
  quick_validate.py <path> --allow-key license --allow-key version
  quick_validate.py <path> --allow-keys name,description
  quick_validate.py <path> --max-name 64 --max-desc 1024
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_ALLOWED_KEYS = ["name", "description"]
HYPHEN_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def split_frontmatter(content: str):
    """Return (ok, frontmatter dict, error). ok is False when there is no parseable block."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return False, {}, "no frontmatter block (file does not open with ---)"
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return False, {}, "frontmatter block is not terminated with a closing ---"
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            return False, {}, f"frontmatter line is not key: value -> {line.strip()!r}"
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip()
    return True, meta, ""


def validate(content: str, allowed_keys, max_name: int, max_desc: int) -> list[dict]:
    errors: list[dict] = []

    ok, meta, parse_error = split_frontmatter(content)
    if not ok:
        return [{"check": "frontmatter", "message": parse_error}]

    extra = [k for k in meta if k not in allowed_keys]
    if extra:
        errors.append({
            "check": "allowed-keys",
            "message": f"unexpected frontmatter keys: {', '.join(sorted(extra))}; allowed: {', '.join(allowed_keys)}",
        })

    name = meta.get("name", "")
    if not name:
        errors.append({"check": "name", "message": "name is missing or empty"})
    else:
        if not HYPHEN_CASE.match(name):
            errors.append({"check": "name", "message": f"name {name!r} is not hyphen-case (lowercase, digits, single hyphens)"})
        if len(name) > max_name:
            errors.append({"check": "name", "message": f"name is {len(name)} chars, over the {max_name} limit"})

    desc = meta.get("description", "")
    if not desc:
        errors.append({"check": "description", "message": "description is missing or empty"})
    else:
        if len(desc) > max_desc:
            errors.append({"check": "description", "message": f"description is {len(desc)} chars, over the {max_desc} limit"})
        if "<" in desc or ">" in desc:
            errors.append({"check": "description", "message": "description contains angle brackets, which break router matching"})

    return errors


def resolve_skill_md(path: Path) -> Path:
    return path / "SKILL.md" if path.is_dir() else path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Structural lint for a skill's SKILL.md frontmatter")
    p.add_argument("path", type=Path, help="skill directory or a SKILL.md file")
    p.add_argument("--allow-key", action="append", default=[], help="add one key to the allowed set (repeatable)")
    p.add_argument("--allow-keys", help="comma-separated set that REPLACES the default allowed keys")
    p.add_argument("--max-name", type=int, default=64, help="max name length (default 64)")
    p.add_argument("--max-desc", type=int, default=1024, help="max description length (default 1024)")
    args = p.parse_args(argv)

    skill_md = resolve_skill_md(args.path)
    if not skill_md.is_file():
        print(json.dumps({"ok": False, "errors": [{"check": "path", "message": f"{skill_md} not found"}]}))
        return 1

    if args.allow_keys:
        allowed = [k.strip() for k in args.allow_keys.split(",") if k.strip()]
    else:
        allowed = list(DEFAULT_ALLOWED_KEYS) + list(args.allow_key)

    errors = validate(skill_md.read_text(encoding="utf-8"), allowed, args.max_name, args.max_desc)
    print(json.dumps({"ok": not errors, "file": str(skill_md), "errors": errors}))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
