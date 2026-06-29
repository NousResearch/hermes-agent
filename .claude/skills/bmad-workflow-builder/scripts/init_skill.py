#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# ///
"""init_skill — deterministic scaffolder for a new skill.

Creates the skill directory and writes SKILL.md from the builder's template, which
carries the embedded archetype guidance and the delete-when-done marker. The name
is normalized to hyphen-case and capped at 64 chars. Only the resource directories
the build flow asked for are stubbed, so the skill starts as small as it can. A
customize.toml is emitted only when customization was accepted, never by default.

This script does the mechanical scaffolding so the model spends its turns on the
content, not on mkdir and string substitution.

Usage:
  init_skill.py --name "My New Skill" --dest /path/to/skills
  init_skill.py --name foo --dest DIR --dirs references,scripts,assets
  init_skill.py --name foo --dest DIR --customizable
  init_skill.py --name foo --dest DIR \
      --template /abs/SKILL-template.md --customize-template /abs/customize-template.toml

Output: one JSON object on stdout describing what was created.
Exit code 0 on success, 1 on failure (e.g. the target already exists).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

KNOWN_DIRS = ("references", "scripts", "assets", "agents")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = SCRIPT_DIR.parent / "assets" / "SKILL-template.md"
DEFAULT_CUSTOMIZE = SCRIPT_DIR.parent / "assets" / "customize-template.toml"


def normalize_name(raw: str, max_len: int = 64) -> str:
    """Lowercase, collapse non-alphanumerics to single hyphens, trim, cap at max_len."""
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s


def fill_template(template: str, skill_name: str) -> str:
    return template.replace("{skill-name}", skill_name)


def scaffold(args) -> dict:
    skill_name = normalize_name(args.name)
    if not skill_name:
        raise ValueError(f"name {args.name!r} normalized to an empty string")

    skill_dir = Path(args.dest) / skill_name
    if skill_dir.exists():
        raise FileExistsError(f"{skill_dir} already exists")

    template_path = Path(args.template) if args.template else DEFAULT_TEMPLATE
    if not template_path.is_file():
        raise FileNotFoundError(f"template not found: {template_path}")

    requested = []
    for d in (args.dirs or "").split(","):
        d = d.strip()
        if not d:
            continue
        if d not in KNOWN_DIRS:
            raise ValueError(f"unknown resource dir {d!r}; known: {', '.join(KNOWN_DIRS)}")
        requested.append(d)

    skill_dir.mkdir(parents=True)
    created = [str(skill_dir)]

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(fill_template(template_path.read_text(encoding="utf-8"), skill_name), encoding="utf-8")
    created.append(str(skill_md))

    for d in requested:
        sub = skill_dir / d
        sub.mkdir()
        created.append(str(sub))

    customize_emitted = False
    if args.customizable:
        ct_path = Path(args.customize_template) if args.customize_template else DEFAULT_CUSTOMIZE
        if not ct_path.is_file():
            raise FileNotFoundError(f"customize template not found: {ct_path}")
        target = skill_dir / "customize.toml"
        target.write_text(
            ct_path.read_text(encoding="utf-8").replace("{skill-name}", skill_name),
            encoding="utf-8",
        )
        created.append(str(target))
        customize_emitted = True

    return {
        "ok": True,
        "skill_name": skill_name,
        "skill_dir": str(skill_dir),
        "dirs_stubbed": requested,
        "customize_toml": customize_emitted,
        "created": created,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Deterministic scaffolder for a new skill")
    p.add_argument("--name", required=True, help="raw skill name; normalized to hyphen-case <=64")
    p.add_argument("--dest", required=True, help="parent directory the skill folder is created under")
    p.add_argument("--dirs", default="", help="comma-separated resource dirs to stub (references,scripts,assets,agents)")
    p.add_argument("--customizable", action="store_true", help="emit customize.toml (only when customization was accepted)")
    p.add_argument("--template", help="override path to the SKILL.md template")
    p.add_argument("--customize-template", help="override path to the customize.toml template")
    args = p.parse_args(argv)

    try:
        result = scaffold(args)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
