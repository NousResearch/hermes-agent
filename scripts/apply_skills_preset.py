#!/usr/bin/env python3
"""Apply a skills preset: enable whitelist + always_keep rules; disable the rest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
_PRESETS = Path(__file__).resolve().parent / "skills_presets"


def _load_preset(name: str) -> dict:
    path = _PRESETS / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"Unknown preset: {name} (no file {path})")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _kdense_frontmatter_names() -> set[str]:
  """Names from SKILL.md under research/kdense-* directories."""
  import re
  from pathlib import Path

  names: set[str] = set()
  root = Path.home() / ".hermes" / "skills" / "research"
  for skill_md in root.glob("kdense-*/SKILL.md"):
    m = re.search(r"^name:\s*(.+)$", skill_md.read_text(encoding="utf-8")[:8000], re.M)
    if m:
      names.add(m.group(1).strip())
  return names


_KDENSE_NAMES: set[str] | None = None


def _should_keep(skill: dict, preset: dict) -> bool:
    global _KDENSE_NAMES
    name = skill["name"]
    enabled = set(preset.get("enabled") or [])
    if name in enabled:
        return True
    if preset.get("include_kdense_bundle"):
        if _KDENSE_NAMES is None:
            _KDENSE_NAMES = _kdense_frontmatter_names()
        if name in _KDENSE_NAMES:
            return True
    for prefix in preset.get("always_keep_prefixes") or []:
        if name.startswith(prefix):
            return True
    for sub in preset.get("always_keep_substrings") or []:
        if sub in name:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Hermes skills preset")
    parser.add_argument(
        "preset",
        choices=["daily", "dev", "finance", "science"],
        help="Preset name (daily=Feishu+Kanban+light dev, recommended)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print counts only")
    parser.add_argument("--platform", default=None, help="Platform key for platform_disabled")
    args = parser.parse_args()

    sys.path.insert(0, str(_REPO))
    from hermes_cli.config import load_config, save_config
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    from tools.skills_tool import _find_all_skills

    preset = _load_preset(args.preset)
    skills = _find_all_skills(skip_disabled=True)
    if not skills:
        print("No skills found.", file=sys.stderr)
        return 1

    new_disabled = {s["name"] for s in skills if not _should_keep(s, preset)}
    enabled_count = len(skills) - len(new_disabled)

    print(f"Preset: {args.preset} — {preset.get('description', '')}")
    print(f"Skills on disk: {len(skills)}")
    print(f"Will enable: {enabled_count}, disable: {len(new_disabled)}")

    if args.dry_run:
        return 0

    config = load_config()
    save_disabled_skills(config, new_disabled, args.platform)
    print(f"✓ Saved skills.disabled ({len(new_disabled)} disabled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
