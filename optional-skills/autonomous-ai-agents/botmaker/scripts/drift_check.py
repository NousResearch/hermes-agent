#!/usr/bin/env python3
"""Drift tripwire for the botmaker doc layer (one home per rule).

Run after every mint and every doc patch:

    python3 ~/.hermes/skills/autonomous-ai-agents/botmaker/scripts/drift_check.py

Exit 0 = clean. Nonzero = drift; each violation prints on its own line.

Checks:
  1. Owner-only markers: mechanics phrases live only in their owner file.
  2. Forbidden strings (frozen enumerations, retired premises) are absent
     from the live scope (skill tree + botmaker profile SOUL/memories).
  3. The specialist roster table in the vault's making-bots.md matches the
     profiles that actually exist under ~/.hermes/profiles/. Point the
     BOTMAKER_VAULT_GUIDE env var at your vault's Bots/making-bots.md;
     the check is skipped (with a warning) if it is not set.
  4. Botmaker's profile skill links are file-level symlinks and rglob
     discovery still finds SKILL.md — and no profile in the fleet has a
     directory symlink anywhere under its skills/ tree (invisible to
     Python 3.11 rglob).

Reads only documentation files; never reads config values or .env files.
Extend OWNER_ONLY / FORBIDDEN as your fleet earns its own rules.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HOME = Path.home()
SKILL_TREE = HOME / ".hermes/skills/autonomous-ai-agents/botmaker"
PROFILE = HOME / ".hermes/profiles/botmaker"
PROFILES_DIR = HOME / ".hermes/profiles"
VAULT_GUIDE = Path(os.environ["BOTMAKER_VAULT_GUIDE"]) if os.environ.get("BOTMAKER_VAULT_GUIDE") else None

SKILL_DOCS = [
    SKILL_TREE / "SKILL.md",
    SKILL_TREE / "references/process.md",
    SKILL_TREE / "references/soul-craft.md",
    SKILL_TREE / "references/vault.md",
]
PROFILE_DOCS = [
    PROFILE / "SOUL.md",
    PROFILE / "memories/MEMORY.md",
    PROFILE / "memories/USER.md",
]

# Phrase -> the one file (relative to the skill tree) allowed to contain it.
OWNER_ONLY = {
    "does not descend": "references/process.md",
    "config unset model.base_url": "references/process.md",
    "which -a NAME": "references/process.md",
}
# SKILL.md keeps the two preflight commands as its Procedure C stop point.
OWNER_ONLY_EXTRA_ALLOWED = {"which -a NAME": ["SKILL.md"]}

# Frozen enumerations and retired premises. None of these may appear in the
# live scope; vault history (changelog, training stories) is out of scope.
FORBIDDEN = [
    "already have it",
    "already done",
    "will prompt your human",
]

ROSTER_ROW = re.compile(r"^\|\s*`@[\w-]+`\s*\|\s*`([\w-]+)`\s*\|")


def main() -> int:
    errors: list[str] = []

    docs: dict[Path, str] = {}
    for path in SKILL_DOCS:
        if not path.exists():
            errors.append(f"missing file: {path}")
            continue
        docs[path] = path.read_text(encoding="utf-8")
    for path in PROFILE_DOCS:
        if path.exists():
            docs[path] = path.read_text(encoding="utf-8")

    # 1. Owner-only markers.
    for phrase, owner_rel in OWNER_ONLY.items():
        allowed = {owner_rel, *OWNER_ONLY_EXTRA_ALLOWED.get(phrase, [])}
        found_in_owner = False
        for path, text in docs.items():
            if phrase not in text:
                continue
            try:
                rel = str(path.relative_to(SKILL_TREE))
            except ValueError:
                rel = str(path)
            if rel in allowed:
                found_in_owner = found_in_owner or rel == owner_rel
            else:
                errors.append(f"marker escaped its owner: {phrase!r} in {path}")
        if not found_in_owner:
            errors.append(f"marker missing from owner {owner_rel}: {phrase!r}")

    # 2. Forbidden strings in the live scope.
    for phrase in FORBIDDEN:
        for path, text in docs.items():
            for i, line in enumerate(text.splitlines(), 1):
                if phrase in line:
                    errors.append(f"forbidden {phrase!r} at {path}:{i}")

    # 3. Roster table vs profiles on disk.
    if VAULT_GUIDE and VAULT_GUIDE.exists():
        rostered = {
            m.group(1)
            for line in VAULT_GUIDE.read_text(encoding="utf-8").splitlines()
            if (m := ROSTER_ROW.match(line))
        }
        on_disk = {
            p.name
            for p in PROFILES_DIR.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        } if PROFILES_DIR.is_dir() else set()
        for name in sorted(on_disk - rostered):
            errors.append(f"profile exists but has no roster row: {name}")
        for name in sorted(rostered - on_disk):
            errors.append(f"roster row names a missing profile: {name}")
    else:
        print("note: BOTMAKER_VAULT_GUIDE not set (or file missing) — roster check skipped")

    # 4a. Botmaker's own links: file-level symlinks, discovery intact.
    skills_dir = PROFILE / "skills"
    if skills_dir.is_dir():
        link = skills_dir / "autonomous-ai-agents/botmaker/SKILL.md"
        if link.exists() and not link.is_symlink():
            errors.append(f"expected file-level symlink, found otherwise: {link}")
        discovered = {p.parent.name for p in skills_dir.rglob("SKILL.md")}
        if "botmaker" not in discovered:
            errors.append("rglob discovery lost skill: botmaker")

    # 4b. Whole fleet: a directory symlink anywhere under a profile's skills/
    #     is invisible to rglob — the class defect, checked on every profile.
    if PROFILES_DIR.is_dir():
        for prof in sorted(PROFILES_DIR.iterdir()):
            if not prof.is_dir() or prof.name.startswith("."):
                continue
            prof_skills = prof / "skills"
            if not prof_skills.is_dir():
                continue
            for path in prof_skills.rglob("*"):
                if path.is_dir() and path.is_symlink():
                    errors.append(f"directory symlink (invisible to rglob): {path}")

    if errors:
        print(f"DRIFT: {len(errors)} violation(s)")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("drift_check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
