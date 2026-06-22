#!/usr/bin/env python3
"""One-shot structural cleanup: strip profile names from `tasks.skills` in kanban.db.

Symptoms this fixes:
- Workers crash-loop with "Unknown skill(s): code-craftsman" or similar
- Each crash spawns a new PID, hits failure_limit, auto-blocks
- Dispatcher re-queues, repeats forever

Root cause: orchestrator-side templates copied the assignee name into
the skills list. profile names like "code-craftsman" are not valid
skills and crash the worker at CLI startup.

Run: python3 scripts/fix_skills_profile_leak.py [--dry-run]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "kanban.db"),
                        help="Path to kanban.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"kanban.db not found at {db_path}", file=sys.stderr)
        return 1

    profiles_dir = Path.home() / ".hermes" / "profiles"
    if not profiles_dir.is_dir():
        print(f"profiles dir not found at {profiles_dir}", file=sys.stderr)
        return 1

    known_profiles = {p.name for p in profiles_dir.iterdir() if p.is_dir()}
    # Also exclude known non-skill internal names that sometimes leak
    known_profiles |= {"kanban-worker"}

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT id, status, skills FROM tasks WHERE skills IS NOT NULL AND skills != ''"
    ).fetchall()

    fixed = []
    for row in rows:
        try:
            skill_list = json.loads(row["skills"]) if row["skills"] else []
        except json.JSONDecodeError:
            continue
        new_skills = [s for s in skill_list if s not in known_profiles]
        if new_skills != skill_list:
            stripped = sorted(set(skill_list) - set(new_skills))
            fixed.append((row["id"], row["status"], skill_list, new_skills, stripped))

    if not fixed:
        print("No profile-name leaks found. Nothing to fix.")
        return 0

    print(f"Found {len(fixed)} tasks with profile-name leaks:\n")
    for id_, status, before, after, stripped in fixed:
        print(f"  {id_} [{status}]: strip {stripped}")
        print(f"    before: {before}")
        print(f"    after:  {after}")

    if args.dry_run:
        print("\nDry run — no changes written.")
        return 0

    for id_, status, before, after, stripped in fixed:
        new_json = json.dumps(after) if after else None
        db.execute("UPDATE tasks SET skills = ? WHERE id = ?", (new_json, id_))
    db.commit()
    print(f"\nFixed {len(fixed)} tasks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
