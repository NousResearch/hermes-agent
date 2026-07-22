#!/usr/bin/env python3
"""Seed a Hermes state.db with a session exported from a real conversation.

Usage: seed_session_db.py <state_db_path> <fixture_json_path>

Creates the database with the full SessionDB schema (if it doesn't exist)
and imports the session from the JSON fixture. Uses the real
SessionDB.import_sessions() so the data shape matches what the desktop
backend expects.
"""
import json
import sys
from pathlib import Path

# Add the repo root to sys.path so we can import hermes_state.
# The script is invoked from apps/desktop/e2e/ — repo root is ../../..
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from hermes_state import SessionDB  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <state_db_path> <fixture_json_path>", file=sys.stderr)
        sys.exit(1)

    db_path = Path(sys.argv[1])
    fixture_path = Path(sys.argv[2])

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fixture_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    db = SessionDB(db_path=db_path)
    result = db.import_sessions([session_data])

    if not result.get("ok"):
        print(f"Import failed: {result}", file=sys.stderr)
        sys.exit(1)

    imported = result.get("imported", 0)
    skipped = result.get("skipped", 0)
    errors = result.get("errors", [])

    if errors:
        print(f"Import had errors: {errors}", file=sys.stderr)
        sys.exit(1)

    print(f"Seeded {imported} session(s), skipped {skipped} → {db_path}")
    db.close()


if __name__ == "__main__":
    main()
