#!/usr/bin/env python3
"""Hermes session title normalizer.

Two modes:
  list   — print unnormalized sessions (title NULL or not matching the format
           prefix) with a first-user-message excerpt for naming context.
  apply  — take a JSON plan ([{"id": ..., "title": ...}, ...]), precheck for
           conflicts, back up the old titles, rename via the hermes CLI, and
           verify by reading back.

Both modes open state.db read-only. Nothing here writes to the database
directly — renames go through `hermes sessions rename`, the sanctioned path.

Usage:
  retitle.py list  [--db PATH] [--prefix '📅']
  retitle.py apply [--db PATH] plan.json [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

DEFAULT_DB = Path.home() / ".hermes" / "state.db"
DEFAULT_PREFIX = "\U0001F4C5"  # 📅

# Machine-generated first messages to skip when hunting for a naming anchor.
SKIP_PREFIXES = ("[IMPORTANT", "[CONTEXT COMPACTION", "[Note: model", "[PRIOR CONTEXT")


def strip_skip_prefixes(text: str) -> str:
    """Drop machine-generated lead-ins so the real user message is visible."""
    out = text.strip()
    for _ in range(4):  # a few stacked banners happen
        for p in SKIP_PREFIXES:
            if out.startswith(p):
                nl = out.find("\n")
                out = out[nl + 1:].strip() if nl != -1 else ""
                break
        else:
            return out
    return out


def is_normalized(title: str | None, prefix: str = DEFAULT_PREFIX) -> bool:
    """A title counts as normalized when it starts with the format prefix."""
    return bool(title) and title.startswith(prefix)


def build_title(date_mmdd: str, emoji: str, kind: str, topic: str) -> str:
    """Compose the target title: '<prefix> MMDD | <emoji> <type> | <topic>'."""
    return f"{DEFAULT_PREFIX} {date_mmdd} | {emoji} {kind} | {topic}"


def parse_title(title: str) -> dict | None:
    """Parse a normalized title back into its parts (None if it doesn't match)."""
    m = re.match(rf"{re.escape(DEFAULT_PREFIX)}\s+(\d{{4}})\s*\|\s*(\S+)\s+(\S+)\s*\|\s*(.+)$", title)
    if not m:
        return None
    return {"date": m.group(1), "emoji": m.group(2), "type": m.group(3), "topic": m.group(4).strip()}


def precheck_plan(plan: list[dict], existing: set[str]) -> list[str]:
    """Return conflict descriptions; empty list means the plan is safe.

    Checks both internal duplication and collisions with titles already in the
    database (title carries a UNIQUE index, so a collision would hard-fail).
    """
    problems: list[str] = []
    seen: set[str] = set()
    for item in plan:
        t = item.get("title", "")
        if t in seen:
            problems.append(f"internal duplicate: {t}")
        seen.add(t)
        if t in existing:
            problems.append(f"collides with existing title: {t}")
    return problems


def open_ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def cmd_list(args: argparse.Namespace) -> int:
    con = open_ro(Path(args.db))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, title, message_count FROM sessions "
        "WHERE title IS NULL OR title NOT LIKE ? ORDER BY started_at",
        (args.prefix + "%",),
    ).fetchall()
    if not rows:
        print("All sessions normalized.")
        return 0
    for row in rows:
        first = con.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
            "AND content IS NOT NULL ORDER BY timestamp LIMIT 1",
            (row["id"],),
        ).fetchone()
        anchor = strip_skip_prefixes(first["content"])[:80].replace(chr(10), " ") if first else ""
        print(f"{row['id']} | {row['message_count']:>4} msgs | {row['title']} | {anchor}")
    print(f"\n{len(rows)} unnormalized session(s).")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text())
    con = open_ro(Path(args.db))
    existing = {r[0] for r in con.execute("SELECT title FROM sessions WHERE title IS NOT NULL")}

    problems = precheck_plan(plan, existing)
    if problems:
        print("Plan rejected:")
        for p in problems:
            print("  -", p)
        return 1

    backup = []
    for item in plan:
        old = con.execute("SELECT title FROM sessions WHERE id = ?", (item["id"],)).fetchone()
        backup.append({"id": item["id"], "old": old[0] if old else None, "new": item["title"]})
    if args.dry_run:
        print(json.dumps(backup, ensure_ascii=False, indent=2))
        return 0

    failed = []
    for item in plan:
        proc = subprocess.run(
            ["hermes", "sessions", "rename", item["id"], item["title"]],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            failed.append((item["id"], (proc.stderr or proc.stdout).strip()[:120]))

    verified, unverified = 0, []
    for item in plan:
        row = con.execute("SELECT title FROM sessions WHERE id = ?", (item["id"],)).fetchone()
        if row and row[0] == item["title"]:
            verified += 1
        else:
            unverified.append(item["id"])

    print(json.dumps({"renamed": len(plan) - len(failed), "failed": failed,
                      "verified": verified, "unverified": unverified,
                      "backup": backup}, ensure_ascii=False, indent=2))
    return 0 if not failed and not unverified else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p_apply = sub.add_parser("apply")
    p_apply.add_argument("plan")
    p_apply.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return cmd_list(args) if args.cmd == "list" else cmd_apply(args)


if __name__ == "__main__":
    sys.exit(main())
