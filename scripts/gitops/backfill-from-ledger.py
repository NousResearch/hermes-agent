#!/usr/bin/env python3
"""
KENSEI GitOps — backfill historical commits from governance/profile-change-ledger.md.

Parses the two ledger sections:
  1. Main table (lines under "## Format"): | Date | Profile | Trigger | Change | Follow-up | Outcome |
  2. Register (lines under "## Register (chronological...)"): | <date> | <profile> | <change> | <follow-up> | <outcome> |

For each parsed row, creates an empty commit with metadata in the commit message
and a date close to the original change date. Adds a `BACKFILL.md` marker so we
can tell backfill commits apart from real ones.

Idempotent: skips rows whose date+profile already have a backfill commit.

Usage:
    python3 backfill-from-ledger.py [--ledger PATH] [--repo PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class LedgerEntry:
    raw_date: str
    iso_date: str
    profile: str
    trigger: str
    change: str
    follow_up: str
    outcome: str
    source_section: str


_ISO_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{2})$")
_TABLE_SEP = re.compile(r"^\|[\s\-|]+\| *$")
_TABLE_CELL = re.compile(r"\s*\|\s*")
_HEADER_KEYWORDS = ("Date", "Profile", "Trigger")  # main-table header


def _to_iso(raw: str) -> str | None:
    """Convert DD/MM/YY[ HH:MM] or DD/MM/YYYY[ HH:MM] → YYYY-MM-DD. Returns None if unparseable."""
    raw = raw.strip()
    # Strip time suffix if present (e.g. "03/06/2026 17:06" → "03/06/2026")
    raw = raw.split()[0] if " " in raw else raw
    m = _ISO_RE.match(raw)
    if not m:
        # Try DD/MM/YYYY
        m2 = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", raw)
        if not m2:
            return None
        d, mo, y = m2.groups()
    else:
        d, mo, y = m.groups()
        y = "20" + y if int(y) < 70 else "19" + y
    try:
        return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _clean(cell: str) -> str:
    return cell.strip().strip("`").strip()


def parse_ledger(path: Path) -> list[LedgerEntry]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    entries: list[LedgerEntry] = []

    section = "preamble"
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            section = stripped.lower()
            continue

        if not stripped.startswith("|"):
            continue
        if _TABLE_SEP.match(stripped):
            continue

        cells = [c for c in _TABLE_CELL.split(stripped.strip("|"))]

        # Skip header rows (main table only — the register has no header).
        if section == "## format" and any(kw in (cells[0] if cells else "") for kw in _HEADER_KEYWORDS):
            continue
        # Belt-and-braces: any row whose first cell is literally "Date".
        if cells and cells[0].strip() == "Date":
            continue

        if len(cells) < 4:
            continue

        if section.startswith("## register"):
            # Format: | date | profile | change | follow-up | outcome |
            # (no Trigger column)
            raw_date, profile, change, follow_up, outcome = (cells + ["", "", "", "", ""])[:5]
            trigger = "(register)"
        else:
            # Main table: | Date | Profile | Trigger | Change | Follow-up | Outcome |
            if len(cells) < 6:
                continue
            raw_date, profile, trigger, change, follow_up, outcome = cells[:6]

        raw_date = _clean(raw_date)
        if not raw_date or raw_date.startswith("||") or raw_date == "Date":
            continue

        iso = _to_iso(raw_date)
        if iso is None:
            continue

        entries.append(
            LedgerEntry(
                raw_date=raw_date,
                iso_date=iso,
                profile=_clean(profile),
                trigger=_clean(trigger),
                change=_clean(change),
                follow_up=_clean(follow_up),
                outcome=_clean(outcome),
                source_section=section,
            )
        )

    return entries


def existing_backfill_marker(repo: Path) -> set[str]:
    """Return the set of unique date+profile keys already backfilled."""
    res = subprocess.run(
        ["git", "log", "--grep=^BACKFILL:", "--format=%s"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    seen: set[str] = set()
    for line in res.stdout.splitlines():
        if line.startswith("BACKFILL:"):
            seen.add(line)
    return seen


def commit_one(repo: Path, entry: LedgerEntry, dry_run: bool) -> bool:
    key = f"BACKFILL: {entry.iso_date} | {entry.profile}"
    if key in existing_backfill_marker(repo):
        return False

    msg = (
        f"{key}\n\n"
        f"Source: governance/profile-change-ledger.md ({entry.source_section.strip('# ').strip()})\n"
        f"Raw date: {entry.raw_date}\n"
        f"Trigger: {entry.trigger}\n"
        f"Change: {entry.change}\n"
        f"Follow-up: {entry.follow_up}\n"
        f"Outcome (at ledger time): {entry.outcome}\n\n"
        f"This is a backfill commit. The original change is recorded in the\n"
        f"profile-change-ledger.md; this commit restores the audit trail as\n"
        f"git history. The repository state at this date is NOT reconstructed\n"
        f"from the ledger — only the metadata is preserved."
    )

    if dry_run:
        print(f"DRY: would commit {entry.iso_date} | {entry.profile}")
        return True

    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "KENSEI (backfill)",
            "envGIT_AUTHOR_EMAIL": "kensei-backfill@local",
            "GIT_COMMITTER_NAME": "KENSEI (backfill)",
            "GIT_COMMITTER_EMAIL": "kensei-backfill@local",
        }
    )
    env["GIT_AUTHOR_EMAIL"] = "kensei-backfill@local"
    env["GIT_COMMITTER_EMAIL"] = "kensei-backfill@local"

    # Date-aware commit via env. Use noon UTC to avoid timezone edge cases.
    env["GIT_AUTHOR_DATE"] = f"{entry.iso_date}T12:00:00+00:00"
    env["GIT_COMMITTER_DATE"] = f"{entry.iso_date}T12:00:00+00:00"

    res = subprocess.run(
        ["git", "commit", "--allow-empty", "-q", "-m", msg],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"FAIL: {entry.iso_date} | {entry.profile}: {res.stderr}", file=sys.stderr)
        return False
    print(f"OK:   {entry.iso_date} | {entry.profile}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", default="/home/kensei/.hermes/governance/profile-change-ledger.md")
    p.add_argument("--repo", default="/home/kensei/.hermes")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ledger = Path(args.ledger)
    repo = Path(args.repo)
    if not ledger.exists():
        print(f"ledger not found: {ledger}", file=sys.stderr)
        return 1
    if not (repo / ".git").is_dir():
        print(f"not a git repo: {repo}", file=sys.stderr)
        return 1

    entries = parse_ledger(ledger)
    print(f"parsed {len(entries)} entries from {ledger.name}")

    created = 0
    for entry in entries:
        if commit_one(repo, entry, args.dry_run):
            created += 1

    print(f"created {created} backfill commit(s) (dry-run={args.dry_run})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
