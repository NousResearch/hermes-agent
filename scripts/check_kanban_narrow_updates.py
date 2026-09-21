#!/usr/bin/env python3
"""Advisory lint: narrow UPDATEs on the kanban task ledger.

A lost ``WHERE`` on ``tasks`` flips every card on the board; a status flip
without a status predicate lets two writers (dispatcher tick vs CLI operator
vs gateway dashboard) silently overwrite each other's transitions. Every
pattern below is a call-site shape that once caused — or could cause — a
cross-task or cross-transition clobber. The invariant:

1. Every ``UPDATE tasks ...`` names one row: ``WHERE id = ?``.
2. Every ``UPDATE tasks ...`` that assigns ``status`` additionally carries a
   status predicate in the ``WHERE`` clause (``status = ...`` /
   ``status IN (...)`` / ``status != ...``) so the write is a
   Compare-And-Swap on the transition the caller actually validated.
3. Every status transition checks the cursor ``rowcount`` (``!= 1`` = lost
   race: bail, do NOT emit the event).

Rules 1–2 are enforced syntactically here; rule 3 is a reviewer checklist —
a static scan cannot tell a checked cursor from a dropped one, so each
finding below names the transition and the reviewer confirms the
``rowcount`` handling by eye.

Advisory by construction: prints ``file:line  <rule>  why`` for every hit
and ALWAYS exits 0, because dynamic-``SET`` sites (``specify_task`` builds
its ``SET`` list at runtime) and fixture/test writes are legitimate. The
reviewer reads each finding against the transition it guards.

What this scan does NOT cover (documented limits, not gaps to file):

* Dynamic ``SET`` lists: the ``UPDATE`` chunk carries no literal
  ``status =``, so rule 2 cannot fire — the reviewer verifies the built
  list plus its ``WHERE`` by hand (today: ``specify_task``,
  ``SET status='todo' ... WHERE id = ? AND status = 'triage'`` + rowcount).
* ``rowcount`` handling (rule 3): reviewer checklist, see above.
* ``tests/`` fixture writes: out of scope on purpose — fixtures reset the
  whole board by design. Only ``hermes_cli/kanban*.py`` is scanned.

Companion rule (same card, documented here because it shares the motive —
keep per-tick I/O bounded on the 9.8 MB board): ``sqlite3.Connection.backup()``
snapshots belong to repairs, migrations, and manual unblocks — NEVER to the
dispatch tick. A per-tick snapshot serialises every dispatcher on WAL
frames and turns the single-writer lock into a full-board stall. If you add
a ``.backup(`` call under ``hermes_cli/kanban*``, say in its comment which
of the three sanctioned call-sites it is.

Usage:
    python scripts/check_kanban_narrow_updates.py [--files a.py ...]
    python scripts/check_kanban_narrow_updates.py --base origin/main --head HEAD
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = sorted((ROOT / "hermes_cli").glob("kanban*.py"))

# Whole-file exemptions (bulk writers, not single-task transitions):
# * kanban_db_connect.py — schema-migration backfills (ADD COLUMN then
#   ``UPDATE tasks SET <new> = <legacy>`` over every row, by design).
# * kanban_transfer.py — board import/export rewrites the whole task table
#   by design (it snapshots via ``Connection.backup()`` first — the one
#   sanctioned repair/migration-class backup site outside ``repair``).
_EXEMPT_FILES = frozenset({"kanban_db_connect.py", "kanban_transfer.py"})

_UPDATE_RE = re.compile(r"UPDATE\s+tasks\s+SET\b")
_WHERE_ID_RE = re.compile(r"WHERE\s+id\s*=\s*\?")
_STATUS_ASSIGN_RE = re.compile(r"\bstatus\s*=")
# A status predicate in the WHERE clause: equality, membership, or explicit
# exclusion (``status != 'archived'`` in archive_task is a deliberate
# idempotence guard, still a transition CAS).
_STATUS_PREDICATE_RE = re.compile(r"\bstatus\s*(!=|=|\bIN\b|\bIS\b)", re.IGNORECASE)
_CHUNK_CAP = 3000  # chars past the match: one statement, never the whole file


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    detail: str


def _changed_files(base: str, head: str) -> list[Path]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    files = []
    for line in out.stdout.splitlines():
        p = (ROOT / line.strip()).resolve()
        if p.name.startswith("kanban") and p.suffix == ".py" and p.is_file():
            try:
                p.relative_to(ROOT / "hermes_cli")
            except ValueError:
                continue
            files.append(p)
    return sorted(files)


def _scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[Finding] = []
    matches = list(_UPDATE_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[m.start():min(end, m.start() + _CHUNK_CAP)]
        line = text.count("\n", 0, m.start()) + 1
        rel = str(path.relative_to(ROOT))
        if not _WHERE_ID_RE.search(chunk):
            findings.append(Finding(
                rel, line, "missing-WHERE-id",
                "UPDATE tasks without `WHERE id = ?` — must name exactly one row.",
            ))
            continue
        where_at = chunk.find("WHERE")
        set_part, where_part = chunk[:where_at], chunk[where_at:]
        if _STATUS_ASSIGN_RE.search(set_part) and not _STATUS_PREDICATE_RE.search(where_part):
            findings.append(Finding(
                rel, line, "status-without-predicate",
                "assigns `status` but the WHERE clause carries no status "
                "predicate — add `AND status = .../IN (...)` + a rowcount check.",
            ))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--files", nargs="*", help="Files to scan (default: hermes_cli/kanban*.py).")
    ap.add_argument("--base", default=None, help="Diff base for --head (e.g. origin/main).")
    ap.add_argument("--head", default="HEAD", help="Diff head (requires --base).")
    args = ap.parse_args(argv)
    if args.files:
        targets = [Path(f) if Path(f).is_absolute() else (ROOT / f) for f in args.files]
    elif args.base:
        targets = _changed_files(args.base, args.head)
    else:
        targets = DEFAULT_TARGETS
    findings: list[Finding] = []
    for target in targets:
        if target.name in _EXEMPT_FILES:
            continue
        findings.extend(_scan_file(target))
    for f in findings:
        print(f"{f.path}:{f.line}  {f.rule}  {f.detail}")
    if findings:
        print(f"{len(findings)} narrow-UPDATE finding(s) — advisory, reviewer confirms "
              "each transition's rowcount handling by eye.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
