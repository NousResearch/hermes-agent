#!/usr/bin/env python3
"""Three-state SQLite integrity probe for the desktop pre-update guard (#68474).

Usage:
  python scripts/db_integrity_probe.py <path-to-db>

Always exits 0 when it manages to run at all — the caller (Electron's main
process) tells "checked, and it's fine/damaged" apart from "could not check"
by reading stdout, not the exit code. A non-zero exit means the interpreter
or script itself blew up (missing argv, unreadable interpreter), which the
caller treats identically to "could not check".

Stdout is exactly one of:
  ok                        -- PRAGMA integrity_check found nothing wrong
  <row>\\n<row>\\n...        -- corruption found; every row is preserved
  UNVERIFIED: <reason>       -- couldn't get a clean read (see below)

Deliberately never falls back to a read-write connection. This probe runs
while the backend may still hold state.db open, so a second RW connection
here would recreate the exact hazard — two writers on a live database —
behind the 2026-08-21 corruption incident it exists to catch. If the
read-only open fails (e.g. WAL needs a -shm this connection isn't allowed
to create), the honest verdict is "unverified", not a risky RW retry.
"""
import pathlib
import sqlite3
import sys


def probe(db_path: str) -> str:
    p = pathlib.Path(db_path)
    conn = None

    try:
        conn = sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True, timeout=5)
        rows = [str(r[0]) for r in conn.execute("PRAGMA integrity_check")]
    except sqlite3.OperationalError as exc:
        return f"UNVERIFIED: read-only open failed: {exc}"
    finally:
        if conn is not None:
            conn.close()

    if not rows:
        return "UNVERIFIED: integrity_check returned no rows"

    return "\n".join(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print("UNVERIFIED: usage: db_integrity_probe.py <path-to-db>")
        return 0

    print(probe(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
