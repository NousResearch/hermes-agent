#!/usr/bin/env python3
"""Monitor SQLite WAL file sizes across all Hermes state databases.

Runs every 6 hours. If any WAL exceeds 50MB, does a TRUNCATE checkpoint
and reports. Reports OK when all WALs are small.

Designed for no_agent=True cron delivery: empty stdout = all clear.
"""
import sqlite3
import sys
import time
from pathlib import Path

# Databases to monitor — (label, path)
DATABASES = [
    ("root state.db", Path("/home/kensei/.hermes/state.db")),
    ("mnemosyne", Path("/home/kensei/.hermes/mnemosyne/data/mnemosyne.db")),
    ("root kanban", Path("/home/kensei/.hermes/kanban/boards/ops/kanban.db")),
    ("root kanban main", Path("/home/kensei/.hermes/kanban.db")),
]

# Profile state DBs
PROFILE_DIR = Path("/home/kensei/.hermes/profiles")
if PROFILE_DIR.exists():
    for p in sorted(PROFILE_DIR.iterdir()):
        if p.is_dir() and (p / "state.db").exists():
            DATABASES.append((f"profile {p.name}", p / "state.db"))

WAL_THRESHOLD_MB = 50
WAL_CHECKPOINT_THRESHOLD_MB = 10  # Also do TRUNCATE if WAL > 10MB


def format_size(b: int) -> str:
    mb = b / (1024 * 1024)
    return f"{mb:.1f}MB"


def truncate_checkpoint(path: Path) -> tuple[bool, str]:
    """Run PRAGMA wal_checkpoint(TRUNCATE) on the DB. Returns (ok, result_msg)."""
    try:
        con = sqlite3.connect(str(path), timeout=10)
        try:
            result = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            # result = (return_code, pages_in_wal, pages_checkpointed)
            if result and result[0] == 0:
                return True, f"checkpoint ok ({result[1]}→{result[2]} pages)"
            return True, f"checkpoint rc={result[0]} ({result[1]}→{result[2]} pages)"
        finally:
            con.close()
    except Exception as e:
        return False, str(e)


def main():
    issues = []
    actions = []
    for label, path in DATABASES:
        if not path.exists():
            continue
        wal_path = path.parent / (path.name + "-wal")
        if not wal_path.exists():
            continue
        wal_size = wal_path.stat().st_size
        db_size = path.stat().st_size
        wal_mb = wal_size / (1024 * 1024)
        db_mb = db_size / (1024 * 1024)

        if wal_mb < WAL_CHECKPOINT_THRESHOLD_MB:
            continue  # Below report threshold

        if wal_mb >= WAL_THRESHOLD_MB:
            # TRUNCATE checkpoint
            ok, msg = truncate_checkpoint(path)
            new_wal = wal_path.stat().st_size if wal_path.exists() else 0
            new_mb = new_wal / (1024 * 1024)
            if ok:
                actions.append(
                    f"🟡 {label}: WAL was {wal_mb:.0f}MB (DB {db_mb:.0f}MB) → "
                    f"TRUNCATE checkpoint done, now {new_mb:.2f}MB"
                )
            else:
                issues.append(
                    f"🔴 {label}: WAL {wal_mb:.0f}MB (DB {db_mb:.0f}MB) — "
                    f"checkpoint FAILED: {msg}"
                )
        elif wal_mb >= WAL_CHECKPOINT_THRESHOLD_MB:
            # Above report threshold but below action threshold — just note it
            issues.append(
                f"⚠️ {label}: WAL {wal_mb:.1f}MB (DB {db_mb:.1f}MB) — "
                f"above {WAL_CHECKPOINT_THRESHOLD_MB}MB threshold, below {WAL_THRESHOLD_MB}MB action threshold"
            )

    if issues or actions:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
        lines = [f"SQLite WAL check · {timestamp}"]
        if actions:
            lines.append("")
            lines.append("Actions taken:")
            lines.extend(f"  {a}" for a in actions)
        if issues:
            lines.append("")
            lines.append("Observations:")
            lines.extend(f"  {i}" for i in issues)
        print("\n".join(lines))
        return

    # Silent — nothing to report (no_agent=True cron pattern)


if __name__ == "__main__":
    main()
