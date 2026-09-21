#!/usr/bin/env python3
"""
Archive done tasks older than 48h on the ops kanban board.
Designed to run as a daily cron job. Safe to run repeatedly — only archives what's eligible.
"""
import sqlite3
import subprocess
import sys
import time

DB = "/home/kensei/.hermes/kanban/boards/ops/kanban.db"
BOARD = "ops"
CUTOFF_HOURS = 48
CUTOFF_SECONDS = CUTOFF_HOURS * 3600

def main():
    db = sqlite3.connect(DB)
    now = time.time()
    
    rows = db.execute(
        "SELECT id, title FROM tasks WHERE status='done' AND completed_at IS NOT NULL AND completed_at < ? ORDER BY completed_at ASC",
        (now - CUTOFF_SECONDS,),
    ).fetchall()
    db.close()
    
    if not rows:
        print(f"No done tasks older than {CUTOFF_HOURS}h to archive.")
        return
    
    ids = [r[0] for r in rows]
    print(f"Archiving {len(ids)} done tasks older than {CUTOFF_HOURS}h:")
    for r in rows:
        print(f"  {r[0]}  {r[1][:80]}")
    
    result = subprocess.run(
        ["hermes", "kanban", "--board", BOARD, "archive"] + ids,
        capture_output=True, text=True, timeout=30
    )
    
    if result.returncode != 0:
        print(f"Archive failed (exit {result.returncode}):", result.stderr, file=sys.stderr)
        sys.exit(1)
    
    print(result.stdout.strip())

if __name__ == "__main__":
    main()
