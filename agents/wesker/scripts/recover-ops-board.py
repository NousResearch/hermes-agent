#!/usr/bin/env python3
"""Recover ops kanban board from the cleanest available backup."""
import sqlite3, shutil, os, sys

backup = "/home/kensei/.hermes/kanban/boards/ops/kanban.db.corrupt.51d9f20e3d8348d7.bak"
new_db = "/home/kensei/.hermes/kanban/boards/ops/kanban.db.recovered"

shutil.copy2(backup, new_db)
conn = sqlite3.connect(new_db)
conn.execute("PRAGMA journal_mode=DELETE")
conn.execute("PRAGMA synchronous=OFF")

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"Tables: {[t[0] for t in tables]}")

count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
print(f"Tasks: {count}")

cols = conn.execute("PRAGMA table_info(tasks)").fetchall()
print(f"Cols: {[(c[1], c[2]) for c in cols]}")

try:
    ecount = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"Events: {ecount}")
except Exception as e:
    print(f"No events: {e}")

conn.close()
print(f"\nRecovered DB size: {os.path.getsize(new_db)} bytes")