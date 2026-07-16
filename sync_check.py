import sqlite3, json, re, hashlib, os, sys
from pathlib import Path
from datetime import datetime, timezone

hermes_dir = Path(r"C:\Users\downl\.hermes")
state_db = hermes_dir / "state.db"
ebbinghaus_db = hermes_dir / "ebbinghaus_memory.db"
lm_twitterer_jsonl = hermes_dir / "lm-twitterer" / "activity.jsonl"

# Check state.db schema
conn = sqlite3.connect(str(state_db))
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("state.db tables:", tables)

if 'sessions' in tables:
    cur.execute("PRAGMA table_info(sessions)")
    print("sessions columns:", [r[1] for r in cur.fetchall()])