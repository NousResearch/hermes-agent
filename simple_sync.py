#!/usr/bin/env python3
# Simple Hermes memory sync test
import sqlite3
from pathlib import Path

hermes_dir = Path(r"C:\Users\downl\.hermes")
state_db = hermes_dir / "state.db"
ebbinghaus_db = hermes_dir / "ebbinghaus_memory.db"

print(f"Hermes dir: {hermes_dir}")
print(f"State DB exists: {state_db.exists()}")
print(f"Ebbinghaus DB exists: {ebbinghaus_db.exists()}")

if state_db.exists():
    conn = sqlite3.connect(str(state_db))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    session_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM messages")
    message_count = cursor.fetchone()[0]
    print(f"Sessions: {session_count}, Messages: {
