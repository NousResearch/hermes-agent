#!/usr/bin/env python3
import sqlite3
import os

boards_dir = '/home/kensei/.hermes/kanban/boards'
for board in sorted(os.listdir(boards_dir)):
    db_path = os.path.join(boards_dir, board, 'kanban.db')
    if not os.path.isfile(db_path):
        continue
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM tasks WHERE (assignee IS NULL OR assignee = '') AND status IN ('ready', 'todo', 'triage')")
        count = c.fetchone()[0]
        if count > 0:
            c.execute("SELECT id, title, status FROM tasks WHERE (assignee IS NULL OR assignee = '') AND status IN ('ready', 'todo', 'triage')")
            for id, title, status in c.fetchall():
                print(f"  {board:15s} | {status:12s} | {id:15s} | {title}")
        else:
            print(f"  {board:15s} | no unassigned tasks")
    except sqlite3.OperationalError as e:
        print(f"  {board:15s} | ERROR: {e}")
    conn.close()
