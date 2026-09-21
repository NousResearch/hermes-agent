#!/usr/bin/env python3
import sqlite3

board_db = '/home/kensei/.hermes/kanban/boards/apps/kanban.db'
conn = sqlite3.connect(board_db)
c = conn.cursor()

c.execute("SELECT id, title, status, assignee FROM tasks WHERE assignee IS NULL OR assignee = '' ORDER BY status")
rows = c.fetchall()
print(f"Total unassigned tasks on apps board: {len(rows)}")
for id, title, status, assignee in rows:
    print(f"  {id} | {status:12s} | {title}")
conn.close()
