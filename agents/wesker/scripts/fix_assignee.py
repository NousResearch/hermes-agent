#!/usr/bin/env python3
import sqlite3

board_db = '/home/kensei/.hermes/kanban/boards/apps/kanban.db'

conn = sqlite3.connect(board_db)
c = conn.cursor()

# Show the task
c.execute("SELECT id, title, status, assignee FROM tasks WHERE id='t_009c287b'")
row = c.fetchone()
print(f"Before: id={row[0]}, title={row[1]}, status={row[2]}, assignee={row[3]}")

# Show valid profiles from board.json
import json
with open('/home/kensei/.hermes/kanban/boards/apps/board.json') as f:
    board = json.load(f)
    print(f"Board profiles: {board.get('profiles', board.get('assignable_profiles', 'not found'))}")

# Update assignee to octacon
c.execute("UPDATE tasks SET assignee='octacon' WHERE id='t_009c287b'")
conn.commit()

# Verify
c.execute("SELECT id, title, status, assignee FROM tasks WHERE id='t_009c287b'")
row = c.fetchone()
print(f"After: id={row[0]}, title={row[1]}, status={row[2]}, assignee={row[3]}")

conn.close()
