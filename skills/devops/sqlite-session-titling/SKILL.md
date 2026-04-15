---
name: sqlite-session-titling
category: devops
description: Generate and manage titles for sessions stored in SQLite databases with duplicate handling and source tracking
---

# SQLite Session Titling

Generate and manage short, descriptive titles for sessions stored in SQLite databases. Handles duplicate titles with automatic counter suffixes and tracks title sources (auto-generated vs manual).

## When to Use

- You need to add titles to existing session records in a SQLite database
- You want to automatically generate titles from first messages or other content
- You need to preserve manually-set titles while auto-generating for untitled ones
- You're working with the Hermes state.db or similar session storage systems

## Key Patterns

### Database Schema Inspection

Always inspect the schema before making changes:

```python
import sqlite3

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get table schema
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'")
schema = cursor.fetchone()

# Check column existence
cursor.execute("PRAGMA table_info(sessions)")
columns = cursor.fetchall()
```

### Adding Columns Safely

Use ALTER TABLE with error handling for duplicate column errors:

```python
try:
    cursor.execute("ALTER TABLE sessions ADD COLUMN title_source TEXT DEFAULT NULL")
    conn.commit()
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("Column already exists")
    else:
        raise
```

### Duplicate Title Handling

When generating titles, check for duplicates and append counters:

```python
def update_with_duplicate_handling(cursor, session_id, title):
    # Check if title already exists
    cursor.execute(
        "SELECT COUNT(*) FROM sessions WHERE title = ? AND id != ?",
        (title, session_id)
    )
    count = cursor.fetchone()[0]
    
    # If title exists, append a counter
    if count > 0:
        base_title = title
        counter = 2
        while True:
            new_title = f"{base_title} ({counter})"
            cursor.execute(
                "SELECT COUNT(*) FROM sessions WHERE title = ? AND id != ?",
                (new_title, session_id)
            )
            if cursor.fetchone()[0] == 0:
                title = new_title
                break
            counter += 1
    
    # Update with the unique title
    cursor.execute(
        "UPDATE sessions SET title = ?, title_source = 'auto' WHERE id = ?",
        (title, session_id)
    )
    conn.commit()
```

### Title Source Tracking

Always track whether titles are auto-generated or manual:

```python
# Only update auto-generated or null titles
cursor.execute("""
    UPDATE sessions
    SET title = ?, title_source = 'auto'
    WHERE id = ?
    AND (title IS NULL OR title = 'CLI Session' OR title_source = 'auto')
""", (title, session_id))
```

## Common Pitfalls

### UNIQUE Constraint on Title Column

If the sessions table has a UNIQUE constraint on the title column, you'll get constraint errors when trying to set duplicate titles. Always check for duplicates first and use counter suffixes.

### Overwriting Manual Titles

Never overwrite titles that were manually set. Always check the title_source column or verify the title isn't a default value before updating.

### Missing title_source Column

The title_source column may not exist in older databases. Always check for its existence and add it if needed before using it.

## Implementation Example

See the title-sessions skill at `~/.hermes/skills/title_sessions.py` for a complete working example that:
- Reads first user messages from sessions
- Generates short titles using heuristics
- Handles duplicate titles with counters
- Tracks title sources
- Supports normal and backfill modes

## Testing

Always test with a small subset first:

```python
# Test with most recent session only
cursor.execute("""
    SELECT id FROM sessions
    WHERE title IS NULL OR title = 'CLI Session'
    ORDER BY started_at DESC
    LIMIT 1
""")

# Then expand to backfill mode
four_days_ago = (datetime.now() - timedelta(days=4)).timestamp()
cursor.execute("""
    SELECT id FROM sessions
    WHERE title IS NULL OR title = 'CLI Session'
    AND started_at >= ?
    ORDER BY started_at DESC
""", (four_days_ago,))
```

## Cron Job Integration

For automatic titling of new sessions, set up a cron job:

```bash
*/5 * * * * python ~/.hermes/skills/title_sessions.py
```

The script should be lightweight by skipping already-titled sessions.