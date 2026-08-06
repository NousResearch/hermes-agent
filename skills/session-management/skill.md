---
name: session-management
description: Manage Hermes sessions — list, archive, rename, and delete conversations from the desktop app.
---

# Session Management

Use when the user asks you to manage their Hermes sessions (list, archive, rename, delete, or organize).

## Available Tools

These tools are automatically available when the Hermes dashboard is running (desktop app open).

- `session_list` — List sessions with filters (archived status, source, search)
- `session_archive` — Archive or unarchive a session
- `session_rename` — Rename a session
- `session_delete` — Permanently delete a session

## Workflow: Clean up old sessions

1. `session_list(archived='exclude', order='recent')` — get the list
2. Find sessions the user wants to archive/rename/delete
3. Use the appropriate tool on each session ID

## Workflow: Archive a batch of old sessions

1. `session_list(limit=50)` — fetch recent sessions
2. Present a summary with titles and dates
3. For each session to archive: `session_archive(session_id='...')`
