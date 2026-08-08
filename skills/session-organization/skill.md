---
name: session-organization
description: Full workflow for organizing Hermes sessions — archive stale ones, group related ones into folders, rename for clarity.
---

# Session Organization

Complete workflow for keeping your session list tidy. Combines session-management and session-folders tools.

## Full Cleanup Workflow

1. **List all sessions** — `session_list(limit=100)`
2. **Archive stale ones** — Identify old/unused sessions and archive them
3. **Rename vague ones** — `session_rename(session_id='...', title='Design Review - Week 4')`
4. **Create folders** — `session_folder_create(name='Client Projects')`
5. **Group sessions** — `session_folder_add(folder_id='...', session_ids=['...', '...'])`
6. **Verify** — `session_folder_list()` to confirm

## Quick Reference

| Action | Tool | Example |
|--------|------|---------|
| List sessions | `session_list` | `session_list(limit=20)` |
| Archive | `session_archive` | `session_archive(session_id='abc')` |
| Rename | `session_rename` | `session_rename(session_id='abc', title='New Name')` |
| Delete | `session_delete` | `session_delete(session_id='abc')` |
| Create folder | `session_folder_create` | `session_folder_create(name='Bug Reports')` |
| Add to folder | `session_folder_add` | `session_folder_add(folder_id='x', session_ids=['a','b'])` |
| List folders | `session_folder_list` | `session_folder_list()` |
