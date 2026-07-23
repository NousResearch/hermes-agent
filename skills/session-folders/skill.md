---
name: session-folders
description: Organize sessions into user-defined named folders.
---

# Session Folders

Use when the user asks you to organize their sessions into named groups ("folders").

## Available Tools

- `session_folder_list` — View existing folders and their session counts
- `session_folder_create` — Create a new named folder
- `session_folder_add` — Add session(s) to a folder

## Workflow: Create a folder and populate it

1. `session_folder_create(name='Design Projects')` — create the folder
2. `session_list(query='design')` — find relevant sessions
3. `session_folder_add(folder_id='...', session_ids=['...', '...'])` — add them

## Workflow: Organize existing sessions

1. `session_folder_list()` — see what folders exist
2. For each folder, `session_list(search='...')` to find matching sessions
3. Add sessions to the appropriate folder
