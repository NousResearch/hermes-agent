---
name: apple-notes
description: Read, search, create, and organize Apple Notes on macOS.
version: 1.1.0
author: Li shixiong (lishix520); Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Notes, Apple, macOS, note-taking]
    related_skills: [obsidian]
---

# Apple Notes Skill

Use the `terminal` tool to drive Apple Notes natively through the helper script `scripts/apple_notes.py`, which runs `osascript` against Notes.app. This replaces third-party CLIs such as `memo`: no `brew tap` or install step, and every operation is noninteractive and parameterized. macOS only; notes sync to iPhone and iPad through iCloud.

## When to Use

- Read, search, create, or append to Apple Notes
- Save information to Notes.app for cross-device access
- Organize notes into folders or move notes between folders

## When NOT to Use

- Obsidian vault work -> use the `obsidian` skill
- Bear Notes -> separate app, not supported
- Agent-internal notes that do not need to sync -> use the `memory` tool

## Prerequisites

- macOS with Notes.app installed.
- `osascript` ships with macOS; there is no install step.
- Grant Automation access to Notes.app the first time `osascript` drives it (System Settings -> Privacy & Security -> Automation). The prompt appears once per binary.

## How to Run

Invoke the helper script with the `terminal` tool. Every subcommand takes explicit arguments and never prompts:

```bash
SCRIPT="skills/apple/apple-notes/scripts/apple_notes.py"

# Inspect the vault
python3 "$SCRIPT" list-folders
python3 "$SCRIPT" list-notes --folder "Notes"
python3 "$SCRIPT" search --query "standup"

# Read
python3 "$SCRIPT" read --title "Standup Notes" --folder "Notes"

# Create and append (plain-text body is converted to Notes HTML)
python3 "$SCRIPT" create --title "Standup Notes" --body "First entry" --folder "Notes"
python3 "$SCRIPT" append --title "Standup Notes" --body "Second entry" --folder "Notes"

# Folders and moves
python3 "$SCRIPT" create-folder --name "Project Alpha"
python3 "$SCRIPT" move --title "Standup Notes" --src "Notes" --dest "Project Alpha"
```

Pass `--body-html` instead of `--body` to write raw Notes HTML (for example a structured project-update template) without conversion.

## Quick Reference

| Action | Command |
| --- | --- |
| List folders | `list-folders` |
| List notes in a folder | `list-notes --folder F` |
| Search note titles | `search --query Q` |
| Read a note | `read --title T [--folder F]` |
| Create a note | `create --title T --body B [--folder F]` |
| Append to a note | `append --title T --body B [--folder F]` |
| Create a folder | `create-folder --name N` |
| Move a note | `move --title T --dest D [--src S]` |

When `--folder` is omitted, `create` writes to the default folder and `read`/`append`/`move` search across all folders.

## Procedure

1. Run `list-folders` first to resolve the target folder. If the right folder does not exist, create it with `create-folder`.
2. Before creating a note, run `search` by title to avoid duplicates.
3. To update an existing note, prefer `append` over recreating it; only rewrite when the user explicitly asks.
4. If `search` returns multiple matches, narrow by folder or keyword before acting. Do not guess.
5. For moves, confirm source and destination with the user before executing bulk moves.

## Pitfalls

- Notes returns the body as HTML-like content (`<div>`, `<br>`); `read` returns it verbatim.
- `search` matches note titles only. To find text inside a note, `read` it and search locally.
- Folder names resolve across accounts; if two accounts share a folder name, the first match is used.
- Automation permission is per-binary. A different Python interpreter may re-trigger the permission prompt.
- `append` concatenates HTML; very large notes may render slowly in Notes.app.

## Verification

Confirm the skill end to end with the `terminal` tool against a scratch folder, then delete it:

1. `list-folders` returns the current folders.
2. `create-folder --name "Hermes Skill Test"` creates a scratch folder.
3. `create --title "Skill Check" --body "hello" --folder "Hermes Skill Test"` creates a note.
4. `append --title "Skill Check" --body "more" --folder "Hermes Skill Test"` appends.
5. `read --title "Skill Check" --folder "Hermes Skill Test"` returns both entries.
6. `move --title "Skill Check" --src "Hermes Skill Test" --dest "Notes"` moves it.
7. Delete the scratch folder from Notes.app when finished.
