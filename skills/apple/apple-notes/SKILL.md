---
name: apple-notes
description: "Read, search and write Apple Notes via memo + osascript."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Notes, Apple, macOS, note-taking]
    related_skills: [obsidian]
prerequisites:
  commands: [memo]
---

# Apple Notes

Read Apple Notes with `memo`, write them with `osascript`. Notes sync across all Apple devices via iCloud. Everything here runs non-interactively, so it works with no human at the keyboard.

## Prerequisites

- **macOS** with Notes.app
- Install: `brew tap antoniorodr/memo && brew install antoniorodr/memo/memo`
- The calling app needs macOS Automation permission for Notes (System Settings → Privacy & Security → Automation). `osascript` error `-1743` means it was denied.

## When to Use

- User asks to create, view, search, or update Apple Notes
- Saving information to Notes.app for cross-device access

## When NOT to Use

- Obsidian vault management → use the `obsidian` skill
- Bear Notes → separate app (not supported here)
- Quick agent-only notes → use the `memory` tool instead

## Reading (memo, non-interactive)

```bash
memo notes                        # List: lines look like "N. Folder - Title"
memo notes -fl                    # List folders
memo notes -f "Folder Name"       # Filter by folder (N is the same global number)
memo notes -v N                   # Print note N as Markdown
memo notes -nc                    # List bypassing memo's cache (use after writes)
```

To search, filter the list in `terminal` (`memo notes -nc | grep -i "query"`), or use `osascript` for body text.

`memo notes -v N` garbles double quotes (Notes stores `&quot` without the semicolon and memo's converter mangles it). For an exact read use `plaintext` (below).

## Writing and exact reads (osascript)

Pass data to `osascript` as `argv`, never interpolated into the script, so quotes, links, `&` and hostile text are stored literally and nothing is executed. `folder (item 1 of argv)` picks the first match if two accounts (iCloud, On My Mac) have a folder with that name; when ambiguous, name the account in the script: `folder "X" of account "iCloud"`. Identify notes by id (an `x-coredata://...` string), not list number: numbers shift after writes.

```bash
# Get id by folder + title
osascript -e 'on run argv
tell application "Notes" to return id of first note of folder (item 1 of argv) whose name is (item 2 of argv)
end run' "$FOLDER" "$TITLE"

# Create; prints the new id
osascript -e 'on run argv
tell application "Notes" to return id of (make new note at folder (item 1 of argv) with properties {body:(item 2 of argv)})
end run' "$FOLDER" "$HTML"

# Exact read
osascript -e 'on run argv
tell application "Notes" to return plaintext of (first note whose id is (item 1 of argv))
end run' "$ID"

# Replace body
osascript -e 'on run argv
tell application "Notes" to set body of (first note whose id is (item 1 of argv)) to (item 2 of argv)
end run' "$ID" "$HTML"

# Delete (goes to Recently Deleted)
osascript -e 'on run argv
tell application "Notes" to delete (first note whose id is (item 1 of argv))
end run' "$ID"
```

The body is HTML and the note title is its first line (an `<h1>` works). Write simple HTML directly, or convert Markdown with the `mistune` already in memo's venv:

```bash
HTML=$("$(brew --prefix memo)/libexec/bin/python" -c 'import sys,mistune;print(mistune.markdown(open(sys.argv[1]).read()))' note.md)
```

Setting `body` replaces the whole note, and attachments/images are lost. To edit, read the note first, then write the full new body.

## Do not use

- `memo notes -s` launches fzf (interactive).
- `memo notes -a`, `-e`, `-d`, `-m`, `-r`, `-ex`: they open `$EDITOR` or prompt on stdin for the note number. `-a`/`-e` also paste the Markdown→HTML into an AppleScript string with no escaping, so any `"` (e.g. a link's href) fails with `syntax error ... (-2741)`, and crafted text can inject AppleScript, including `do shell script`.

## Rules

1. Confirm with the user before deleting a note or overwriting a note's body.
2. Don't open or read notes in folders whose names suggest secrets (API Keys, Backup Codes, Passwords, ...) unless the user asks for that note specifically.
3. After any write, re-list with `memo notes -nc` or re-read by id to verify.
4. Prefer Apple Notes when the user wants cross-device sync; use the `memory` tool for agent-internal notes and the `obsidian` skill for Markdown-native knowledge management.
