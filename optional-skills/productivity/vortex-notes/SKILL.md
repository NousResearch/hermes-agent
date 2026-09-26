---
name: vortex-notes
description: Read, search, and update the user's encrypted notes vault.
version: 1.1.0
author: Vortex303 (@vortex-303)
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Notes, Knowledge-Base, Memory, Markdown, MCP]
    related_skills: [qmd, obsidian, native-mcp]
---

# Vortex Notes Skill

The user's knowledge base is a folder of plain markdown files ("the vault") that
Vortex Notes exposes through a first-party MCP server with built-in on-device
semantic search. Use this skill to read, search, and write that vault. It does
NOT manage arbitrary files or the user's editor, and it never touches sync keys —
it speaks only to the paired Vortex Notes vault.

## When to Use

- The user says "note this down", "add to my notes", "remember that…".
- You need prior context: "what do my notes say about X", "what did we decide".
- Daily journaling: observations, decisions, progress logs.
- Building context before a task from the user's accumulated knowledge.

## Prerequisites

This skill drives the **`vortex-notes` MCP server** (preferred), with a CLI fallback.

Install and connect once, via the `terminal` tool:

```sh
npm install -g vortex-notes
npx vortex-notes pair          # prints a 6-letter code; the user approves it in the app
```

Pairing gives this agent its own scoped, revocable key — it never sees the user's
recovery phrase, and every edit it makes is signed as the agent. Once the host
has the `vortex-notes` MCP server wired, no per-call CLI is needed. CLI-fallback
vault resolution: `$VORTEX_NOTES_VAULT` > the paired vault.

## How to Run

Prefer the MCP tools exposed by the `vortex-notes` server:

- `search_notes` — hybrid keyword + semantic, multilingual. **Always start here.**
- `read_note` — fetch a note by path.
- `write_note` / `edit_note` — create or surgically update a note.
- `append_daily` — add a timestamped line to today's journal.
- `remember` — record a durable fact; supersedes the prior version rather than
  overwriting it.
- `build_context` — assemble the relevant notes for the current task.

If the MCP server is unavailable, fall back to native Hermes tools against the
vault directory: `search_files` to locate notes, `read_file` to read a `.md`,
and `patch` to edit one.

## Quick Reference

| Task | MCP tool | Native fallback |
|---|---|---|
| Find notes (start here) | `search_notes` | `search_files` |
| Read a note | `read_note` | `read_file` |
| Add / update a note | `write_note` / `edit_note` | `patch` |
| Daily entry | `append_daily` | `patch` on `daily/YYYY-MM-DD.md` |
| Durable fact | `remember` | `patch` on `memory/facts.md` |

## Procedure

1. **Search before writing.** Call `search_notes` (or `search_files`) to see
   whether the topic already exists; prefer updating over duplicating.
2. **Read the candidate** with `read_note` / `read_file` before editing.
3. **Write the smallest change** with `edit_note` / `patch`, keeping notes plain
   markdown so they stay Obsidian-compatible.
4. **Journal transient observations** with `append_daily`; reserve `remember`
   for durable facts that should supersede an earlier belief.
5. **Write in the user's voice only when asked** — the agent's edits are already
   signed as the agent.

## Pitfalls

- Don't create a new note when `search_notes` shows a close match — update it.
- `remember` supersedes; don't hand-delete the old fact — the struck-through
  trail is intentional and auditable.
- The vault may be end-to-end encrypted; never read raw sync blobs or the relay —
  always go through the MCP server or the paired vault files.
- A password-locked note returns a placeholder from `read_note`; do not treat
  that placeholder as the note's content.
- Pairing is per-agent and revocable; if calls start failing with an auth error,
  the user likely revoked this agent — ask them to re-pair.

## Verification

- `search_notes "<recent topic>"` returns the note you just wrote.
- After `append_daily`, `read_note daily/<today>` shows the new timestamped line.
- After `remember`, the fact appears once, with any prior version struck through
  and linked — not duplicated.
