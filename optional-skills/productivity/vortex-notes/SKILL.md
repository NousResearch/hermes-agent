---
name: vortex-notes
description: Read, search, and write the user's Vortex Notes vault — plain-markdown
  notes with zero-config local semantic search, daily journaling, and durable facts
  with supersession. Use for "note this down", "remember that…", "what do my notes
  say about…", and building context before a task. CLI and MCP integration.
version: 1.0.0
author: Vortex303
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Notes, Knowledge-Base, Memory, Markdown, MCP, Search, Local-AI]
    related_skills: [qmd, obsidian, native-mcp]
---

# Vortex Notes — the vault your agents can actually use

The user's knowledge base is a folder of plain `.md` files ("the vault") with a
first-party MCP server and built-in hybrid semantic search (BM25 + multilingual
embeddings, fully local, no API key).

## When to Use

- The user says "note this down", "add to my notes", "remember that…"
- You need prior context: "what do my notes say about X", "what did we decide"
- Daily journaling: observations, decisions, progress logs
- Any task that benefits from the user's accumulated knowledge

## Prerequisites

```sh
npm install -g vortex-notes
vortex-notes init            # creates ~/VortexNotes (or ask the user for their vault)
```

Vault resolution: `--vault <dir>` flag > `$VORTEX_NOTES_VAULT` > `~/VortexNotes`.

## Quick Reference

| Task | Command / action |
|---|---|
| Search (always start here) | `vortex-notes search "<query>" --vault <vault>` |
| Read a note | read the `.md` file directly |
| Daily entry | append `- **HH:MM** <entry>` to `daily/YYYY-MM-DD.md` |
| Durable fact | append dated bullet with `^f-<id>` to `memory/facts.md` |
| Rebuild index | `vortex-notes index --vault <vault>` |

Search is hybrid keyword + semantic and multilingual — natural-language queries
work, and a query in English finds notes written in Spanish.

## Writing Conventions

- Notes are markdown with YAML frontmatter (`title`, `tags`, `created`, `updated`).
  `[[Wikilinks]]` reference other notes by title or filename — follow them.
- **Daily journal**: `daily/YYYY-MM-DD.md`, one timestamped bullet per entry:
  `- **HH:MM** shipped the relay deploy`. Create with `title: YYYY-MM-DD` and a
  `daily` tag if missing.
- **Facts** ("remember that X"): `memory/facts.md` (or `memory/<topic>.md`):
  `- **YYYY-MM-DD** <fact> \`^f-<8 random lowercase chars>\``.
  If a fact REPLACES an older one, strike the old line through and point forward:
  `- ~~old line~~ → superseded YYYY-MM-DD by \`^f-new\`` — never delete history.
- **Edits**: surgical (append a section, fix a line); preserve frontmatter and
  bump `updated`. Never rewrite whole notes unless asked.

The vault's watcher reindexes automatically after any write.

## MCP Integration

Preferred when the harness supports MCP (stdio):

```json
{ "command": "vortex-notes", "args": ["mcp"], "env": { "VORTEX_NOTES_VAULT": "<vault>" } }
```

If the user granted this agent its own scoped access via `npx vortex-notes pair`,
that command wires Hermes automatically and a bare `vortex-notes mcp` (no flags)
serves the paired vault — nothing else to configure.

If the user granted this agent its own scoped access (an `vnat1_…` agent token),
a single self-bootstrapping command replaces all setup:

```json
{ "command": "vortex-notes", "args": ["agent", "mcp", "<vnat1_token>"] }
```

Tools: `search_notes`, `read_note`, `write_note`, `edit_note` (surgical),
`append_daily`, `remember` (facts with supersession), `build_context` (top notes
plus one hop of wikilinks in one call), `recent_activity`, `list_notes`.
Add `--read-only` for a search/read-only connection. The MCP tools encode all
the conventions above — prefer `remember` and `append_daily` over manual writes.

## Best Practices

1. Search before writing — the note or fact may already exist (then edit/supersede).
2. Use `build_context` (MCP) at task start instead of several search+read rounds.
3. Put durable knowledge in `memory/`, ephemera in `daily/` — don't mix.
4. Keep facts self-contained: they must make sense without conversation context.
5. Respect `--read-only` vaults: search and read, never suggest workarounds.

## Data Storage

Vault = plain markdown, portable, Obsidian-compatible. Search index lives in
`<vault>/.vortex/index.db` (disposable cache — the files are the truth).
Optional E2EE multi-device sync exists (`vortex-notes sync`); it never changes
the on-disk format.

## References

- https://github.com/vortex-303/vortex-notes
- `vortex-notes help`
