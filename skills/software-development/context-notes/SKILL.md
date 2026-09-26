---
name: context-notes
description: Persist reasoning across context as searchable notes.
version: 0.1.0
author: al157 (al157), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Context, Memory, Notes, Long-Running, Compaction, Session]
---

# Context Notes — Cross-Window Working Memory

## Why
When a long session's context window fills, the built-in compressor summarizes history into a single dense block. Summaries lose the *reasoning* behind events: why a fix failed, how a component behaves, which approach was abandoned. Context notes fix this by splitting memory into two streams.

## When to Use
- Long or multi-session tasks that span multiple context windows (debugging, large refactors).
- When you need to preserve the *reasoning* behind events (why a fix failed, how a component behaves) rather than just results.
- Before compaction is about to fire, to sink detail into searchable notes first.

## Core rule
**Keep the reasoning; recover the evidence.**
- Notes store conclusions (why, how, what next) — never raw evidence.
- Evidence (failing test output, tool output, exact error text) stays in searchable history and is pulled back on demand.

## PRESERVE — write notes before compaction fires
Write a note at each of these events; do not wait for the compressor to trigger:
- A fix failed → record the error, what was tried, the result, why it was abandoned.
- The user corrected you or made a decision → record it immediately.
- A milestone completed → record progress, next step, open questions.
- Context is nearly full / compaction imminent → write a handoff note with the reasoning, not the raw evidence.

Notes are written with `ctx_index(source: "<task>-notes")` so the full text is persisted and searchable — never summarized or truncated.

## RETRIEVE — three-way lookup at the start of each long-task turn
1. `ctx_search` — hook-captured events + manually indexed notes (BM25/FTS5).
2. `session_search` — the full session history, including raw tool output. This is the only full-history entry point; do not skip it.
3. `hindsight_recall` — semantic / entity-graph recall (main model only; delegated subagents may not have this tool — fall back to the other two).

## Relationship to compaction tools
- Compaction (`context.engine: compressor`, `ctx_execute`, threshold routing) saves tokens but loses detail.
- Context notes are the anti-compaction layer: they sink detail into searchable notes *before* compaction fires, so the loss is no longer fatal.
- They complement each other: compaction saves tokens, notes save semantics.
