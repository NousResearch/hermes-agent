---
name: od-fragment-consolidator
author: ox-alpha (design profile)
description: Use when the Hermes sidebar is full of OD session fragments and they should be merged into per-project ledgers.
version: 1.0.0
license: MIT
metadata:
  hermes:
    tags: [opendesign, acp, consolidation, fragments, cleanup, transcript]
    category: integration
---

# OD Fragment Consolidator (merge fragments into per-project ledgers)

## Overview

After the session-reuse patches stop new fragments from spawning, the sidebar still shows the historical debris: dozens of one-message sessions titled `# Instructions (read first)` or `Set up OpenDesign ask mode #7`. This skill merges that debris into one readable "ledger" session per OD project, renames each ledger to the OD project's name, and asks the user before touching the emptied fragments.

## When to Use

- The sidebar is full of OD session fragments and the user wants them organized by project
- The user asks "can these be merged into one session per project?"
- After applying the session-reuse patches (see the sibling skill `od-hermes-session-reuse`), to clean up history

## How It Works (verified mechanics)

1. **Project assignment** — every charter envelope embeds the OD project path (`projects/<uuid>/`). Extract it per session; majority vote wins for sessions touching multiple projects. Fallback when envelopes were already translated away: align each session's activity window to OD conversations (coverage ≥ 50% of the session's lifespan; ties go to the tighter, more specific window).
2. **Ledger choice** — the newest session in each project (max message id) becomes the ledger; all sibling fragments merge into it.
3. **Merge** — rewrite `messages.session_id` in one transaction. Hermes resumes conversations by auto-increment row id (`ORDER BY id`), so the merged timeline is automatically chronological. Alternation breaks (`user;user` from turns that produced no reply) are auto-repaired by Hermes' built-in `repair_alternation` on resume.
4. **Rename** — ledger title := the OD project's display name (`title_source='user'`), so the sidebar mirrors the OD project list.
5. **Mapping pre-fill** — write conversation→ledger pairs into OD's `agent_sessions` so the next message in each OD window resumes the ledger. The table has no unique constraint: use DELETE+INSERT in a transaction. If OD holds the lock, skip gracefully — resume failure just falls back to a fresh session.
6. **Interactive cleanup** — after merging, ask the user what to do with the emptied fragments: **archive** (recommended; `archived=1`, reversible), **keep**, or **delete** (requires typing `DELETE`). Never decide for them.

## Usage

```bash
python scripts/consolidate.py            # dry-run: assignment + merge plan, no writes
python scripts/consolidate.py --apply    # execute; still prompts before cleanup
python scripts/consolidate.py --apply --yes  # execute; auto-confirm archive cleanup
```

Env overrides: `HERMES_STATE_DB`, `OD_DATA_DIR` (defaults match a standard Windows install).

## Safety Model

- Timestamped full-DB backup before the first write
- Dry-run by default; the merge plan is printed for review
- Merge is re-attributing rows, never deleting them
- Cleanup is a separate, explicit, user-confirmed step
- Idempotent: running against an already-consolidated DB yields an empty plan

## Pitfalls

| Pitfall | Detail |
|---------|--------|
| `group_concat` truncation | Aggregating all user messages per session to search for project paths silently truncates on big sessions — scan messages row by row instead |
| Envelope-less DBs | If transcripts were already translated, project UUIDs are gone; use the activity-window fallback |
| Long-lived window bias | Raw max-overlap assignment favors projects that stay open longest; normalize by coverage ratio and tie-break by tighter window |
| `ON CONFLICT` on `agent_sessions` | The table has no PRIMARY KEY/UNIQUE — upserts fail; DELETE+INSERT instead |
| Locked OD DB | OD holds the lock while running; pre-fill is best-effort and safe to skip |
| Ledger = active session | Merging into the live session is fine; the envelope of every new turn re-injects project context, so old envelopes in history are redundant anyway |

## Verification

- Every ledger: `min(id)` → `max(id)` monotonic (timeline order preserved)
- Alternation health: count `user;user` breaks per ledger (informational; auto-repaired)
- Post-merge resume: the model recalls context from hours-old turns
- Sidebar: one session per OD project, named exactly like the OD project
