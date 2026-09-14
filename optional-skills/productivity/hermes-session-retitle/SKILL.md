---
name: hermes-session-retitle
description: Bulk-rename Hermes session titles to a consistent format.
version: 1.0.0
author: caesarjue
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [hermes, sessions, title, state-db, batch-ops]
    related_skills: [hermes-state-db-maintenance, hermes-agent]
---

# Hermes Session Retitle

Normalize Hermes session titles in `state.db` to one consistent, scannable
format — either as a one-shot cleanup of the whole history or as an incremental
pass over newly created sessions. This skill manages **titles only**: it never
touches message content, and every rename goes through the `hermes sessions
rename` CLI rather than writing to the database directly.

The default format is a convention the original author uses — treat it as a
recommended starting point and adapt it to any format the user prefers. What
makes a format *work* is a stable machine-checkable prefix (used for the
normalized / not-normalized test) and a fixed number of `|`-separated parts.

## When to Use

- User says "rename session titles", "重命名会话标题", "clean up my session list",
  or complains the session sidebar is a wall of inconsistent auto-generated names.
- After a burst of sessions (batch jobs, parallel agents, cron reruns) left
  titles like `AI早报 · Sep 12 09:42` or `None`.
- Periodically, when the user wants newly created sessions to keep a format the
  rest of the history already follows.

## Prerequisites

- `hermes` CLI on PATH — the rename path is `hermes sessions rename <session_id> "<new title>"`.
- `sqlite3` available for read-only inspection (`terminal` is fine for this).
- The database lives at `~/.hermes/state.db` (profile-aware:
  `~/.hermes/profiles/<name>/state.db`). All reads open it read-only
  (`file:...?mode=ro`); no mode that can write is ever used.

## Quick Reference

Recommended default format (customize per user):

```
📅 MMDD | <emoji> <type> | <topic>
```

- `MMDD` comes from the session id's first 8 digits (`20260911_...` → `0911`).
- `<type>` is a small closed set; one emoji per type. Example taxonomy:

  | Type | Emoji |
  |---|---|
  | development | 💻 |
  | fix / repair | 🔧 |
  | research | 🔍 |
  | content / writing | ✍️ |
  | ops | ⚙️ |
  | finance | 💰 |
  | life | 🏠 |
  | other | 📦 |

- `<topic>` keeps the user's own anchor words; prefer their language, strip
  dates and boilerplate.
- Cron sessions: keep the task name and the `HH:MM` from the run — same task
  reruns on one day must stay distinguishable for the UNIQUE index.
- Working in the DB:

  | Object | Detail |
  |---|---|
  | `sessions` primary key | `id` (NOT `session_id`) |
  | session creation time | `started_at` (REAL, unix seconds) |
  | `messages` join key | `session_id` |
  | normalized test | `title LIKE '📅%'` (or your chosen prefix) |

- Helper script: `scripts/retitle.py list` prints every unnormalized session
  with a first-message excerpt; `scripts/retitle.py apply plan.json` prechecks,
  backs up, renames, and verifies (see `## Procedure`).

## Procedure

1. **Inventory** — run `scripts/retitle.py list` (or query `state.db`
   read-only). For each candidate also pull the first user message; skip the
   machine-generated banners (`[IMPORTANT`, `[CONTEXT COMPACTION`, …) to reach
   the real anchor text. Sessions with zero messages are never named.
2. **Name** — classify each session into a type; keep the user's own words for
   the topic. For mixed-topic sessions, sample a few mid-history messages
   before deciding. When the user's existing title already states the topic,
   keep its wording.
3. **Precheck & back up** — build the plan as JSON (`[{"id": ..., "title": ...}]`).
   `scripts/retitle.py apply` rejects the plan if (a) two proposed titles are
   identical, or (b) one collides with a title already in the database —
   `title` carries a UNIQUE index, so either case would hard-fail mid-batch.
   The old titles are captured before any rename.
4. **Apply** — `scripts/retitle.py apply plan.json` runs `hermes sessions
   rename` per entry (~0.3s each) and reads every row back afterwards.
5. **Verify & report** — the script prints renamed / failed / verified counts.
   Finish with a full-DB ratio (`SUM(title LIKE '📅%') / COUNT(*)`) and report
   the result plus the backup location. The backup JSON is the undo record.

## Pitfalls

- **Column names**: `sessions.id` (not `session_id`); `sessions.started_at`
  (not `created_at`). A wrong column name returns an empty set silently on
  some sqlite3 invocations — verify with a count before trusting an empty list.
- **UNIQUE title index**: duplicates abort a batch partway. Disambiguate with
  time suffixes for cron reruns; always run the precheck.
- **`title_source` is not a freshness signal**: sessions processed by a
  previous pass also read `user`. Detect stale titles with the format prefix
  instead.
- **Skip zero-message sessions** (e.g. abandoned "Bot Chat" shells) — they
  exist to hold a chat id, not to be read; naming them adds noise.
- **Don't skip categories on your own initiative**: users who ask to
  "normalize everything" mean cron sessions and hand-named sessions too. Only
  empty sessions are a rule-based skip.
- **Idempotence**: a title already matching the format is left alone. The
  prefix test makes reruns safe.

## Verification

- Every renamed row read back byte-identical (the script does this and reports
  `unverified` on any mismatch).
- Full-DB ratio: `SELECT COUNT(*), SUM(title LIKE '📅%') FROM sessions;` — the
  remaining unnormalized count should equal exactly the intentional skips
  (empty sessions).
- Backup file present with one `{id, old, new}` record per change.
