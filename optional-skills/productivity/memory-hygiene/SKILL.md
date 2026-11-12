---
name: memory-hygiene
description: "Deterministic dry-run scanner + quarantine protocol for the Hermes memory stores (MEMORY.md / USER.md)."
version: 1.0.0
author: lioxatugol
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Memory, Hygiene, Consolidation, Quarantine]
    related_skills: [hermes-session-log]
---

# Memory Hygiene

Consolidation and junk-cleanup of Hermes' *injected* memory layer — `MEMORY.md`
(agent notes, target `memory`) and `USER.md` (user profile, target `user`).
Both are injected into every session and have a hard character budget, so when
they fill up, low-signal entries (status logs, stacked duplicates, stale facts)
evict the standing rules that actually matter. This skill provides:

- a **deterministic, read-only scanner** (`scripts/memory_hygiene.py`) that
  reports near-duplicates, status-log-like entries, and budget pressure;
- a **quarantine protocol**: every content-losing edit to memory is logged to
  `<HERMES_HOME>/mem-quarantine.log` *before* it happens, so deletions stay
  auditable and reversible.

It complements `hermes-session-log` (session archive): a removed memory entry
can always be recovered from the quarantine log + the session archive. The
session database (`state.db`) is never touched — that is the archive, not the
cleanup target.

## Stores

Per profile (default layout, overridable via CLI flags):

- `<HERMES_HOME>/memories/MEMORY.md` — agent notes; entries separated by a
  line containing only `§`; budget enforced over the whole file (~2200 chars);
- `<HERMES_HOME>/memories/USER.md` — user profile; same format (~1375 chars);
- `<HERMES_HOME>/mem-quarantine.log` — created on first use; one line per
  content-losing operation (see below).

`HERMES_HOME` resolution: `$HERMES_HOME` env var, else `%LOCALAPPDATA%/hermes`
on Windows, else `~/.hermes`.

## When to Use

- A store is at 80%+ of its budget, or a new entry no longer fits.
- The user asks to "clean up memory", "merge duplicates", "rebuild MEMORY/USER".
- Before ANY `memory` remove/patch that loses content (quarantine is mandatory).
- Periodic check: run the scanner (it is a dry run, safe to run anytime).

## Procedure

1. **Scan**: `python <skill>/scripts/memory_hygiene.py` — read-only report:
   entry counts, per-store budget pressure (>80% prints a soft flag, not an
   error), exact/near-duplicate pairs, status-log-like entries. The script
   never edits or deletes anything.
2. **Plan**: read both stores in full; classify each entry —
   stale (contradicted by a newer fact), duplicate (merge into the canonical
   entry), misplaced (user facts → USER, machine/environment → MEMORY;
   one-copy rule: a fact must not live in both stores).
3. **Quarantine BEFORE deletion**: append one line per content-losing entry to
   `mem-quarantine.log`, before the operation, not after:
   `<date> | <memory|user> | <full text lost> | <reason>`.
   Full text, not a placeholder — the log + session archive are the only way
   back.
4. **Apply** via the `memory` tool: one call per target, batch the operations
   (patch + remove in one batch, atomic); `old_text` = the full entry or a
   unique fragment. The budget is checked on the final result of the batch,
   not per operation — a remove frees room for an add in the same call.
5. **Verify**: re-run the scanner; report to the user what changed/was removed,
   with pointers to the quarantine lines.

## Rules

- **One lesson = one entry**: a lesson learned twice → patch the old entry,
  don't stack a new one on top.
- **Status logs** ("cycle 14/14", "run complete") and one-off task progress do
  not belong in memory (their home is the session DB); keep a pointer, drop
  the log line.
- **Owner statements** (user preferences) are pruned only on explicit
  staleness evidence; the reason goes into the quarantine line.
- **Per-run cap**: at most 5 changes, unambiguous candidates only (exact
  duplicates, clearly stale). Anything doubtful waits for a human decision.
- **Deletion is never silent**: the quarantine line is written *before* the
  operation.
- **No LLM "by-eye" cleanup in a loop**: the pipeline is deterministic scanner
  → agent decision with quarantine. An LLM inside the deletion loop is
  non-reproducible and can delete what is needed.
- A healthy free margin is ~20–40%; do not cut signal just to pass the 80%
  threshold — the scanner's flag is a soft signal.

## Script

`scripts/memory_hygiene.py` (stdlib only, read-only, exit code always 0):

    python memory_hygiene.py [--mem PATH] [--user PATH]
                             [--budget-mem 2200] [--budget-user 1375]

Duplicate detection: exact match after normalization, or
`difflib.SequenceMatcher` ratio ≥ 0.85 for entries longer than 40 chars.
Status-log detection is a regex heuristic (N/N counters, "cycle done", common
completion phrases in EN/RU); tune `STATUS_RE` at the top of the script for
your vocabulary.

## Gotchas (tool `memory`)

- `patched_entries` in the tool result is numbered sequentially over the
  touched entries/operations, not by entry index in the file — verify batch
  results by re-reading the store, not by the map keys.
- `remove` needs `old_text` identifying the entry; the safest is the full
  entry text (without the `§` line).
- `usage` in the result ("83% — 1,836/2,200") is measured over the whole file.
