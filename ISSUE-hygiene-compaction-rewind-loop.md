# Hygiene compaction enters a no-progress fixed point for long-lived sessions, duplicating the transcript into `active=0, compacted=0` rows on every turn (~300k rows/day)

**Area:** sessions / compression / gateway hygiene
**Version:** `v0.22.5+530.g99fcd69` (2026.9.24), commit `99fcd6944360`

## Summary

For a session whose live transcript sits above `hygiene`'s `hard_msg_limit`
(400), session-hygiene compaction can never make progress. It fires on every
turn, covers at most the bounded 400-message payload, treats every *other*
active row as `unseen`, and then both rewinds it (`active=0, compacted=0`) and
clones it back as a fresh `active=1` row. The live count therefore never drops
below the value that triggers compaction, so the cycle repeats forever and each
pass leaves ~1,000 superseded rows behind.

The affected session accumulated **2.8M message rows — 96% of the entire
`state.db`** — degrading the whole WebUI (session list rebuilds of 228–378 s,
and `POST /api/session/archive` taking 51–177 s so the browser's 30 s fetch
timeout surfaced as `Archive failed: Request timed out.` on requests that the
server actually answered `200`).

## Root cause

The four steps, with locations at `99fcd6944360`:

1. **Compaction triggers on every turn.**
   `gateway/run_turn_hygiene.py:191`
   ```python
   _needs_compress = _approx_tokens >= _compress_token_threshold or _msg_count >= hs.hard_msg_limit
   ```
   The session held ~1,420 active rows against `hs.hard_msg_limit = 400`, so
   this is `True` unconditionally, every turn.

2. **But the payload handed to the compressor is cut to that same limit.**
   `gateway/run_turn_hygiene.py:817` `_bound_hygiene_payload()` applies
   `bound_model_input_without_hygiene(history, hs.hard_msg_limit)`, which is
   what emits the log line:
   ```
   Session hygiene did not land for <sid>: bounding the model payload to 400 of
   1366 messages (hard limit 400) — the stored transcript is unchanged
   ```

3. **So coverage is capped at ~400 rows.**
   `agent/conversation_compression.py:3659`
   ```python
   covered_ids, unresolved_held = coverage_for_commit(
       agent._session_db, agent.session_id, original_messages, verbatim_tail)
   ```
   `coverage_for_commit()` (`agent/conversation_compression_archive.py:101`)
   derives coverage from that bounded in-memory list, so `covered_ids` can never
   name the ~1,000 active rows the compressor did not see.

4. **Everything uncovered is rewritten every cycle.**
   `hermes_state_messages.py:_archive_named_rows` (line 851)
   ```python
   unseen = [row_id for row_id in active_ids if row_id not in covered_set]
   rewind_ids += unseen
   conn.execute("UPDATE messages SET active = 0, compacted = 0 WHERE session_id = ? AND id IN (...)")
   ...
   if unseen:
       self._clone_message_rows(conn, unseen)   # fresh active=1 rows
   ```
   Rewind flags are intentional (see the `archive_and_compact` docstring:
   *"superseded duplicates and get rewind flags … so search doesn't return each
   carried message once per compaction"*), but applying them to the bulk of the
   transcript on every pass, forever, is pure garbage accumulation.

Live count after each pass = `|summary| + |unseen| + new turns`, which is still
`>= hard_msg_limit`, so step 1 re-fires next turn. **This is a stable fixed
point, not a race** — the compressor is working correctly, it just can never
shrink the transcript below its own trigger.

## Evidence

`gateway.log` spanning 5.5 days, for this one session:

| Event | Count |
|---|---|
| `Session hygiene did not land` | 890 |
| `another compression is already in flight` | 466 |
| `exceeded the turn-hold budget` | 417 |
| `finished after the turn-hold was released` | 417 |
| `did not rotate or compact in place` | 7 |

Database measurements:

- Runaway session `20260710_110441_43c01f` (source `homeassistant`) held
  **2,805,135 of 2,936,683** message rows (96%). The next-largest session had
  893 rows; only one session in 8,332 exceeded 5k.
- Flag composition of that session: `active=0, compacted=0` → **2,665,853**;
  `active=0, compacted=1` → 140,671; `active=1` → **1,368**.
- The live transcript itself was **healthy**: 710 user + 710 assistant, duplicate
  factor **1.00×**, opening with a proper `[CONTEXT COMPACTION — REFERENCE ONLY]`
  marker — confirming compaction does work.
- Duplicates: **44.3×** duplication factor overall; **198×** for
  `[interrupt: no assistant output for this turn]` (393,961 rows across only
  1,987 distinct timestamps); identical microsecond timestamps re-written across
  a **322,155-row** id spread.
- Growth was measured directly: **+271,032 rows in 21.1 h ≈ 309,000 rows/day**,
  of which **99.6% landed as `active=0, compacted=0`** — exactly the rewind
  flags from step 4.

**Arithmetic check:** 309,000 rows/day ÷ ~1,020 `unseen` rows per cycle
(≈1,420 live − 400 covered) ≈ **303 cycles/day**, matching the ~400 hygiene
events/day observed in the log.

## Impact

- `state.db` grew to 3.1 GB on a host where it is normally a few hundred MB.
- `GET /api/sessions` rebuilds took **228–378 s** live.
- `POST /api/session/archive` took **51.7 s / 161.4 s / 176.9 s / 56.7 s** and
  returned `200` each time, while the browser's `api()` wrapper
  (`static/workspace.js`, default `timeoutMs` 30000, no timeout retry for POST)
  aborted first — so users saw `Archive failed: Request timed out.` on
  successful operations. The same helper backs rename/pin/move.
- Any session-listing query paid the cost: the sidebar candidate CTE + `LEFT
  JOIN messages … GROUP BY` measured **24.72 s** with that session in the
  candidate window vs **0.14 s** with it excluded (176×).

## Proposed fixes

**A (principled).** In `coverage_for_commit()`, when proved coverage cannot
reach a meaningful fraction of the active set, return `(None, None)` and take
the existing watermark path. That path archives all active rows and clones only
rows that genuinely arrived after the watermark, so the transcript actually
shrinks below `hard_msg_limit` and the loop terminates on its own.

**B (safest stopgap).** Skip/back off a compaction that would archive only a
small fraction of active rows, with a cooldown longer than the current
`failure_cooldown_seconds` (300 s). Leaves the transcript oversized but stops
the garbage writes.

Either way, a **no-progress guard** seems worth having regardless: a compaction
that archives ~0 rows while cloning ~1,000 is measurably not helping and should
not repeat indefinitely.

## Reproduction conditions

A session whose stored active message count exceeds `hygiene`'s
`hard_msg_limit` (400) — realistically a long-lived gateway session that
receives a steady event stream (e.g. a Home Assistant event channel) and so
never gets rotated. Compaction then fires every turn and cannot converge.

## Workaround (upstream-unaffected deployments)

Until this is fixed, deleting only `active=0, compacted=0` rows that have an
identical *visible* twin is provably safe — the schema's own visibility rule is
`active = 1 OR compacted = 1` (`hermes_state_common.py:801-843`, the partial
display indexes, and the display triggers all enforce it), so those rows are
unreachable by construction. That removed 2.66M rows in ~3.5 minutes with
`PRAGMA integrity_check = ok` and both FTS indexes verifying clean.
