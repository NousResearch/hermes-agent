# Codex SpendGuard — proactive global Codex spend cap

**Status:** approved design, pre-implementation
**Date:** 2026-06-09
**Branch:** `codex/cron-usage-limit-circuit-breaker` (complements the existing reactive cron breaker)

## Problem

Every existing usage-limit mechanism in Hermes is **reactive** — it acts only
*after* the provider returns a capacity/quota error:
- `cron/scheduler.py` pauses recurring cron jobs on provider-capacity errors, but
  only covers cron jobs (not Telegram/interactive Codex calls) and only after a
  call already failed.
- `agent/nous_rate_guard.py` gates the Nous provider after a 429 (not Codex).
- `agent/credits_tracker.py` / `agent/account_usage.py` only *display* usage.

There is **no proactive, global, cross-process cap at the Codex call site**. A
single Telegram-driven or interactive Codex call still spends even near
depletion, and the three watcher processes (`hermesgeneralist`/`2`/`3`) each spend
independently with no shared ceiling. This is the gap that let Codex tokens be
exhausted.

## Goal

A **proactive** pre-call gate at the Codex chokepoint that consults a **global,
cross-process sliding-window ledger** and refuses a Codex call *before it spends*
when a hard ceiling would be exceeded — bounding the worst case regardless of
which path (cron, Telegram, CLI) drives the call, across all Hermes processes.

## Enforcement stance (operator-directed)

The cap is a **hard backstop for a should-never-happen runaway**; in normal
operation it must never trip and must never destabilize Hermes. Therefore:
- **Block** a Codex call only on a genuine ceiling breach (raise a clean,
  caught `CodexSpendCapError` → the agent surfaces a "Codex spend cap reached"
  message rather than crashing). Auto-resumes when the sliding window drops back
  under the ceiling.
- **Fail OPEN on the guard's own errors.** If the ledger file is locked,
  missing, or corrupt — anything other than a real ceiling breach — the guard
  logs loudly at `:warning` and **allows** the call. The cap must never be the
  cause of an outage. (This is the deliberate difference from Symphony's
  fail-*closed* SpendGuard: that one is in-process; this ledger is cross-process
  file state where transient contention is normal.)

## Design

### `agent/codex_spend_guard.py`

A small module mirroring the **file-based shared-state pattern already proven in
`agent/nous_rate_guard.py`** (atomic temp-file + `os.replace`, advisory file lock
for the read-modify-write), persisting to a **single shared** ledger at
`~/.hermes/codex_spend.json` (NOT a per-profile `state.db` — must be global
across the watcher processes).

Ledger contents (compact, pruned every access):
- a list of recent call timestamps (unix seconds), and
- a list of `(timestamp, total_tokens)` for token accounting.

Hard ceilings (module constants — un-disableable; effective limit =
`min(ceiling, configured)`, so config can only *lower* them):
- `MAX_CALLS_PER_HOUR = 60`
- `MAX_CALLS_PER_DAY = 400`
- `MAX_TOKENS_PER_DAY = 5_000_000`

Public API:
- `reserve(now=None) -> Reservation` — prune the window, check all three
  ceilings; if a ceiling is breached return `Reservation(allowed=False,
  reason=...)`; otherwise append a call timestamp (atomic write) and return
  `Reservation(allowed=True)`. Any internal error (lock/IO/JSON) → log warning,
  return `Reservation(allowed=True, failed_open=True)` (**fail-open**).
- `record_tokens(total_tokens, now=None)` — append `(now, total_tokens)`,
  best-effort; errors swallowed + logged.
- `snapshot()` — current window counts (for `/usage` display + tests).

Window math is pure and unit-tested independent of the file I/O (a pure
`evaluate(calls, tokens, now, limits)` helper returns the allow/deny + reason).

Limits read from config (`codex_spend_cap:` in `config.yaml` via
`hermes_cli/config.py:load_config()`), each clamped to the hard ceiling. Absent
config → the hard ceilings apply (enforcement is always on).

### Integration at the chokepoint

In `agent/codex_runtime.py`:
- **`run_codex_stream(...)`** — at the very top, before the retry loop /
  `active_client.responses.create(**stream_kwargs)`: call `reserve()`. If
  `not allowed`, raise `CodexSpendCapError(reason)` (a new exception type; callers
  already wrap Codex calls in error handling, and `error_classifier` can map it to
  a clean non-retryable "usage limit" message). After a successful response, call
  `record_tokens(prompt + completion)` using the usage already extracted in
  `agent/conversation_loop.py:~1990` (or directly from `final.usage`).
- **App-server path** (`agent._codex_session.run_turn(...)`, ~line 69) — same
  `reserve()` gate before `run_turn`, `record_tokens` after.

The guard is consulted at the **single narrowest Codex entry points**, so cron,
Telegram, and CLI paths are all covered.

## Testing (TDD, pytest)

Following the existing `tests/agent/` patterns (`Mock` + `SimpleNamespace`, temp
files via `tmp_path`):
- **Pure window eval:** ceilings on calls/hour, calls/day, tokens/day; under →
  allow; at/over → deny with the right reason; sliding window prunes old entries.
- **Ledger I/O:** `reserve` appends and persists; a second guard instance
  (simulating another process) sees the first's calls (shared file); prune on
  read.
- **Fail-open:** unreadable/corrupt/locked ledger → `reserve` returns
  `allowed=True, failed_open=True` and logs; never raises.
- **Config clamp:** a configured limit above the hard ceiling is clamped down; a
  lower configured limit is honored; absent config → hard ceilings.
- **Chokepoint gate:** `run_codex_stream` raises `CodexSpendCapError` when
  `reserve` denies (Codex `responses.create` NOT called); allows + records tokens
  when permitted (mock the client, assert `responses.create` called once and
  tokens recorded).

## Rollout

In-repo change on the existing breaker branch; no other Hermes behavior changes.
Takes effect when the watcher processes are restarted (operator action). The
ledger file is created on first call. Ceilings are tunable via `config.yaml`
under the hard caps. Out of scope: per-profile sub-caps, cost(\$) accounting,
seeding the ceiling from the Codex subscription window (future).
