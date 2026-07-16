# Code Review: Stalled-Thread Follow-Up

**Reviewed**: 2026-07-16
**Branch**: `feat/stalled-thread-followup` (local, uncommitted)
**Mode**: Local review (python-reviewer on the detector + self-review of the state SQL)
**Decision**: APPROVE (fixes applied)

## Summary
New opt-in detector + `hermes threads` CLI, reusing the governor/router/scan infrastructure. The specialist review found two real fail-soft gaps (CRITICAL) which are now fixed and covered by tests; the remaining findings were either false alarms (verified) or acceptable-as-documented trade-offs.

## Validation Results
| Check | Result |
|---|---|
| Lint (ruff) | Pass (all 8 files) |
| Unit tests | Pass (27, +5 review-driven) |
| Regression | Pass (89 across stalled/governor/omi/delivery; earlier 242 across proactive suite) |
| Live-data verification | Pass (real DBs; standalone path works post-refactor) |

## Findings

### CRITICAL (fixed)
- **Fail-soft gaps in `run_stalled_thread_scan` / `list_stalled_candidates`.** Per-source try/excepts covered Sources A/B, but three orchestration ops sat outside any guard: the `SessionDB(...)` open, `stall_nudged_recently` (dedup), and `record_stall_nudge` (post-delivery write). A DB error there would propagate to the CLI/cron caller, violating the module's fail-soft invariant. FIXED: wrapped both public functions in a top-level try -> `{"error": ...}` guard (mirrors the governor's `should_deliver` -> `_impl` fail-open idiom). Added `test_record_nudge_failure_is_fail_soft` + `test_scan_db_open_failure_is_fail_soft`.

### MEDIUM (fixed / accepted)
- **Log levels too low for error conditions** (classify failure, non-JSON output, digest delivery failure were `info`/`debug`). FIXED -> `warning`.
- **N+1 last-message query** in `list_live_threads_for_stall` (one extra query per live thread). Accepted — runs every 12h over dozens of threads, indexed lookup; documented trade-off.
- **`started_at` uniqueness tiebreaker** in the "newest session per session_key" filter. Accepted — clock-collision is rare and dedup absorbs a duplicate; documented.

### LOW (deferred — style, not shipped)
- Magic numbers (`12000`, `1500`) could be named constants; redundant `isinstance` in `_cfg`. Skipped to keep the diff focused; no behavior impact.

### Rejected (verified false alarms)
- **"`_to_epoch` can raise on malformed dates" (claimed HIGH)** — verified false: `_to_epoch` catches `ValueError`/`OSError` and returns `None` for `2026-99-99`, `garbage`, `''`, etc. Added `test_parse_due_malformed_date_returns_none` to lock it in. No code change needed.
- **"exception between should_deliver and send in `_deliver_digest`"** (self-downgraded by reviewer) — the whole function is in a try -> return-False, so a pre-send error correctly yields "not delivered" and no nudge recorded. Correct as-is.

### Verified-correct (no change)
- Dedup BEFORE classify (avoids wasting an LLM call on cooldown'd items).
- `record_stall_nudge` only on `delivered=True` (suppressed digest doesn't burn cooldown).
- `min_confidence` gate + `max_items` cap ordering.
- Last-message via `ORDER BY id DESC` (not timestamp — non-monotonic clock).
- SQL fully parameterized.

## Refactor applied during review
Hoisted `SessionDB` / `get_hermes_home` to module-level imports in `agent/stalled_threads.py` (were function-local). Behavior-neutral; makes the fail-soft DB-error paths patchable/testable and matches how the module uses them.

## Files Reviewed
| File | Change |
|---|---|
| `agent/stalled_threads.py` | Added (detector) — fail-soft + log fixes + import hoist applied |
| `tests/agent/test_stalled_threads.py` | Added (27 tests) |
| `hermes_state.py` | Modified (`stalled_nudges` table + 3 methods) — SQL reviewed, sound |
| `hermes_cli/config.py`, `cli-config.yaml.example` | Modified (config section) |
| `hermes_cli/subcommands/notify.py`, `notify_cmd.py`, `main.py` | Modified (CLI) — mirrors reviewed omi pattern |

## Decision
APPROVE. Both CRITICAL fail-soft gaps fixed and tested; validation green.
