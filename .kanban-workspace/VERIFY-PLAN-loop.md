# Verification Report: `/loop` Command Implementation Plan

> Task: verify-plan t_54a3383a
> Plan source: `/home/d/Desktop/agenda/hermes-agent/.kanban-workspace/PLAN-loop-implementation.md`
> Live source: `/home/d/Desktop/agenda/hermes-agent` (fork, working tree)
> Date: 2026-05-09

---

## Overall Decision: ISSUES

The plan is **structurally sound** but contains **multiple stale line numbers** and **one non-existent function reference** that would mislead an implementer. The code patterns, logic, and approach are correct. With line-number corrections and the `_is_gateway_running` reference fixed, the plan is implementable.

---

## Issue 1: Stale Line Numbers in cli.py (CRITICAL)

The plan references cli.py line numbers that have shifted by ~84 lines due to code drift since the plan was written.

| Claim in Plan | Actual Location | Shift |
|---|---|---|
| `elif canonical == "cron":` at line 6857-6858 | Line 6941-6942 | +84 |
| `elif canonical == "curator":` at line 6859-6860 | Line 6943-6944 | +84 |
| `elif canonical == "kanban":` at line 6861-6862 | Line 6945-6946 | +84 |
| `_handle_cron_command()` at line 6325-6580 | Line 6409-6653 | +84 |
| `_handle_curator_command()` after ~6580 | Line 6654 | +84 |
| `process_command()` entry at 6675-6694 | Not verified at this exact range | — |

**Impact:** An implementer inserting code at plan-stated line numbers would place it in the wrong location (inside `show_banner()` / `skin_engine` code, not the dispatch chain).

**Fix:** Update all cli.py line references to current locations. The insertion point should be:
- Dispatch branch: after line 6942 (`elif canonical == "cron":`) and before line 6943 (`elif canonical == "curator":`)
- Handler method: after line 6653 (end of `_handle_cron_command`) and before line 6654 (`_handle_curator_command`)

---

## Issue 2: Stale Line Numbers in gateway/run.py (CRITICAL)

Gateway line numbers have shifted by ~132 lines.

| Claim in Plan | Actual Location | Shift |
|---|---|---|
| `if canonical == "background":` at line 5697-5698 | Line 5830-5831 | +132 |
| `if canonical == "steer":` at line 5700 | Line 5833 | +132 |
| `if canonical == "kanban":` at line 5646-5647 | Line 5779-5780 | +132 |
| `_handle_background_command()` ends ~9390 | Line 9415 (starts here) | +25 |
| `_handle_kanban_command()` at 7444-7519 | Line 7577-7636 | +133 |

**Impact:** Same as Issue 1 — wrong insertion points.

**Fix:** Update all gateway/run.py line references. Insertion should be:
- Dispatch branch: after line 5831 (`return await self._handle_background_command(event)`) and before line 5833 (`if canonical == "steer":`)
- Handler method: after `_handle_background_command()` (starts at 9415) or near other tool handlers

---

## Issue 3: Non-Existent Function `_is_gateway_running()` (HIGH)

The plan's CLI handler code (lines 192-197) and gotcha #7 both reference:

```python
from hermes_cli.cron import _is_gateway_running
```

**This function does not exist anywhere in the codebase.**

Search performed across `/home/d/Desktop/agenda/hermes-agent`:
- `hermes_cli/cron.py` — no `_is_gateway_running` function
- `cli.py` — no such function
- `gateway/status.py` — has PID-file based detection (`_pid_exists`, `_scan_gateway_pids`) but no function by this name

**Impact:** The CLI handler code would raise `ImportError` at runtime.

**Fix:** Either:
- (A) Remove the gateway-running check from the CLI handler entirely (simplest), OR
- (B) Implement `_is_gateway_running()` in `hermes_cli/cron.py` using `gateway/status.py` logic, OR
- (C) Import from `gateway.status` directly, e.g.:
  ```python
  from gateway.status import _pid_exists, _get_gateway_pid_path
  def _is_gateway_running():
      pid_path = _get_gateway_pid_path()
      if not pid_path.exists():
          return False
      pid = int(pid_path.read_text().strip())
      return _pid_exists(pid)
  ```

---

## Issue 4: commands.py Insertion Position (MINOR — Clarification)

The plan says insert the `CommandDef` after `/cron` (line 160-162) and before `/kanban` (line 166-170). However, the actual order is:

- Line 160-162: `/cron`
- Line 163: blank
- Line 164-165: `/curator`
- Line 166-170: `/kanban`

**Impact:** Inserting between `/cron` and `/kanban` as stated would actually place it between `/cron` and `/curator`. This is still correct (scheduling commands grouped), but the plan should explicitly say "after `/cron` and before `/curator`" for precision.

**Fix:** Update plan text to say "after `/cron` (line 162) and before `/curator` (line 164)".

---

## Verified Correct References

The following plan claims were confirmed accurate (or very close) in live source:

| Claim | File:Line | Status |
|---|---|---|
| `CommandDef` dataclass | `hermes_cli/commands.py:45-57` | ✅ Confirmed |
| `COMMAND_REGISTRY` list | `hermes_cli/commands.py:64` | ✅ Confirmed |
| `GATEWAY_KNOWN_COMMANDS` frozenset | `hermes_cli/commands.py:287-292` | ✅ Confirmed |
| `is_gateway_known_command()` | `hermes_cli/commands.py:295-313` | ✅ Confirmed |
| `should_bypass_active_session()` | `hermes_cli/commands.py:339-359` | ✅ Confirmed |
| `_origin_from_env()` | `tools/cronjob_tools.py:71-88` | ✅ Confirmed |
| `cronjob()` tool signature | `tools/cronjob_tools.py:257-278` | ✅ Confirmed (line 257, close to plan's 257-275) |
| `create_job()` | `cron/jobs.py:423-577` | ✅ Confirmed (starts at 423, matches plan) |
| `parse_duration()` | `cron/jobs.py:104-122` | ✅ Confirmed (starts at 104, matches plan) |
| `parse_schedule()` | `cron/jobs.py:125-211` | ✅ Confirmed (starts at 125, matches plan) |
| `get_job()` | `cron/jobs.py:579` | ✅ Confirmed (plan says 423-577 for create_job, get_job is nearby at 579) |
| `_format_job()` includes `name` | `tools/cronjob_tools.py:227` | ✅ Confirmed |
| `_format_job()` includes `prompt_preview` | `tools/cronjob_tools.py:230` | ✅ Confirmed |
| `_format_job()` includes `state` | `tools/cronjob_tools.py:242` | ✅ Confirmed |
| `schedule_display` in job record | `cron/jobs.py:550` | ✅ Confirmed |
| Test template `test_goals.py` | `tests/hermes_cli/test_goals.py` | ✅ Confirmed |
| Test template `test_scheduler.py` | `tests/cron/test_scheduler.py` | ✅ Confirmed |
| Test template `test_goal_verdict_send.py` | `tests/gateway/test_goal_verdict_send.py` | ✅ Confirmed |

---

## Code Pattern Verification

### CLI Handler Pattern ✅
The proposed `_handle_loop_command()` matches the existing `_handle_cron_command()` pattern:
- Imports `cronjob_tool` from `tools.cronjob_tools`
- Defines `_cron_api()` wrapper for JSON parsing
- Uses `shlex.split()` for tokenization
- Prints kawaii status lines
- Manual subcommand branching

### Gateway Handler Pattern ✅
The proposed `async def _handle_loop_command()` matches existing gateway handlers:
- Async method returning `str`
- Strips command prefix from `event.text`
- Uses `shlex.split()` for args
- Returns markdown-formatted strings

### Import Paths ✅
- `tools.cronjob_tools` — valid import path
- `cron.jobs` — valid import path (both `from cron import get_job` and `from cron.jobs import get_job` work)

### `deliver="origin"` ✅
The `cronjob` tool auto-populates origin via `_origin_from_env()` at line 334 in `tools/cronjob_tools.py`. Passing `deliver="origin"` is correct.

### Subcommand Logic ✅
- `list` with `name.startswith("loop:")` filter — works because `name` is stored in job record
- `pause`, `resume`, `remove` — all valid `cronjob` tool actions
- No race condition between create and list — `create_job()` writes atomically to `jobs.json`

---

## Test Strategy Verification ✅

- `tests/hermes_cli/test_goals.py` exists — good template for CLI handler tests
- `tests/cron/test_scheduler.py` exists — good template for cron integration tests
- `tests/gateway/test_goal_verdict_send.py` exists — good template for gateway dispatch tests
- Proposed new test files (`test_loop_command.py` in both `hermes_cli` and `gateway`) follow existing conventions

---

## Edge Cases Verification ✅

All edge cases listed in the plan (3.5) are correctly handled by the proposed code or by existing cron infrastructure:
- Empty prompt — handled in both CLI and gateway
- Short interval warning — CLI only, acceptable
- No gateway running — **blocked by Issue 3** (function doesn't exist)
- Duplicate loops — `create_job()` always generates new UUID
- Non-existent loop removal — `cronjob` tool returns `success: false`
- Invalid schedule — `parse_schedule()` raises ValueError, caught by tool
- Threat patterns — `_scan_cron_prompt()` blocks
- Long prompt — truncated in name, full stored in record
- Special characters — `shlex.split()` handles quotes

---

## Gotchas/Pitfalls Verification

| # | Claim | Status |
|---|---|---|
| 1 | `CommandDef` alone doesn't add execution — must add dispatch in both CLI and gateway | ✅ Correct |
| 2 | Cron scheduler runs in gateway, not CLI | ✅ Correct |
| 3 | `create_job()` uses `_origin_from_env()` automatically | ✅ Correct |
| 4 | Gateway handler is async, CLI is sync | ✅ Correct |
| 5 | No tests requiring running gateway daemon | ✅ Correct |
| 6 | `/cron` is `cli_only=True`, `/loop` is not | ✅ Correct |
| 7 | `_is_gateway_running()` imported from `hermes_cli.cron` | ❌ **FALSE — function does not exist** |
| 8 | Job name prefix `loop:` is filter key | ✅ Correct |
| 9 | `GATEWAY_KNOWN_COMMANDS` auto-derived | ✅ Correct |
| 10 | `cronjob` tool returns JSON strings | ✅ Correct |
| 11 | No-agent mode out of scope | ✅ Correct |
| 12 | Skill attachment out of scope | ✅ Correct |

---

## Summary of Required Plan Corrections

1. **cli.py line numbers:** Add ~84 to all cli.py line references
2. **gateway/run.py line numbers:** Add ~132 to all gateway/run.py line references
3. **`_is_gateway_running()`:** Either remove the check, implement the function, or import from `gateway.status`
4. **commands.py insertion text:** Clarify "before `/curator`" not "before `/kanban`"

After these corrections, the plan is **APPROVED** for implementation.
