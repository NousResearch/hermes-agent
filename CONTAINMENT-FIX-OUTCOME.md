# Plan-mode containment fix — guard/write base-divergence

PR #61961 (feat/plan-mode-enforcement). Finding: `is_plan_path()` proved
containment against a path resolved differently from the one `file_tools`
actually writes.

## 1. Reachability verdict: CONFIRMED REACHABLE

Evidence chain (traced against HEAD 9f1391913):

- **Guard call site** — `agent/tool_executor.py` `_plan_mode_block_reason()`
  → `hermes_cli/plan_mode.py:tool_block_reason()` →
  `is_plan_path(target)`. The OLD `is_plan_path` (plan_mode.py ~L311) derived
  the plans-dir root from the string prefix up to `.hermes/plans` and called
  `Path(root_str).resolve()` / `Path(normalized).resolve()`. `Path.resolve()`
  anchors RELATIVE paths against the **process** `os.getcwd()`.
- **Write path** — `write_file_tool` (`tools/file_tools.py:1667`) and
  `patch_tool` (L1750) resolve via `_resolve_path_for_task(path, task_id)` →
  `_resolve_base_dir(task_id)`, which anchors relative paths against the
  **task** base: live terminal cwd → registered workspace override →
  `TERMINAL_CWD` → (only last resort) process cwd (L372-427). Its own docstring
  (L389-397) documents this divergence as the "worktree-cwd divergence bug".
- **task_id availability at the guard** — both guard call sites
  (`tool_executor.py` L525 and L1211) already have `effective_task_id` in
  scope (the enclosing `execute_tool_calls_concurrent` / `_sequential`
  parameter), and pass it to every other resolver — but were NOT passing it to
  the plan guard.

**Concrete divergence proof** (reproduced, see commit): with process cwd = P
(benign real `P/.hermes/plans/`) and task base = T (`TERMINAL_CWD`, with
`T/.hermes/plans/escape` a symlink out of the plans dir), the OLD guard returns
`True` (allow) for the relative path `.hermes/plans/escape/app.py` — resolving
it to `P/.hermes/plans/escape/app.py` (contained) — while the real write
resolves to `T/.hermes/plans/escape/app.py` → follows the symlink →
`T/outside/app.py`, **outside the plans dir**. Guard says contained; write
escapes. Reachable.

## 2. The fix

`hermes_cli/plan_mode.py`
- **`is_plan_path(path, task_id="default")`** (~L311): dropped the
  process-cwd `Path().resolve()` + string-derived root. Now reuses
  `tools.file_tools._resolve_path_for_task` — the identical resolver the write
  uses — for BOTH the candidate and the plans-dir root
  (`_resolve_path_for_task(".hermes/plans", task_id)`), then requires
  `candidate.is_relative_to(root)`. Base + symlink resolution can no longer
  diverge from the write's. Fail-closed preserved: resolver error
  (OSError/ValueError/RuntimeError/ImportError) or non-comparable paths → block.
- **`tool_block_reason(..., task_id="default")`**: new trailing param, passed
  into `is_plan_path(target, task_id)`.

`agent/tool_executor.py`
- **`_plan_mode_block_reason(agent, function_name, function_args, task_id="default")`**:
  new trailing param, forwarded to `tool_block_reason(...)`.
- Both call sites (L525 concurrent, L1211 sequential) now pass
  `effective_task_id`.

### How the task base is threaded
`execute_tool_calls_{concurrent,sequential}(effective_task_id)` →
`_plan_mode_block_reason(..., effective_task_id)` →
`tool_block_reason(..., task_id=effective_task_id)` →
`is_plan_path(target, task_id)` →
`_resolve_path_for_task(target, task_id)` (same call the write makes).

### Blast radius
Exactly the 3 files the packet allowed: `hermes_cli/plan_mode.py`,
`agent/tool_executor.py`, `tests/hermes_cli/test_plan_mode.py`. All added
params are trailing with defaults — no ripple to other callers.

## 3. Tests (`scripts/run_tests.sh`)

New / changed in `tests/hermes_cli/test_plan_mode.py::TestIsPlanPath`:
- `test_guard_anchors_to_task_base_not_process_cwd` — **core regression**.
  Process cwd ≠ task base (`TERMINAL_CWD`); the `escape` symlink exists only at
  the task base. Asserts a genuine `.hermes/plans/plan.md` is allowed (True,
  proving the guard resolves at the task base) and the escape is blocked
  (False). Verified this returns **True on the old code** (bug) and **False on
  the fix**.
- `test_symlinked_plans_dir_at_task_base_stays_consistent` — plans dir itself
  is a symlink at the task base; genuine plan write allowed, traversal out
  still blocked (guard + write resolve the symlink identically).
- `test_unresolvable_path_fails_closed` — resolver raising OSError → blocked.
- `test_plans_segment_not_at_task_root_is_rejected` — `work/.hermes/plans/...`
  and `src/module/.hermes/plans/...` now blocked (see note below).
- Kept green: legit-nested, dot-dot traversal, absolute-outside,
  symlink-inside-plans-dir.

### Existing tests updated (encoded the old string-derived-root assumption)
The old guard accepted a `.hermes/plans` segment **anywhere** in the path
(string-derived root). The fix anchors the root to the task's real
`<task_base>/.hermes/plans` (matching "the plan skill saves under
`.hermes/plans/` in the active workspace"). Two assertions asserted the looser,
less-safe behavior and were updated:
- `test_plan_file_write_allowed`: `patch` path `work/.hermes/plans/p.md`
  → `.hermes/plans/feature/p.md` (genuinely under the task plans dir).
- `test_legit_nested_plan_path_passes`: dropped the `work/.hermes/plans/p.md`
  assertion; its inverse is now asserted in the new
  `test_plans_segment_not_at_task_root_is_rejected` (this is a deliberate
  tightening — `work/.hermes/plans/anything` was a mutating write to an
  arbitrary in-workspace location that plan mode is meant to forbid).
- `test_symlink_inside_plans_dir_pointing_outside_is_rejected` and
  `test_absolute_path_outside_is_rejected`: added `monkeypatch.chdir(tmp_path)`
  so the task base (process-cwd fallback, env is hermetic) matches where the
  fixtures live — the assertions themselves are unchanged.

### Counts
- `tests/hermes_cli/test_plan_mode.py`: **32 passed** (was ~28; +4 net).
- Related suites, all green: `test_plan_mode_fail_closed_guard.py` (6),
  `test_plan_ready_sequential_dispatch.py` (1), `tui_gateway/test_plan_command.py`
  (7), `gateway/test_plan_command.py` (5), `tools/test_plan_ready_tool.py` (5),
  `hermes_cli/test_plan_command_request.py` (4), `agent/test_tool_dispatch_helpers.py`.

## Deviations
- The packet said "check containment against the task's REAL plans dir
  (`<task_base>/.hermes/plans`)" AND "keep the existing nested test green". The
  existing `work/.hermes/plans/p.md` assertion contradicts a task-base-anchored
  root, so it was updated (documented above). This is the correct, stricter
  reading — it also closes a secondary bypass (arbitrary in-workspace writes via
  a nested `.hermes/plans` segment), not just the symlink divergence.
