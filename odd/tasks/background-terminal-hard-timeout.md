# Background Terminal Hard Timeout

## Objective
Make `terminal(background=true, timeout=N)` enforce a durable hard runtime deadline, terminate the owned process tree, emit a single truthful completion/timeout notification, and survive gateway restart without leaking descendants.

## Problem
`terminal_tool.py` accepts and plans a timeout, but the background branch does not pass it to `spawn_background_process`. `ProcessRegistry` already has checkpoint persistence, completion delivery, and verified tree-kill helpers, but `ProcessSession` has no runtime deadline or deadline watchdog. Consequently, a blocked subprocess can outlive the requested timeout and produce no completion notification.

## Why
A real CodeGraph child kept a Hermes background session alive past 900 seconds despite a requested timeout. One session became untracked; another remained running with an empty log. The defect is in the Hermes runtime contract, not in CodeGraph.

## Scope
- Thread the background timeout into process spawning.
- Persist and recover a deadline for live background sessions.
- Enforce the deadline independently of an agent polling `process(wait)`.
- Reuse the existing PID-identity and process-tree termination path.
- Deliver exactly one truthful terminal event for normal exit or timeout.
- Cover checkpoint recovery and descendant cleanup with tests.
- Update the terminal tool description/schema contract where needed.

## Constraints
- Strict TDD: observed RED, then minimal GREEN, then REFACTOR.
- Smallest coherent diff; no unrelated refactors.
- Do not modify RSI scripts, OS-Hermes, CodeGraph, `.codegraph/`, `graft/`, or `kalman_shapes.py`.
- Do not run `codegraph sync`, reindex, or mutate the CodeGraph database.
- Preserve silent daemons: no deadline when the caller intentionally omits one.
- Do not claim a process died while survivors remain; report failure honestly and keep it killable.
- No remote operations, push, PR, merge, credentials, or secret-bearing output.

## Authorized Route
Delegated direct implementation: one writer for tightly coupled runtime + tests. Parent performs independent verification and RDD review.

## Tasks
- [x] T1: Add a failing regression test proving a background timeout is forwarded and enforced.
- [x] T2: Implement deadline persistence, enforcement, tree termination, and truthful completion delivery.
- [x] T3: Add restart/recovery and descendant-cleanup regression coverage.
- [x] T4: Run focused tests, related process/terminal tests, and a real background smoke scenario.
- [x] T5: Refactor only while green; commit one coherent Conventional Commit work unit.

## Acceptance Criteria
- A background command with `timeout=N` cannot remain silently active after `N` seconds plus only the configured kill grace window.
- The timeout is independent of `process(action='wait')` polling.
- The owned root and descendants are terminated using PID-start-time-safe logic.
- A normal exit before the deadline does not trigger timeout handling.
- Timeout completion is delivered exactly once when `notify=true`.
- If a kill cannot be verified, the event says so, does not fabricate exit, and the session remains manageable.
- A recovered session with an unexpired deadline is rearmed; an expired recovered session is handled safely.
- Existing foreground timeout promotion, watcher, checkpoint, heartbeat, PTY, and silent-daemon behavior remain green.

## Verification
Use the repository's existing pytest environment. Required focused areas:
- `tests/tools/test_terminal_tool.py`
- `tests/tools/test_process_registry.py`
- `tests/tools/test_notify_on_complete.py`
- `tests/tools/test_process_heartbeat.py`
- new focused regression tests added by the implementation

Also run a real local background smoke test that creates a child, exceeds a short timeout, and proves both root and descendant are gone.

## Delivery / Review
- TDD mode: strict, source = user profile.
- Delivery strategy: single work-unit commit; target under 400 authored changed lines.
- RDD: globally on. Parent will run the native assessment/review after the commit.
- Engram mirror: pending because the Engram MCP mutation tools are unavailable in this session.

## Progress
- Research completed with local source grounding and Wigolo RDR over Node, Python, SQLite, psutil, and Hermes documentation.
- Confirmed wiring gap: `terminal_tool.py:1335-1344` does not pass `plan.effective_timeout`.
- Confirmed available primitives: `ProcessRegistry._terminate_host_pid`, `_post_kill_survivors`, atomic checkpointing, completion queue, and gateway watcher.
- Implemented one shared registry-owned deadline watchdog; explicit background timeouts are persisted absolute deadlines, while omitted timeouts remain daemon-safe.
- Recovery rearms persisted deadlines. Verified termination publishes one `timed_out` completion with real exit state; unverifiable survivors publish one actionable `timeout_error` and keep the session manageable.

## Verification Evidence
- TDD RED forwarding: `uv run --frozen python -m pytest -q tests/tools/test_terminal_task_cwd.py::test_background_command_forwards_effective_runtime_deadline` → exit 1, `KeyError: 'runtime_deadline'`.
- TDD RED enforcement: `uv run --frozen python -m pytest -q tests/tools/test_background_deadline.py::test_runtime_deadline_is_enforced_without_wait` → exit 1, `AssertionError: deadline was not enforced`.
- Focused deadline suite: `uv run --frozen python -m pytest -q tests/tools/test_background_deadline.py` → exit 0, `7 passed`.
- Final related suite: `uv run --frozen python -m pytest -q tests/tools/test_background_deadline.py tests/tools/test_terminal_task_cwd.py tests/tools/test_notify_on_complete.py tests/tools/test_process_heartbeat.py tests/tools/test_process_registry.py tests/tools/test_terminal_tool.py` → exit 0, `139 passed, 18 skipped, 1 warning`.
- Earlier focused process registry: `uv run --frozen python -m pytest -q tests/tools/test_process_registry.py` → exit 0, `97 passed, 17 skipped`.
- Earlier terminal tool: `uv run --frozen python -m pytest -q tests/tools/test_terminal_tool.py` → exit 0, `10 passed`.
- Real descendant cleanup is exercised by `test_deadline_kills_root_and_descendant_tree`: root + child PIDs are checked through `/bin/ps`, and the test passed in the focused suite. Final post-suite census found no lingering `sleep 30` process.

## Next Step
Parent performs independent verification and review; no push or PR authorized.
