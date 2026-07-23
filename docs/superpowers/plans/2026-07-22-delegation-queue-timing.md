# Delegation Queue Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show truthful queue wait and execution durations for pending, running, and terminal background delegation children without changing scheduling, configuration, or message-send behavior.

**Architecture:** The existing async child record is the lifecycle source of truth. `delegate_tool.py` stamps `queued_at` when the child record is created and stamps `started_at` at the same execution boundary that registers a child in the active registry; `async_delegation.py` preserves those fields, sets `ended_at` at terminal child update, and keeps `completed_at` as a compatibility alias. The roster row builder computes timing from those fields plus an injected clock, while `gateway/run.py` continues to edit one watcher-owned bubble and also accepts aggregate records whose status is pending.

**Tech Stack:** Python 3, pytest through `scripts/run_tests.sh`, asyncio gateway tests, existing `time.time()` lifecycle clocks, and the existing roster formatter.

## Global Constraints

- Capture `queued_at` at authoritative child async-record creation, before scheduler admission.
- Capture `started_at` only at the authoritative pending-to-running execution transition.
- Capture `ended_at` at the authoritative terminal transition; retain `completed_at` compatibility.
- Use async child records plus the active-subagent registry; do not infer lifecycle from Telegram timestamps, refresh time, process discovery, or renderer-local origins.
- Missing, malformed, or inverted timestamps omit only unavailable timing and preserve legacy rendering.
- Single-child and one-or-many batch children receive identical lifecycle fields.
- Aggregate pending records remain visible during watcher refresh.
- Do not change scheduler policy, concurrency, tool schemas, configuration, or Telegram send count.
- Tests use a temporary `HERMES_HOME`; do not touch the real Hermes home.
- Do not restart the gateway or push commits.

---

### Task 1: Persist child lifecycle timestamps at delegation transitions

**Files:**
- Modify: `tools/delegate_tool.py:169-179, 2705-2727, 3861-3879`
- Modify: `tools/async_delegation.py:126-164, 184-264, 488-550, 627-678`
- Test: `tests/tools/test_async_delegation.py`

**Interfaces:**
- Consumes: existing child dictionaries passed to `dispatch_async_delegation_batch`, child `_delegate_progress_ref`, and the active-registry registration boundary.
- Produces: `_normalise_children()` records with `queued_at`, `started_at`, `ended_at`, and `completed_at`; a lifecycle-start update helper callable from the child execution transition; terminal child records with `ended_at` and compatibility `completed_at`.

- [ ] **Step 1: Write the failing lifecycle propagation tests**

Add focused tests that construct real async-record dictionaries and assert the public child shape, for example:

```python
def test_batch_children_preserve_all_lifecycle_fields():
    children = ad._normalise_children(
        [
            {
                "task_index": 0,
                "subagent_id": "sa-0",
                "goal": "one",
                "status": "pending",
                "queued_at": 100.0,
                "started_at": 110.0,
                "ended_at": 120.0,
                "completed_at": 120.0,
            }
        ],
        ["one"],
        "model",
    )
    assert children[0]["queued_at"] == 100.0
    assert children[0]["started_at"] == 110.0
    assert children[0]["ended_at"] == 120.0
    assert children[0]["completed_at"] == 120.0
```

Also add a terminal-update test that seeds `ad._records` with one pending child, calls the lifecycle update path with a completed result, and asserts `ended_at` and `completed_at` are both populated while `duration_seconds` remains the existing execution value. Include a started-transition test asserting `started_at` is written without changing the child status away from the authoritative pending value.

- [ ] **Step 2: Run the focused tests and record the expected RED failure**

Run:

```bash
HERMES_HOME="$(mktemp -d)" scripts/run_tests.sh tests/tools/test_async_delegation.py -q
```

Expected: failure because lifecycle fields are currently dropped by `_normalise_children` and no started-transition plumbing exists. Save the command and actual failing assertion/output in the implementation handoff; do not edit production code before this failure is observed.

- [ ] **Step 3: Add lifecycle fields to async child normalization and terminal updates**

Extend the normalized child dictionary without removing existing keys:

```python
"queued_at": child.get("queued_at"),
"started_at": child.get("started_at"),
"ended_at": child.get("ended_at"),
"completed_at": child.get("completed_at"),
```

At the existing terminal child update, capture one `ended_at = time.time()` value and assign both `target["ended_at"]` and `target["completed_at"]` from it. Keep the current status, execution duration, tool count, cost, and error behavior unchanged. Add the smallest helper needed to set only `started_at` on the matching child by task index or subagent id.

- [ ] **Step 4: Stamp queued and started transitions for every child path**

In the `children` loop that builds `_child_records`, capture `queued_at` before the child is exposed to async dispatch and put the same lifecycle key set on every child record. At the existing `_register_subagent(...)` execution transition, read the child’s async delegation id from `_delegate_progress_ref` and call the lifecycle-start helper with the active child id and `time.time()`. If the reference is absent, do nothing so synchronous and legacy callers retain current behavior.

- [ ] **Step 5: Run the focused lifecycle tests GREEN**

Run:

```bash
HERMES_HOME="$(mktemp -d)" scripts/run_tests.sh tests/tools/test_async_delegation.py -q
```

Expected: all tests in the focused file pass, including the new lifecycle assertions and existing async delegation coverage.

- [ ] **Step 6: Commit the lifecycle plumbing**

Do not create this commit until the focused GREEN run is fresh:

```bash
git add tools/delegate_tool.py tools/async_delegation.py tests/tools/test_async_delegation.py
git commit -m "feat: record delegation child lifecycle timestamps"
```

The implementation commit must not include the plan document.

### Task 2: Render queue and execution timing with safe compatibility fallbacks

**Files:**
- Modify: `gateway/async_subagent_roster.py:24-164`
- Modify: `gateway/subagent_roster.py:81-102, 429-555`
- Test: `tests/gateway/test_async_subagent_roster.py`
- Test: `tests/gateway/test_subagent_roster.py`

**Interfaces:**
- Consumes: child `status`, `queued_at`, `started_at`, `ended_at`, `completed_at`, existing `duration_seconds`, active registry rows, and an explicit `now` value.
- Produces: a pure timing formatter such as `format_subagent_lifecycle_timing(status, queued_at, started_at, ended_at, now) -> Optional[str]`; roster rows carrying an optional timing string while retaining existing `elapsed` fields for legacy records.

- [ ] **Step 1: Write failing renderer tests**

Add exact behavior tests using fixed timestamps:

```python
def test_pending_row_shows_queue_wait():
    record = {"children": [{
        "task_index": 0, "subagent_id": "sa-0", "goal": "wait",
        "status": "pending", "queued_at": 100.0,
    }]}
    rows = build_async_subagent_roster_rows(record, [], now=142.0)
    assert rows[0]["timing"] == "queued 42s"
    assert "◦ `wait` · queued 42s" in format_subagent_roster(rows)


def test_running_row_keeps_frozen_queue_and_live_execution():
    record = {"children": [{
        "task_index": 0, "subagent_id": "sa-0", "goal": "run",
        "status": "pending", "queued_at": 100.0, "started_at": 130.0,
    }]}
    rows = build_async_subagent_roster_rows(
        record, [{"subagent_id": "sa-0", "started_at": 130.0}], now=133.0
    )
    assert rows[0]["timing"] == "running 3s · queued 30s"
    assert "▶ `run` · running 3s · queued 30s" in format_subagent_roster(rows)


def test_terminal_row_shows_execution_and_queue_history():
    record = {"children": [{
        "task_index": 0, "subagent_id": "sa-0", "goal": "done",
        "status": "completed", "queued_at": 100.0,
        "started_at": 130.0, "ended_at": 138.0, "completed_at": 138.0,
        "duration_seconds": 8.0,
    }]}
    rows = build_async_subagent_roster_rows(record, [], now=200.0)
    assert rows[0]["timing"] == "completed in 8s · queued 30s"
```

Add one parametrized or table-driven test covering failed/interrupted terminal statuses, missing values, non-numeric values, inverted `started_at < queued_at`, and inverted `ended_at < started_at`. Assert rendering does not raise, unavailable components are omitted, and an old record with only `duration_seconds` retains its existing `· ` elapsed output.

- [ ] **Step 2: Run the renderer tests and record RED**

Run:

```bash
HERMES_HOME="$(mktemp -d)" scripts/run_tests.sh tests/gateway/test_async_subagent_roster.py tests/gateway/test_subagent_roster.py -q
```

Expected: the new timing assertions fail because rows and formatter currently expose only the legacy elapsed field.

- [ ] **Step 3: Implement the pure timing formatter and row timing fields**

Implement finite numeric coercion that rejects booleans, non-numbers, non-finite values, and negative derived durations. Build timing components by authoritative state:

```python
queue = started - queued if valid(started) and valid(queued) and started >= queued else None
execution = (
    ended - started
    if valid(ended) and valid(started) and ended >= started
    else None
)
if status == "pending" and queued is valid:
    parts = [f"queued {format_elapsed(max(0, now - queued))}"]
elif running and started is valid:
    parts = [f"running {format_elapsed(max(0, now - started))}"]
    if queue is not None:
        parts.append(f"queued {format_elapsed(queue)}")
elif terminal and execution is not None:
    parts = [f"{status} in {format_elapsed(execution)}"]
    if queue is not None:
        parts.append(f"queued {format_elapsed(queue)}")
```

Use `duration_seconds` as the existing terminal `elapsed` fallback when lifecycle execution is unavailable. Only replace the visible row timing with the lifecycle string when at least one valid lifecycle component exists; otherwise let `format_subagent_roster` produce the exact legacy row. Do not use a stray `started_at` on an authoritative pending row to change its bucket; active registry membership remains the running decision.

- [ ] **Step 4: Run renderer tests GREEN**

Run the same focused command:

```bash
HERMES_HOME="$(mktemp -d)" scripts/run_tests.sh tests/gateway/test_async_subagent_roster.py tests/gateway/test_subagent_roster.py -q
```

Expected: all timing, malformed/inverted timestamp, terminal-state, and legacy compatibility tests pass.

- [ ] **Step 5: Commit the renderer implementation**

```bash
git add gateway/async_subagent_roster.py gateway/subagent_roster.py tests/gateway/test_async_subagent_roster.py tests/gateway/test_subagent_roster.py
git commit -m "feat: render delegation queue timing"
```

### Task 3: Keep aggregate pending records visible and refresh one watcher bubble

**Files:**
- Modify: `gateway/run.py:15672-15849`
- Test: `tests/gateway/test_async_subagent_roster.py`

**Interfaces:**
- Consumes: aggregate async records with `status` equal to `pending` or `running`, child lifecycle timestamps, and the existing `_async_roster_bubbles` publisher state.
- Produces: pending aggregate rows on watcher ticks and in-place timing refreshes through `adapter.edit_message` after the first seed.

- [ ] **Step 1: Write the failing watcher tests**

Add a pending-aggregate visibility test and a timing-refresh test. The refresh test must use a controllable clock and prove one send followed by one edit:

```python
@pytest.mark.asyncio
async def test_watcher_refresh_edits_existing_bubble_for_queue_timing(monkeypatch):
    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)
    record = _record(status="pending")
    record["children"] = [{
        "task_index": 0, "subagent_id": "sa-0", "goal": "wait",
        "status": "pending", "queued_at": 100.0,
    }]
    clock = {"now": 142.0}
    monkeypatch.setattr("gateway.async_subagent_roster.time.time", lambda: clock["now"])
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "display": {"platforms": {"telegram": {"subagent_roster": "on"}}}
    })

    await runner._tick_async_delegation_rosters([record], [])
    clock["now"] = 145.0
    await runner._publish_async_delegation_roster(
        record, [], force=True, collapsed=False, allow_seed=True
    )

    assert len(adapter.sent) == 1
    assert len(adapter.edits) == 1
    assert "queued 45s" in adapter.edits[0]["content"]
```

- [ ] **Step 2: Run the watcher tests and record RED**

Run:

```bash
HERMES_HOME="$(mktemp -d)" scripts/run_tests.sh tests/gateway/test_async_subagent_roster.py -q
```

Expected: the pending aggregate test is skipped by `_tick_async_delegation_rosters`, and the timing-refresh assertion fails because the row has no lifecycle timing text.

- [ ] **Step 3: Accept pending aggregate records without changing publication ownership**

Change only the watcher record-state filter from a single running state to the existing aggregate pending/running states. Keep the same `running_ids` bubble cleanup, throttle, seed, edit, and finalizer behavior. The pending record must call `_publish_async_delegation_roster` and use the same bubble dictionary, never a second send path.

- [ ] **Step 4: Run the watcher tests GREEN**

Run:

```bash
HERMES_HOME="$(mktemp -d)" scripts/run_tests.sh tests/gateway/test_async_subagent_roster.py -q
```

Expected: the pending record is visible, duration text changes between refreshes, and the existing bubble is edited with exactly one initial send.

- [ ] **Step 5: Commit the watcher change**

```bash
git add gateway/run.py tests/gateway/test_async_subagent_roster.py
git commit -m "fix: refresh pending delegation rosters"
```

### Task 4: Run final narrow gates and verify commit scope

**Files:**
- Verify only: `tools/delegate_tool.py`
- Verify only: `tools/async_delegation.py`
- Verify only: `gateway/async_subagent_roster.py`
- Verify only: `gateway/subagent_roster.py`
- Verify only: `gateway/run.py`
- Verify tests: `tests/tools/test_async_delegation.py`, `tests/gateway/test_async_subagent_roster.py`, `tests/gateway/test_subagent_roster.py`

- [ ] **Step 1: Run the complete focused delegation/gateway test set in an isolated home**

```bash
HERMES_HOME="$(mktemp -d)" scripts/run_tests.sh \
  tests/tools/test_async_delegation.py \
  tests/gateway/test_async_subagent_roster.py \
  tests/gateway/test_subagent_roster.py -q
```

Expected: zero failures and the final pytest summary recorded with the exact passed count.

- [ ] **Step 2: Run applicable Python static checks**

Use the repository’s available formatter/linter/type commands against the changed Python files. At minimum run the project formatter/linter entry points discovered in the checkout, then run a syntax compilation check for all five changed production files:

```bash
python -m compileall -q tools/delegate_tool.py tools/async_delegation.py \
  gateway/async_subagent_roster.py gateway/subagent_roster.py gateway/run.py
```

Expected: every applicable command exits zero; any unavailable tool or baseline failure is reported explicitly rather than inferred green.

- [ ] **Step 3: Verify whitespace and exact implementation scope**

```bash
git diff --check HEAD~3..HEAD
git status --short --branch
git show --stat --oneline HEAD
```

Expected: no whitespace errors, the three implementation commits contain only the listed implementation/test files, the earlier plan commit contains only `docs/superpowers/plans/2026-07-22-delegation-queue-timing.md`, and no push/restart was performed.

- [ ] **Step 4: Record uncertainties without weakening compatibility**

Report only evidence-backed uncertainties, such as a formatter/type tool not installed or a baseline failure outside changed files. Do not add migration, scheduler changes, configuration, or extra Telegram sends to resolve speculative concerns.
