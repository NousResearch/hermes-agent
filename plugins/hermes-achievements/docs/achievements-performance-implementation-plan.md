# Hermes Achievements Performance Implementation Plan

Status: Ready for execution after hackathon review window
Constraint: Plugin remains frozen until judging is complete
Decision: `/overview` and top-banner slots are out of scope and will be removed.

---

## Phase 0 — Baseline & Safety (no behavior change)

### Task 0.1: Add perf benchmark script (local)
Objective: Repro baseline before/after.

Acceptance:
- Can print endpoint timings for `/achievements` (3 runs each, cold + warm).

### Task 0.2: Define acceptance thresholds
Objective: Lock success criteria now.

Acceptance:
- Documented SLOs:
  - `/achievements` p95 < 1s (cached)
  - max active scan jobs = 1

---

## Phase 1 — Remove unused overview/slot surface (highest certainty)

### Task 1.1: Remove `/overview` backend route
Objective: Eliminate duplicate heavy endpoint path.

Acceptance:
- `plugin_api.py` no longer exposes `/overview`.

### Task 1.2: Remove slot registration and SummarySlot frontend code
Objective: Remove cross-tab banner fetch behavior.

Acceptance:
- No `registerSlot(..."sessions:top"...)` or `registerSlot(..."analytics:top"...)`.
- No frontend call to `api("/overview")`.

### Task 1.3: Update plugin manifest
Objective: Reflect final UI scope.

Acceptance:
- `manifest.json` removes `slots` declarations.
- Tab registration remains intact.

---

## Phase 2 — Shared snapshot persistence + single-flight for `/achievements`

### Task 2.1: Introduce snapshot store abstraction + on-disk persistence
Objective: Single source of truth for Achievements data that survives process restarts.

Acceptance:
- One structure contains dataset consumed by `/achievements`.
- Repeated requests do not recompute when cache is fresh.
- Snapshot persisted at `~/.hermes/plugins/hermes-achievements/scan_snapshot.json`.

### Task 2.2: Single-flight scan coordinator
Objective: Prevent concurrent recomputes.

Acceptance:
- Simultaneous requests result in one compute run.

### Task 2.3: Refactor `/achievements` to read snapshot
Objective: Remove direct repeated compute from request path.

Acceptance:
- `/achievements` does not run independent full recompute per request when cache is valid.

---

## Phase 3 — Stale-While-Revalidate

### Task 3.1: TTL state (`FRESH`/`STALE`)
Objective: Serve immediately when stale, refresh in background.

Acceptance:
- Cached response returned quickly even when expired.
- Refresh is asynchronous.

### Task 3.2: Add `scan-status` endpoint (optional)
Objective: Let UI/ops inspect scan state.

Acceptance:
- Returns state, last success time, last duration, last error.

### Task 3.3: Add metadata fields to `/achievements`
Objective: Improve transparency.

Acceptance:
- Response includes `generated_at`, `is_stale`, maybe `scan_id`.

---

## Phase 4 — Incremental Scanning (optional but recommended)

### Task 4.1: Add per-session checkpoint file
Objective: Track session-level changes, not just global scan time.

Acceptance:
- Checkpoint persisted at `~/.hermes/plugins/hermes-achievements/scan_checkpoint.json`.
- For each session: `session_id`, fingerprint (`updated_at`/message_count/hash), and cached contribution.

### Task 4.2: Incremental aggregation
Objective: Recompute only changed/new sessions and reuse unchanged contributions.

Acceptance:
- Typical refresh time drops materially below full scan.
- Aggregate rebuild uses: subtract old contribution + add new contribution for changed sessions.

### Task 4.3: Full rebuild fallback
Objective: Preserve correctness.

Acceptance:
- Manual full rescan always possible.
- Schema/version changes invalidate checkpoint and force full rebuild.

---

## Test Plan

1. Unit tests
- Snapshot lifecycle transitions
- Dedupe logic under parallel requests
- `/achievements` response compatibility

2. Integration tests
- Opening Achievements repeatedly causes <=1 heavy scan while in-flight
- `/achievements` warm-cache load is fast
- manual rescan updates snapshot and timestamps

3. Manual benchmarks
- Compare pre/post `/achievements` timings with same history dataset

---

## Rollout Plan

1. Release internal branch with Phase 1 (remove overview/slots).
2. Validate no UI regression in Achievements tab.
3. Add Phase 2 snapshot/dedupe.
4. Add Phase 3 stale-while-revalidate + status metadata.
5. Optional: incremental scanner.

Rollback: keep old compute path behind temporary feature flag for one release window.

---

## Definition of Done

- Achievements tab remains fully functional (counts, latest, tiers, cards, filters).
- No `/overview` endpoint or slot calls remain.
- Repeated Achievements loads feel immediate after warm cache.
- Metrics/unlocks remain unchanged versus baseline.
