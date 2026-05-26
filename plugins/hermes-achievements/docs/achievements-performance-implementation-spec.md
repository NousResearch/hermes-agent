# Hermes Achievements Implementation Spec (Detailed)

This document is implementation-facing detail to execute the performance refactor later.

Decision scope: keep only Achievements tab flow; remove `/overview` + top-banner slot integration.

---

## A) Current Behavior Summary

- `evaluate_all()` performs:
  - full `scan_sessions()`
  - `SessionDB.list_sessions_rich(...)`
  - `db.get_messages(session_id)` for each session
  - text/tool regex analysis + aggregation + evaluation
- `/overview` and `/achievements` both currently call `evaluate_all()` directly.
- slot calls (`sessions:top`, `analytics:top`) currently invoke `/overview`.

Consequence: repeated full recomputes and contention.

---

## B) De-scope/Removal Changes

1. Remove backend route:
- `GET /overview`

2. Remove frontend slot usage:
- `SummarySlot` component
- `registerSlot("sessions:top")`
- `registerSlot("analytics:top")`

3. Remove manifest slot declarations:
- `"slots": ["sessions:top", "analytics:top"]`

4. Keep:
- tab route/page for Achievements
- `/achievements` endpoint and full tab rendering

---

## C) Target Internal Interfaces

### 1) `SnapshotStore`
Responsibilities:
- hold latest computed snapshot in memory
- persist/load snapshot from disk
- expose age and staleness checks

Storage path:
- `~/.hermes/plugins/hermes-achievements/scan_snapshot.json`

Methods (conceptual):
- `get()` -> snapshot | null
- `set(snapshot)`
- `is_stale(ttl_seconds)`

### 2) `ScanCoordinator`
Responsibilities:
- single-flight guard for compute jobs
- track scan status

Methods:
- `run_if_needed(force: bool = false)`
- `get_status()`

State fields:
- `state`: `idle|running|failed`
- `started_at`, `finished_at`
- `last_error`
- `run_count`

### 3) `build_snapshot()`
Responsibilities:
- execute current compute logic once
- on first run, perform full scan and materialize per-session contributions
- on subsequent runs, process only changed/new sessions via checkpoint fingerprints
- produce shape consumed by `/achievements`

Output:
- `achievements`
- count fields
- optional `scan_meta`

---

## D) Endpoint Behavior Matrix (No `/overview`)

| Endpoint | Cache fresh | Cache stale | No cache | Force rescan |
|---|---|---|---|---|
| `/achievements` | return cached | return stale + trigger bg refresh | blocking bootstrap scan | n/a |
| `/rescan` | trigger refresh | trigger refresh | trigger refresh | yes |
| `/scan-status` | status only | status only | status only | status only |

Notes:
- At most one scan run active.
- Other callers either await same run or receive stale snapshot according to policy.

---

## E) Data Shape (Proposed)

```json
{
  "generated_at": 0,
  "is_stale": false,
  "scan_meta": {
    "duration_ms": 0,
    "sessions_scanned": 0,
    "messages_scanned": 0,
    "mode": "full",
    "error": null
  },
  "achievements": [],
  "unlocked_count": 0,
  "discovered_count": 0,
  "secret_count": 0,
  "total_count": 0,
  "error": null
}
```

Compatibility guidance:
- Keep existing `/achievements` keys.
- Add metadata keys without breaking old callers.

Checkpoint file (new):
- `~/.hermes/plugins/hermes-achievements/scan_checkpoint.json`

Suggested checkpoint shape:
```json
{
  "schema_version": 1,
  "generated_at": 0,
  "sessions": {
    "<session_id>": {
      "fingerprint": {
        "updated_at": 0,
        "message_count": 0,
        "hash": "optional"
      },
      "contribution": {
        "metrics": {}
      }
    }
  }
}
```

Notes:
- fingerprint mismatch => recompute that session contribution only.
- unchanged fingerprint => reuse stored contribution.

---

## F) Concurrency Contract

- Any request path that needs fresh data must pass through single-flight coordinator.
- If a scan is running:
  - do not start second scan
  - either await in-flight run (bounded) or serve stale snapshot immediately
- lock scope must include scan start/finish state transitions.

---

## G) Error Handling Contract

- If refresh fails and prior snapshot exists:
  - return prior snapshot with `is_stale=true` and error metadata
- If refresh fails and no prior snapshot:
  - return explicit error response (current behavior equivalent)
- `scan-status` should always return last known state/error.

---

## H) Frontend Integration Contract

- Achievements page:
  - one fetch on mount to `/achievements`
  - optional background refresh indicator if stale
- no top-banner slot integration
- avoid duplicate in-flight calls during fast navigation by cancellation/debounce.

---

## I) Validation Checklist

- [ ] `/overview` route removed
- [ ] manifest has no `sessions:top`/`analytics:top` slots
- [ ] frontend has no `api("/overview")` calls
- [ ] repeated Achievements navigation does not create multiple heavy scans
- [ ] average warm load times meet SLOs
- [ ] unlock totals match pre-refactor baseline for same history
- [ ] no schema regression in `/achievements` response

---

## J) Suggested File Placement for Future Work

- backend changes: `dashboard/plugin_api.py`
- optional extraction:
  - `dashboard/perf_snapshot.py`
  - `dashboard/perf_scan_coordinator.py`
- frontend request hygiene: `dashboard/dist/index.js` (or source if available)
- plugin metadata: `dashboard/manifest.json`
- persisted runtime files:
  - `~/.hermes/plugins/hermes-achievements/state.json` (existing unlock state)
  - `~/.hermes/plugins/hermes-achievements/scan_snapshot.json` (new)
  - `~/.hermes/plugins/hermes-achievements/scan_checkpoint.json` (new)

---

## K) Post-Implementation Reporting Template

Record:
- dataset size (sessions/messages/tool calls)
- pre/post `/achievements` timings (cold/warm)
- whether single-flight dedupe triggered under repeated tab open
- any behavioral diffs in unlock counts
