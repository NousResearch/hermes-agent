# Hermes Achievements Performance Spec (Post-Hackathon)

Status: Draft (no code changes yet)
Owner: hermes-achievements plugin
Scope: `dashboard/plugin_api.py` + `dashboard/dist/index.js` request behavior
Decision: **Drop `/overview` and top-banner slots**; keep only Achievements tab data path.

---

## 1) Problem Statement

Current plugin endpoints `/achievements` and `/overview` both execute a full history recomputation (`evaluate_all()`), which performs a full SessionDB scan each request.

Observed on this machine/repo:
- ~83 sessions
- ~7,125 messages
- ~3,623 tool calls
- `evaluate_all()` ~13–16s per call
- `/achievements` ~13–15s per call
- `/overview` ~12–15s per call
- Overlap between endpoints increases perceived wait.

Given current product direction, `/overview` and cross-tab top-banner slots are not needed.

---

## 2) Goals

- Keep achievement correctness unchanged.
- Keep all Achievements-tab UX/data (unlocked/discovered/secrets/highest/latest/cards).
- Remove unused summary path (`/overview`) and slot wiring.
- Make Achievements tab faster by avoiding duplicate endpoint pathways.
- Ensure at most one heavy scan can run at a time.

Non-goals (phase 1):
- Rewriting achievement rules.
- Changing badge semantics/states.

---

## 3) Endpoint Semantics (Target)

### `GET /api/plugins/hermes-achievements/achievements`
Single source endpoint for Achievements UI.
Returns full payload used by the tab:
- `achievements`
- `unlocked_count`
- `discovered_count`
- `secret_count`
- `total_count`
- `error`

### `POST /api/plugins/hermes-achievements/rescan` (optional)
Manual refresh trigger.
Prefer async trigger + immediate status response.

### `GET /api/plugins/hermes-achievements/scan-status` (optional new)
Reports scan state for UX/ops.

### Removed
- `GET /api/plugins/hermes-achievements/overview`

---

## 4) UI Scope (Target)

Keep:
- Achievements page/tab (`/achievements` in plugin tab manifest)
- All existing Achievements tab stats/cards/filters

Remove:
- Top-banner summary slot components using `sessions:top` and `analytics:top`
- Any frontend call path to `/overview`

---

## 5) Runtime State Machine (for `/achievements`)

- `FRESH`: cached snapshot age <= TTL
- `STALE`: snapshot exists but expired
- `SCANNING`: background recompute running
- `FAILED`: last recompute failed, last good snapshot still served

Rules:
1. FRESH -> serve immediately.
2. STALE + not scanning -> serve stale snapshot immediately and launch background refresh.
3. SCANNING -> do not start another scan; join single-flight in-flight job.
4. No snapshot yet -> allow one blocking bootstrap scan.

---

## 6) Caching & Invalidation

### Phase 1
- In-memory cache + persisted snapshot file.
- TTL: 60–180 seconds (configurable).
- Single-flight dedupe for scan requests.
- Persist plugin data under:
  - `~/.hermes/plugins/hermes-achievements/scan_snapshot.json`

### Phase 2
- Incremental scan checkpoints with per-session fingerprints.
- Persist checkpoint data under:
  - `~/.hermes/plugins/hermes-achievements/scan_checkpoint.json`
- Checkpoint stores, per session:
  - `session_id`
  - fingerprint (`updated_at`, message_count, or hash)
  - cached per-session contribution used for aggregate recomposition
- Scan policy:
  - First run: full scan and materialize snapshot + checkpoint.
  - Next runs: process only new/changed sessions, reuse unchanged contributions.
- Full rebuild only on:
  - schema/version change
  - checkpoint corruption
  - explicit full rescan

---

## 7) Frontend Contract

- Achievements tab requests `/achievements` once on mount.
- No slot-based summary fetches.
- If response says `is_stale=true`, UI may display “Updating in background”.
- Avoid duplicate mount-triggered calls and cancel stale requests on navigation.

---

## 8) SLO Targets

- `/achievements` p95 < 1s (cached)
- Max concurrent heavy scans: 1
- Background refresh should not block UI

---

## 9) Observability Requirements

Track:
- scan count
- scan duration avg/p95
- dedupe hit count (joined in-flight scans)
- stale-served count
- failures + last error

Expose minimal diagnostics in `/scan-status`.

---

## 10) Backward Compatibility

- Keep `/achievements` response shape backward-compatible.
- Removing `/overview` is acceptable because slot UI is intentionally removed.
- If temporary compatibility is needed, `/overview` can return static deprecation response for one release.

---

## 11) Risks

- Stale data confusion -> mitigate with `generated_at` and explicit refresh status.
- Cache invalidation bugs -> start with conservative TTL + manual rescan.
- Concurrency bugs -> protect scan section with lock/single-flight guard.
- Session mutation edge cases -> use per-session fingerprint invalidation (not global timestamp only).

---

## 12) Persistence Files (Explicit)

Plugin state directory:
- `~/.hermes/plugins/hermes-achievements/`

Files:
- `state.json` (existing): unlock tracking
- `scan_snapshot.json` (new): latest materialized achievements payload
- `scan_checkpoint.json` (new): per-session fingerprints + contributions for incremental refresh
