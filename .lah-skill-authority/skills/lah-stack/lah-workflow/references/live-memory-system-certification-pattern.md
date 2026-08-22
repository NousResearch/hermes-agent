# Live Certification of a Governed Memory System (deployed container)

Pattern established 2026-08-10 during CLOE_CANONICAL_DECISION_CONTINUITY_AND_OPERATIONAL_LEARNING_MEMORY_V1 deploy + live cert (PR #754/#755, deployed 668ef9ac). Use when a mission requires proving cross-session memory/supersession/playbook behavior against the REAL deployed container without polluting production or opening a global write gate.

## 1. Bounded per-process write window (never edit container policy)

The production gate (`CLOE_CANONICAL_MEMORY_WRITE`) stays UNSET in `.env` and in the container (fail-closed). To certify, open the gate ONLY for the certification process, scoped to that `docker exec` invocation:

```bash
docker exec -e CLOE_CANONICAL_MEMORY_WRITE=true <container> node -e "import('/app/src/services/canonical-memory-write-seam.js').then(...)"
```

- The `-e` env applies to that exec process only; the container's own env (and `.env`) is untouched → restore to intended production policy is automatic (nothing persisted).
- The write seam forces its internal `MEMORY_APPEND_WRITE` flag when its own gate is open (same pattern as the evidence route) — do NOT set both.
- Reads never need the gate: run plain `docker exec <container> node -e "...retrieve..."`.

## 2. Synthetic namespace + scope isolation

- Use a unique namespace per cert run: `cert.memory.<timestamp>`. Every record carries `metadata.certification: true` and `metadata.cert_namespace: <ns>`.
- Scope records to a synthetic provider (`scope: { provider: 'cert-synthetic' }`) so scope matching can never match a business query.
- Proof of isolation (Phase 12): query a REAL business subject/scope and assert `business_scope_cert_leak === 0`; assert cert decisions never appear in business queries; assert the only rules visible to business scopes are the intended generic `VALIDATED_OPERATING_RULE` playbook records.
- Append-only audit policy: do NOT delete cert records; keep them isolated by namespace + scope. Mark `expires_at` in the past on the write if deactivation is required.

## 3. Session sequence (cross-session, supersession, long-session)

- SESSION A: write operator decision (ALPHA) via seam, capture record id.
- SESSION B: FRESH retrieval (new process, no session object, no transcript) — assert the decision comes back with provenance (record id + `validated_by`). This proves cross-session through canonical continuity, not in-context transcript.
- SESSION C: write superseding decision (BETA) with `supersedes_record_id: <ALPHA-id>`.
- SESSION D: fresh retrieval — assert ONLY BETA effective, ALPHA in `superseded`, zero mixture, `unresolved_ambiguity` contains no cert records (legacy evidence-route conflicts may appear — they are pre-existing, not cert).
- LONG-SESSION: persist a `continuity_marker` via `persistSessionContinuity` AFTER simulating >12 turns (the SessionCollector window), then fresh `retrieveSessionContinuity` → decision recovered with `record_ref`.

## 4. null-dir-default-override trap (retriever family)

`readCanonicalMemoryRecords({ memoryEventsDir = DEFAULT_MEMORY_EVENTS_DIR })` — a caller that defaults its own param to `null` (e.g. `retrieveSessionContinuity({ memoryEventsDir = null })`) passes EXPLICIT null, and destructuring defaults only apply to `undefined`. Result: `existsSync(null)` path reads an empty store → `no_continuity` in production while the unit test (which injects `records` explicitly) passes. Same family as `references/null-dir-default-override-pattern.md` (omit override keys when unset, never pass null), but the FIX location is the callee, not the caller:

```js
export function readCanonicalMemoryRecords({ memoryEventsDir = DEFAULT_MEMORY_EVENTS_DIR, ... } = {}) {
  const eventsDir = memoryEventsDir || DEFAULT_MEMORY_EVENTS_DIR;  // never trust the destructuring default for explicit null
  ...
}
```

Add a regression test proving `readCanonicalMemoryRecords({ memoryEventsDir: null })` does not crash and returns the default store (or a boolean store_exists), plus a live-container check that `retrieveSessionContinuity` finds a marker written through the seam.

## 5. Post-deploy verification that catches the wrong data mount

After any deploy (deployer OR manual), verify in one pass:
- `docker exec <c> sh -c 'echo $GIT_COMMIT'` == target SHA
- `EXOCLICK_LIVE_ENABLED` == false, `CLOE_CANONICAL_MEMORY_WRITE` == UNSET
- `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4000/health` == 200
- `docker inspect <c> --format '{{range .Mounts}}...'` shows canonical data source (rw) + evidence graph (ro)
- `docker exec <c> sh -c 'ls /app/data/memory-events | wc -l'` matches the pre-deploy store count (catches the temp-workdir mount)
- `RestartCount` == 0, siblings unchanged
