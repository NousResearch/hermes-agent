# Bounded Evidence Memory Subsystem Pattern (CLOE_CAMPAIGN_MEMORY_LOT_1)

Class-level pattern for adding a bounded, evidence-safe memory subsystem to a
CLOE/OpenClaw service: schema extension, identity-based idempotency, atomic
writes, deterministic conversation routing, and a fail-closed HTTP surface.
Established during CLOE_CAMPAIGN_MEMORY_END_TO_END (Lot 1, 2026-08-04).

## When to use

- A mission requires persisting structured records (campaign memory, zone
  events, evidence events) with idempotency, conflict detection, durability,
  and a deterministic read/conversation path.
- You are extending an existing JSON-per-file store (`data/<store>/*.json`)
  rather than migrating storage.

## Architecture decisions (validated)

1. **Keep the JSON-per-file store.** Do NOT migrate to NDJSON/SQLite unless
   durability cannot be achieved. One file per record, atomic tmp+fsync+rename,
   bind-mounted rw for persistence across container recreation.

2. **Schema extension is backward-compatible.** Keep the existing tag model
   (`campaign:`, `offer:`, `geo:`...) as the query surface; ADD optional
   structured fields (metrics, metric_state, event_type, provider, freshness,
   conflict_state, provenance, payload_hash, idempotency_key). Zero and missing
   are distinct: `metric_state` uses observed_zero | missing | derived |
   not_supported.

3. **Identity-based idempotency (spec §7):**
   - `deriveIdempotencyKey` = sha256(provider | provider_account | event_type |
     source_record_id | campaign_id | event_timestamp)
   - **payload_hash must be EXCLUDED from the identity** — otherwise two writes
     with the same stable fields but different payloads get different
     identities and conflicts are undetectable (second write = ACCEPTED instead
     of CONFLICT_DETECTED). This is the #1 mistake.
   - same identity + same payload_hash → `IDEMPOTENT_REPLAY` (no new file,
     return existing id)
   - same identity + different payload_hash → `CONFLICT_DETECTED` (write new
     record marked conflict_state=conflicted, conflicts_with_record_ids,
     resolution_state=unresolved; never overwrite)
   - **Disable legacy content-dedup when a canonical identity exists** —
     otherwise identical content across two different campaigns gets falsely
     deduped (test: same content, different identity → both accepted).

4. **Atomicity + concurrency:**
   - tmp file in SAME directory as target (same fs → no cross-device rename),
     unique tmp name (`<id>.json.tmp`), fsync best-effort, rename atomic.
   - Per-identity advisory lock `.lock-<idempotency_key>` created with
     `openSync(path, 'wx')` (O_EXCL), bounded wait, then **re-check identity
     after acquiring the lock** (another process may have written while we
     waited) → IDEMPOTENT_REPLAY/CONFLICT_DETECTED determinism.
   - finally: close fd + remove lock; zero tmp/lock leftovers.
   - Legacy content-dedup (24h window) stays ONLY for records without a
     canonical identity (fallback).

5. **Corruption reporting (spec §8):** never silently skip malformed files.
   Reader returns `corruption_count` + `corrupt_files`; append refuses to
   silently ignore a corrupt tail. Tests: truncated + malformed final lines.

6. **Deterministic conversation routing:**
   - New read-only intent block must be placed BEFORE the generic
     `cloe_assistant` pre-check (which can capture "do you remember ...")
     AND before the mutation keyword check (whose legacy list contains the
     bare domain keyword).
   - **Bare-keyword trigger is a regression trap.** "find affiliate angle for
     this campaign" must NOT route to campaign memory. Gate on identity OR
     term+context: `hasIdentity = CAMP- | "campaign memory" | "mémoire de
     campagne"`; `hasTerm = campaign|campagne`; `hasMemoryContext =
     memory|history|remember|résumé|perform|failed|échou|winning|gagnant|
     creative|conversion|zone|conflit|stale|périmé|fraîch|reuse|réutilis`;
     trigger = hasIdentity OR (hasTerm AND hasMemoryContext), minus mutation
     verbs (delete/update/write/launch/create/play/pause...).
   - Apply the SAME narrowed logic in every classifier module (cognitive
     intent-classifier AND read-only conversation router) — they drift easily.
   - Verify with an 8-case focus matrix incl. the negative affiliate prompt,
     and re-run the full router baseline against the exact base SHA to prove
     zero new regressions.

7. **Fail-closed HTTP surface (kill-switch):**
   - READ route: admin auth, allowlisted filters, bounded limit/offset,
     canonical reader + deterministic summary, corruption/conflict metadata,
     read_only_proof, never provider, never mutation, bounded errors (no stack
     trace).
   - WRITE route: disabled by default via `CLOE_<STORE>_EVIDENCE_WRITE_ENABLED`
     (absent/invalid/false → 403). Accepts ONLY an evidence namespace
     (source_type=EVIDENCE_TEST, evidence_test=true, record_id
     `EVID-<LOT>-<unique>`, campaign_id `CAMP-EVIDENCE-LOT1-*`). Rejects normal
     production writes. Returns IDEMPOTENT_REPLAY / CONFLICT_DETECTED (409).
   - Enable the flag ONLY for a controlled evidence window, then disable.
   - Fail-closed matrix tests: absent, false, malformed, true+normal record,
     true+valid evidence, replay, conflict, no provider, no mutation, read
     while disabled.

## Testing routes without a live server

Express Router can be dispatched directly with stubbed req/res:

```js
function buildReq(overrides = {}) {
  const headers = { 'x-admin-api-key': KEY, ...(overrides.headers || {}) };
  return { headers, query: {}, body: {}, params: {}, url: '/query', method: 'GET',
    header(n){ return headers[String(n).toLowerCase()]; },
    get(n){ return this.header(n); }, ...overrides };
}
function buildRes() {
  return { statusCode: 200, body: null,
    status(c){ this.statusCode = c; return this; },
    json(p){ this.body = p; return this; } };
}
function invokeRouter(router, req) {
  const res = buildRes();
  req.query = { ...(req.query||{}), ...parseQueryString(req.url||'') }; // express does NOT parse query in standalone dispatch
  res.json = (p) => { res.body = p; return res; };
  router.handle(req, res, () => {});
  return { res };
}
```

Key traps:
- express `Router.handle` does NOT parse the query string without the query
  middleware — parse `req.url` manually in the harness.
- `requireAdminApiKey` calls `req.header()` — stub it or auth returns 500
  `ADMIN_API_KEY_NOT_CONFIGURED`.
- Middleware layers have no `route` property; dispatch the whole stack via
  `router.handle`, not a hand-rolled per-layer matcher.

## Baseline regression proof

Compare failure lists against a clean worktree at the EXACT base SHA:

```bash
git worktree add /tmp/<base-verify> <BASE_SHA>
cd /tmp/<base-verify>/lah-openclaw-mvp && npm ci
node --test --test-concurrency=1 <globs> 2>&1 | grep "^not ok" | sed 's/^not ok [0-9]* - //' | sort > /tmp/base-fails.txt
# same for your worktree -> /tmp/mine-fails.txt
comm -23 /tmp/mine-fails.txt /tmp/base-fails.txt   # regressions (must be empty)
comm -13 /tmp/mine-fails.txt /tmp/base-fails.txt   # fixed (informational)
```

## Deployment pitfalls (this mission)

- **Stale `GIT_COMMIT` in runtime `.env` overrides Dockerfile ENV.** The
  compose `env_file: .env` re-injects a literal `GIT_COMMIT=<old-sha>` line,
  masking the correct build-arg at runtime. Verify the RUNTIME env via
  `docker inspect <c> --format '{{range .Config.Env}}{{println .}}{{end}}'`,
  not just the image. Remove the stale line from the deploy worktree's `.env`
  and recreate the container.
- **Write-path tests pollute the real store.** Tests that write via the
  default store path (not a temp dir) leave `data/<store>/mem-*.json` files:
  first run `written`, second run `duplicate_skipped` → test fails. Move
  generated artifacts out of the worktree between runs (bounded move to /tmp,
  never rm -rf without consent). NEVER run write-path suites in the canonical
  checkout whose `data/` is bind-mounted to production.
- Build from a deploy worktree at the exact merge SHA with
  `GIT_COMMIT=<sha> docker compose build`; verify the ENV in the image; stop +
  rm the container (NOT `--force-recreate`, which errors on name conflict)
  then `docker compose up -d --no-deps <service>` with a volume override that
  keeps the canonical `data` mount.

## Files (reference implementation)

- `src/services/<store>-schema.js` — event types registry, metric states,
  payload_hash, idempotency_key, filters
- `src/services/cloe-memory-append.js` — identity dedup, atomic write, lock
- `src/services/<store>-writer.js` — thin wrapper, passes memoryEventsDir
- `src/brain/<store>-reader.js` — filters, pagination, corruption report
- `src/brain/<store>-summary.js` — stats, insufficient_evidence, classification
- `src/brain/<store>-query-interpreter.js` — intent patterns
- `src/routes/<store>.routes.js` — read route + evidence write route
- `src/cognitive/intent-classifier.js` + `readonly-conversation-router.js` —
  narrowed intent detection (identical logic in both)
