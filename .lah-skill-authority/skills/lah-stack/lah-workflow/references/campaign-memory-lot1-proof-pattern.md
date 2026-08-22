# Campaign Memory Lot 1 — Proof & Promotion Session Notes (2026-08-04)

Established during `CLOE_HIGH_ROI_BUSINESS_CAPABILITY_GRAPH_V1 / LOT_1_CAMPAIGN_MEMORY_END_TO_END`
finalization. Deployed SHAs: 0363935 (PR #683) → f30eaea (PR #684 route null-dir fix) →
fdbd861 (PR #685 conversational null-dir + case-preserving extraction fix).

## 1. Null-dir default-override defect recurs at SIBLING call sites

PR #684 fixed the HTTP route (`src/routes/campaign-memory.routes.js`):
```js
const options = { query, limit, offset };
if (memoryDir) options.memoryEventsDir = memoryDir;   // null → reader default applies
```
The SAME bug survived in the conversational handler
(`src/services/gateway/readonly-conversation-router.js` → `handleCampaignMemoryPrompt`):
```js
const memoryDir = env?.CLOE_CAMPAIGN_MEMORY_DIR || null;
const retrieval = queryCampaignMemory({ query, memoryEventsDir: memoryDir }); // null → existsSync(null) → 'empty'
```
Runtime symptom: Chloé's canonical conversational path (canonical conversation service →
front-router → readOnlyRouter) answered "Campaign memory store does not exist yet" while the
HTTP route AND the in-container canonical reader both saw the real 35-event store.

**Lesson:** after fixing a null-overrides-default bug in ONE call site, grep every OTHER
call site of the target function for `env?.X || null` / explicit `null` before deploying.
Fix pattern is identical everywhere: build options, set the key only when truthy.

**RED test shape (R12):** use `createReadonlyConversationRouter({ env: {} })` (env param is
injectable, default `process.env`), write one event into the reader's DEFAULT dir
(`<mvp>/data/memory-events`, derived from `fileURLToPath(import.meta.url)`), await
`router.route({ prompt: 'campaign memory CAMP-X', sessionKey })`, assert `intent ===
'campaign_memory'`, answer includes the stored campaign, and answer does NOT contain
"does not exist yet". Clean the fixture file in `finally`.

## 2. Case-preserving extraction of prefixed IDs

`interpretCampaignQuery` lowercased the prompt first, then matched
`/\bCAMP[-]([a-z0-9_-]+)\b/i` → captured only the suffix: "CAMP-CONV" → `campaign_id:
"conv"`, which never matched stored `CAMP-CONV` (the schema filter exact-matches or
legacy-slices `tagValue.slice(5)`, but "conv" ≠ "CONV" — the lowercase strip also broke
that path). Fix: extract from the ORIGINAL prompt, case-preserved, full token:
```js
const original = String(userPrompt ?? '');
const campMatch = original.match(/\bCAMP[-][A-Za-z0-9_-]+\b/);
if (campMatch && !query.campaign_id) query.campaign_id = campMatch[0];
```
Same for OFFER-. Update existing tests that assert the old suffix-only value (test 28:
`'001'` → `'CAMP-001'`).

## 3. Shadow-proof fixture construction traps (10-scenario harness)

1. **`deriveIdempotencyKey` includes `event_timestamp`** — two fixtures sharing a timestamp
   (same event_type, no source_record_id) collide: the 2nd write becomes
   CONFLICT_DETECTED instead of ACCEPTED. Give every fixture a unique `event_timestamp`.
2. **`event_type` must be in the canonical registry** (campaign-memory-schema.js
   `CAMPAIGN_EVENT_TYPES`). Ad-hoc names like `CREATIVE_RESULT_OBSERVED` /
   `ZONE_PERFORMANCE_OBSERVED` → `VALIDATION_FAILED` / `blocked_validation`. Use
   `CREATIVE_ATTACHED` / `ZONE_OBSERVED` / `CAMPAIGN_STATS_OBSERVED`.
3. **Reader `maxContextChars` default 8000 truncates OLDER events** (sorted newest-first):
   a full multi-scenario fixture exceeds the serialized bound → summary silently misses the
   oldest campaigns (`summary.campaigns['CAMP-PROFIT'] === undefined` while a 4-event
   mini-test works). Pass `maxContextChars: 100000` on every `queryCampaignMemory` call in
   the harness.
4. **`createReadonlyConversationRouter().route()` is async** — forgetting `await` returns
   `{}` and the test fails with `intent: undefined`.
5. **IDEMPOTENT_REPLAY early-return leaves `conflicts_with_record_ids` undefined** (not
   `[]`) → assert `(!x || x.length === 0)`.
6. File accounting: 10 base fixtures + 1 conflict version + 1 correction = 12 files (the
   conflict legitimately writes a second version — no silent overwrite means BOTH are
   traceable). Capture `filesAfterIdempotent` BEFORE the conflict/correction writes to
   assert the identical replay added nothing.

## 4. Bounded live proof (kill-switch window)

- `CLOE_CAMPAIGN_MEMORY_EVIDENCE_WRITE_ENABLED` is read from env captured at router boot
  (`createCampaignMemoryRouter({ env: process.env })`) — NOT live-reloadable. Container
  recreate is required to change it (mission authorizes; record before/after identities).
- Window ON: extra override file `docker-compose.lot1.evidence.yml` with the env var;
  `docker stop/rm` then `GIT_COMMIT=<sha> docker compose -f docker-compose.yml -f
  docker-compose.lot1.override.yml -f docker-compose.lot1.evidence.yml up -d --no-deps
  lah-openclaw-mvp`. lah-tools-runtime untouched (StartedAt/RestartCount unchanged).
- Sequence proved: ACCEPTED → HTTP read → identical replay IDEMPOTENT_REPLAY (same id) →
  changed payload CONFLICT_DETECTED (HTTP 409) → no overwrite (record_count=2,
  conflict_count=1) → conversation path memory_boundary / provider_used=false → window OFF
  (recreate without evidence override) → 403 EVIDENCE_WRITE_DISABLED again → read
  available → **persistence across recreate** (record survives, metrics visible).
- Store count: 35 baseline → 37 after live proof (original + conflict version).

## 5. Graph promotion (§8 SHA-rebind) — measured numbers

- `evidence-policy.v1.json` defaults `source_sha_match: true` (+ `deployed_sha_match:
  true` for high-trust dims) → a candidate built at a NEW `intendedSourceSha` mechanically
  REJECTS every older receipt with `SOURCE_SHA_MISMATCH`. Not a bug — the two-phase
  pattern (bootstrap → recollect at new SHA → superseding receipts → final) is REQUIRED.
- Bootstrap collect harness:
  `node tools/cloe-evidence-pilot.mjs --app-root <clean worktree at new SHA> --ledger-root
  <ledger> --graph-staging <staging> --deployed-sha <new sha> --deployed-graph-hash <current
  served hash> --out-sources /tmp/sources.json --out-matrix /tmp/matrix.json`
- Harness facts:
  - DEFAULTS ARE STALE: `--deployed-sha` defaults to `949795a...`, `--deployed-graph-hash`
    to `f24eed48...` — MUST be overridden with real values.
  - Reads `--admin-key` or `process.env.ADMIN_API_KEY` (source the deploy `.env` with
    `set -a && . <deploy .env> && set +a` to avoid the secret-masker mangling).
  - APPENDS receipts to the ledger itself (296 → 367 = 71 appended, verify_ok true).
  - Runs a dry-run against staging; matrix + sources written to `--out-matrix`/`--out-sources`.
  - `data/memory-events` side-effects from mapped tests block every subsequent run with
    REPOSITORY_DIRTY (pre-clean with operator consent, or run from a fresh worktree).
  - Run from a FRESH `git worktree add /tmp/<name> <merge-sha>` — the deploy worktree has
    untracked compose override files (docker-compose.lot1.override.yml) that make
    `allowDirty: false` fail.
- Real dry-run on the canonical graph dir: promotions 16 / demotions 15 / changed 31. All
  15 rejection reasons were SOURCE_SHA_MISMATCH (mechanical, expected). Campaign-memory
  advanced: runtime_reachable NOT_VERIFIED→VERIFIED, shadow_verified NOT_VERIFIED→VERIFIED
  (implemented/statically_wired/tested already VERIFIED at Lot 0).
- **Watch-out:** 4 business caps (creative-inventory-lineage, governed-campaign-creation,
  business-health, tracking-attribution) drop `tested` VERIFIED→NOT_VERIFIED because
  HIGH_ROI_TEST_MAP maps campaign-memory but not those caps — plan re-attestation or
  document as expected during the recollect phase.
- Candidate identity: graph_hash f1db76e1..., receipt_set_hash 0e19baf1..., source_sha =
  deployed_sha = fdbd861.

## 6. Post-merge deploy flow (established, reused)

1. `git worktree add /tmp/cloe-lot1-deploy-v1 <merge-sha>` (or checkout in existing).
2. Build `GIT_COMMIT=<sha> docker compose -f docker-compose.yml -f
   docker-compose.lot1.override.yml build lah-openclaw-mvp`.
3. Verify image statically: `docker image inspect ... --format '{{json .Config.Env}}'` —
   MUST contain `GIT_COMMIT=<sha>` BEFORE recreate. (Do NOT use `docker run --rm` for this —
   operator denied; inspect is sufficient.)
4. `docker stop <c> && docker rm <c>` then `up -d --no-deps` (force-recreate name-conflict
   pitfall). lah-tools-runtime untouched.
5. The canonical `.env` may contain a STALE `GIT_COMMIT=` line that overrides the build
   arg at runtime — the deploy worktree `.env` must have no GIT_COMMIT line, and the
   explicit env prefix `GIT_COMMIT=<sha>` wins.
