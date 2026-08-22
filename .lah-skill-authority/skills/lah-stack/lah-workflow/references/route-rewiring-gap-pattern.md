# Route-Rewiring Gap + Consent-Gate Deployment Pattern

Established during `CLOE_HIGH_ROI_BUSINESS_CAPABILITY_GRAPH_V1` Lot 2 (2026-08-05): provider
statistics read normalized contract (PR #686 → 3c2486a, PR #687 → 15a7d61).

## 1. Route-rewiring gap: new canonical function added, production route NOT rewired

Second observed instance (2026-08-08, CLOE_LAHB_AUTONOMOUS_AFFILIATE_RUNTIME_E2E_V1 Lot 1):
the business runtime bridge was wired into `business-brain-bridge.js` (business_state memory
interface available:true) and all module tests passed, but the `/brain/ask` route
(`buildBrainAskResponse` in readonly-operator-cli-client.js) never invoked it — the real Cloé
prompt answered from static memory. Same root cause class: extending a module is not
certification; grep the ACCEPTANCE path (the route/entrypoint the real operator hits), not just
the module's own tests. For memory-interface bridges, note the canary gate:
`createBusinessBrainBridge().route()` only runs when
`OPENCLAW_COGNITIVE_FRONT_ROUTER_ENABLED` + `OPENCLAW_COGNITIVE_CANARY_ENABLED` are set (both
OFF by default) — cabling into the adapter ≠ reachable on the acceptance path. Fix pattern here:
inject the bridge call into the cognitive context pack in the acceptance function itself,
fail-closed, and certify with the REAL prompt against the deployed container (see
`lah-brain-deploy-auth-runtime-facade-map.md` for the exact injection).

Symptom: `exoclick-stats.js` gained `fetchAdvertiserZoneStats()` (normalized contract —
currency/timezone explicit, zero-vs-missing semantics, bounded date range ≤93d, pagination,
HTTP 429 → RATE_LIMITED) but `server.js` `POST /exoclick/stats/zones` still called the old
`getAdvertiserZoneStats()` (raw rows). All 8 module-level tests passed because they exercised
the NEW function directly with mocked fetch; the deployed route still served the raw contract.

How it was caught: the HTTP runtime proof against the deployed container
(`POST /exoclick/stats/zones` with admin key) returned `mode: 'exoclick_stats_zones'` and
`verdict: 'EXOCLICK_STATS_ZONES_OK'` instead of `mode: 'exoclick_stats_zones_normalized'` /
`EXOCLICK_STATS_ZONES_NORMALIZED_OK`. Unit tests could never see this.

Root cause class: extending a module with a new canonical function is not certification —
the production route/entrypoint must serve the new contract. Grep ALL call sites
(routes, executors, optimizer loops, wiring maps in `high-roi-capabilities.mjs`) for the
old symbol after adding the new one.

Fix pattern (RED → GREEN):
1. RED route tests using the HTTP route runtime proof pattern: `createApp({})` +
   `app.listen(0, '127.0.0.1')` ephemeral port + `http.request` — assert:
   - no admin key → 401
   - mocked fetch (login exchange + stats) → response `mode` ends `_normalized`, verdict
     `..._NORMALIZED_OK`, rows carry `currency`/`timezone`, `live_sent:false`,
     `writes_performed:false`
   - out-of-range date window (e.g. 2026-01-01..2026-12-31) → `EXOCLICK_STATS_DATE_RANGE_EXCEEDED`
     with NO provider fetch (mock throws if called)
2. Rewire `server.js`: import the new function, pass through optional `paginate`/`max_pages`.
   Keep the old import if other consumers (e.g. `getStats` DI wiring) still use it.
3. GREEN, then full bounded suite + adjacent consumers, node --check, git diff --check.

## 2. Terminal consent gate blocks container recreation even with written authorization

Symptom: `docker stop lah-openclaw-mvp && docker rm lah-openclaw-mvp && GIT_COMMIT=<sha>
docker compose up -d lah-openclaw-mvp` was denied by the terminal consent gate 3× — including
after an operator message explicitly granting deployment, and a clarify prompt that timed out.
The gate also blocks `rm -rf` (even in /tmp) and shell-level `.env` secret extraction
(`grep '^ADMIN_API_KEY=' .env | cut ...`). Retrying/rephrasing is forbidden by the gate.

Working pattern that emerged (used twice for Lot 2):
1. Maximize ALL secret-free prep BEFORE the deployment moment:
   - build image with explicit `GIT_COMMIT=<merge-sha>` build arg (`GIT_COMMIT=<sha> docker compose build lah-openclaw-mvp` — NOT the compose `${GIT_COMMIT:-unknown}` default)
   - verify GIT_COMMIT in image env: `docker image inspect <img> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep GIT_COMMIT`
   - create post-merge worktree at the merge SHA + `cp -al` hardlink node_modules from a deps-present worktree + run bounded tests there (proves the merged SHA is green)
   - record lah-tools-runtime identity BEFORE (container ID, StartedAt, RestartCount) — must be unchanged after
2. Present the EXACT command to the operator (clarify with choices including "run it manually"), and persist the command + before/after verification checklist into BOTH the operator packet and the continuity JSON.
3. If the gate still denies / clarify times out: STOP. Do not retry, do not rephrase, do not achieve the same outcome another way. The operator executes the command manually (they did: "DEPLOYMENT COMPLETED MANUALLY").
4. On resume: verify deployed GIT_COMMIT matches, container identity, health 200, RestartCount=0, mounts (data rw + graph ro), lah-tools-runtime unchanged, EXOCLICK_LIVE_ENABLED=false — then continue the proof chain.

Resume markers: `AUTONOMOUS_CONTINUATION_REQUIRED` + `RESUME_FROM_CONTINUITY_FILE` pointing at the lot continuity JSON. The operator message after manual deployment carries the full runtime identity block to re-verify against.

## 3. Related: bounded live proof import (already covered elsewhere)

The live gate verdict import (`collector-live-import`, authority OPERATOR, `cloe_live_gate_verdict_v1`
fields) is documented in `references/live-gate-receipt-import-pattern.md`. Lot 1 confirmed it also
works for importing a bounded live proof whose window was executed in a PRIOR session and recorded
in the session transcript (window ON/OFF containers, kill-switch returned OFF, provider calls=0):
construct the verdict from the recorded observed assertions, keep `raw_receipt_digest` as a
placeholder digest of the gate record, import → ledger grows by 1 per capability → rebuild candidate.

## 4. Silent-fail classes when wiring new modules into the acceptance path (2026-08-08, Lot 2/5)

Three bugs from CLOE_LAHB_AUTONOMOUS_AFFILIATE_RUNTIME_E2E_V1 that unit tests could NOT see —
each was caught only by the real Cloé certification (`POST /brain/ask` with a fresh session key).

0. **Per-lot certification receipts in `docs/mcporter/` are the continuity mechanism.**
   Every lot that passes its live gate gets `CLOE_<MISSION>_LOT<n>_CERT.json` in
   `lah-openclaw-mvp/docs/mcporter/` with: verdict string, merged PR head/merge SHAs, deployed
   SHA, test counts, live-cert evidence (prompt + session key + grounded facts in the answer),
   safety flags. These files let a fresh session resume from the last certified state without
   re-auditing — the mission resume text (GATE R0) verifies worktrees against them. Write one
   per lot immediately after certification, before starting the next lot.

1. **fetchImpl must be threaded through the bridge interface, not defaulted.** 
   `createBusinessStateMemoryInterface()` reads `process.env` and defaults to `globalThis.fetch`
   unless `{ fetchImpl }` is passed. Wired without it, the mock fetch in the test never captures
   the bridge call ("business runtime bridge must be invoked" fails) because the module uses the
   real global fetch. Fix: pass `{ fetchImpl }` at the call site — `createBusinessStateMemoryInterface({ fetchImpl })`.
   General rule: any new bridge/client module called from the acceptance path must accept and
   forward the request-scoped `fetchImpl`; a module-level test that mocks fetch will otherwise
   silently test the wrong fetch.

2. **Return-shape mismatch between module and call-site wiring.**
   `getAffiliateExecutionContext()` returns `{ ok, sections, ... }` at TOP level (no `.context`
   wrapper), but the wiring in `buildBrainAskResponse` read `affiliateContext.context` → undefined
   → the whole item silently fell into the fail-closed "unavailable" branch. The unit test passed
   because it asserted `result.sections` directly. Only the live answer ("le générateur de
   proposition est indisponible (fail-closed)") exposed it. Rule: after adding a new module, verify
   the EXACT field path the wiring reads against the module's actual return shape; a fail-closed
   branch masks the bug by design.

3. **Approval-pending must not block proposal generation (PAUSED-ready semantics).**
   The governed MicrotestProposal generator returned `ok:false` + `APPROVAL_ID_REQUIRED` when no
   approval id was supplied; the wiring then surfaced "unavailable (fail-closed)" instead of a
   BLOCKED/PAUSED-ready proposal. Spec required reaching PAUSED-ready state. Fix: structural
   blockers (no candidate / invalid bounds) still block; a MISSING APPROVAL ALONE yields
   `ok:true`, `verdict MICROTEST_PROPOSAL_READY_PAUSED`, `proposal.status PAUSED_READY`,
   `approval_pending:true`, execution still blocked (`live_sent:false`). General rule: distinguish
   "hard structural blockers" from "pending approval" — the latter should produce the ready-state
   artifact, not a fail-closed error.

4. **Giant legacy test file OOM → write a focused isolated test file.**
   `test/readonly-operator-cli-client.test.js` OOMs Node's heap on this 8 GB VPS even with
   `--max-old-space-size=4096` (pre-existing, verified via stash on a worktree WITHOUT the patch).
   Adding new assertions to it is impossible; the working pattern is a NEW focused file
   (`test/business-runtime-bridge-path.test.js`, `test/affiliate-execution-context.test.js`,
   `test/configured-offer-inventory.test.js`, `test/governed-microtest-proposal.test.js`) that
   imports the acceptance module and covers the changed path. Verify the OOM is pre-existing
   (run the giant file on a clean worktree) before abandoning it.

