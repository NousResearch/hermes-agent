# LAH Brain — Deploy Path, Auth Matrix, and Business Runtime Facade

Verified 2026-08-07 during CLOE_LAHB_AUTONOMOUS_AFFILIATE_RUNTIME_E2E_V1 (Lot 0 + Lot 1 implementation).

## Deploy path (what actually ships lah-brain to prod)

- Remote host: Hostinger VPS serving `leanframeworklab.com` (A: 92.113.23.69 / 92.113.16.120, IPv6 `2a02:4780:*`).
- **NO SSH from the OpenClaw VPS** (connect timeout), **NO GitHub Actions** (workflows list empty), **NO webhooks**, **NO local lah-brain container** on this box. The user blocks SSH probes to the remote VPS — do not attempt them.
- **Deploy mechanism = merge to origin/main; the remote pulls.** Evidence: `/version` commit == origin/main HEAD exactly (31c462c), and the redirect gateway (`/go/:token`) shipped via PR #202/#203 is live (302 known token / 404 unknown). So for lah-brain: implement in a clean worktree → PR → merge → verify public HTTP. Build_time lag was first observed ~hours-to-a-day after merge (2026-08-07), but 2026-08-09 (PR #207 tracking/behavioral-summary) the pull was NEAR-INSTANT: merge at 14:33, `/version` commit == merged SHA dacab5e with build_time 14:38 (~5 min). Do NOT assume a long lag — probe `/version` right after merge; a transient 503 on `/health` during the pull/restart is normal (retry after a few seconds).
- Live verification probes (read-only, allowed):
  - `curl https://leanframeworklab.com/version` → commit + build_time + version
  - `curl https://leanframeworklab.com/health` → `{"ok":true,"service":"lah-brain-v6.2",...}` (includes provider_evidence_guard + commit)
  - `curl -s -o /dev/null -w '%{http_code}' https://leanframeworklab.com/go/<known-token>` → 302; unknown token → 404

## Auth matrix — what openclaw can call with `x-admin-api-key`

- **Session-cookie only** (admin key → 401): `/money/*` (`routes/money.js` requireAuth → `isAuthenticated()` only), `/campaign-factory/*`, `/obs/*`, monolithic `src/server.js` routes.
- **Admin-key accepted** (`x-admin-api-key` == `OPENCLAW_ADMIN_API_KEY`, or HMAC session): `/openclaw-state/*`, `/approvals/*` (requireAuth falls back to admin key), `/admin/*` via requireAdminAuth (`routes/api.js`) — including `/admin/tracking/ingestion-readiness` and `/admin/analytics/behavioral-summary` (added 2026-08-09, PR #207; both read_only, provider_write=false, db_write=false).
- **Consequence:** a business-runtime facade MUST compose in-process on the lah-brain side (money is not readable via admin key over HTTP). Expose ONE admin-key endpoint; openclaw calls it once. Do not try to read money directly from openclaw.

## Business runtime facade pattern (CLOE_LAHB Lot 1 — implemented, tests green)

- **lah-brain side** `routes/business-runtime.js`: `GET /business/runtime-context` with admin-key auth (copy openclaw-state requireAuth pattern), composing in-process:
  - money → `buildMoneyReport(db, window)` + `buildQueues(...)` (roi-engine + queue-engine); surface summary + queue counts + data_loaded
  - provider → exoclick sync config + last-run + `traffic_costs` range
  - campaigns → `listHistory(db, 20)` from campaign-factory.store
  - marketplace → OFFER_MAP inventory (see below), redacted
  - tracking → `listTrackingTrustBlockers()` (export name, NOT `trackingTrustBlockersContract`)
  - receipts → `listAllReceipts(db)` from openclaw-receipt-store
  - plus generated_at / window / provenance.components / missing[] / conflicts[] / safety{read_only:true, provider_write:false, secrets_returned:false}
- **Zones trap:** derive zones from money report `by_zone` (status OBSERVED). Do NOT feed money rows into `buildSourceDna` — it expects a `computeMetrics` report shape (`report.groups[dimension]` rows with `name`/`cta_ctr`/`focus_rate`/...) and crashes on money-row shapes. Best-effort sourceDNA → status NOT_SUPPORTED when the computeMetrics shape is absent.
- **openclaw-runtime side** `src/services/business-runtime-bridge.js`:
  - `getBusinessRuntimeContext({from,to,fetchImpl,timeoutMs})` — GET facade with `LAHB_ADMIN_API_KEY`, 5s AbortController timeout, fail-closed (`LAHB_URL_REQUIRED` / `LAHB_ADMIN_API_KEY_REQUIRED` / `..._TIMEOUT` / `..._MALFORMED`), never fabricates.
  - `createBusinessStateMemoryInterface()` → `{name:'business_state', type:'business_state', readOnly:true, available, retrieve}` — the canonical memory-interface shape consumed by `createMemoryRetrievalAdapter`.
  - Wire into `src/cognitive/business-brain-bridge.js`: replace the stub `business_state` interface (default stub has `available:false`) with the bridge interface so Cloé can obtain authoritative business state without memory_search reconstruction.
  - ⚠️ ROUTE-REWIRING GAP (found 2026-08-08 at live certification): wiring the bridge into `business-brain-bridge.js` makes `business_state` available ONLY on the canary path — `createBusinessBrainBridge().route()` is gated by `OPENCLAW_COGNITIVE_FRONT_ROUTER_ENABLED` + `OPENCLAW_COGNITIVE_CANARY_ENABLED` (OFF by default in docker-compose). The `/brain/ask` acceptance path (`buildBrainAskResponse` in `src/services/readonly-operator-cli-client.js`) NEVER calls the bridge, so a real Cloé prompt answers from static memory ("je n'ai pas de visibilité en temps réel"). Unit tests of the bridge module pass while the deployed route is still uncertified — same class as `route-rewiring-gap-pattern.md`.
  - Fix that certified the path: import `getBusinessRuntimeContext`/`createBusinessStateMemoryInterface` into `readonly-operator-cli-client.js` and inject a `business_runtime_context` available_item into `cognitiveContextPack` (available_items + compact_summary + attach_to_prompt=true) after the project-knowledge block, BEFORE the LLM call. Fail-closed: when the interface is unconfigured/unavailable/errors, push an `available:false` item and append `Business runtime: UNAVAILABLE (reason)` — never fabricate numbers. VERIFY the injected bridge call by asserting the fetch to `/business/runtime-context` happened (RED→GREEN test) AND by re-running the real Cloé prompt against the deployed container.
  - Quick pre-certification probe that isolates lah-brain from the pipeline: `curl -s -o /dev/null -w '%{http_code}' https://leanframeworklab.com/business/runtime-context` → expect 401 without key; with `x-admin-api-key` → 200 + full composition (money/zones/offers redacted). Prove the endpoint first, then prove the pipeline invokes it.
- **Test pattern (lah-brain routes):** set env (`ADMIN_PASSWORD`, `SESSION_SECRET`, `INGEST_SECRET`, `OPENCLAW_ADMIN_API_KEY`, `SQLITE_PATH`, `LAH_TRACKING_BASE_URL`, `DEFAULT_OFFER_URL`) BEFORE any module load; mkdtemp fixture DB; `unloadModules()` cache-bust; ephemeral-port express app; `x-admin-api-key` header; assert no `urlTemplate` / affiliate host / `api_token` in the serialized payload.

## OFFER_MAP = configured offer inventory source (Lot 3)

- `src/redirect-gateway.js` exports `OFFER_MAP` (a Map). Entries: opaque tokens → `{urlTemplate, description, active}`.
- Live tokens (network 406295 = CrakRevenue): `jm_home_01` (Jerkmate cam PPS), `chat_home_01` (Chaturbate revshare), `mfc_home_01` (MyFreeCams revshare), `wh_doi_01` (WannaHookup DOI), `ib_home_01` (Instabang PPS), `fl_home_01` (Fling PPS), `pc_home_01` (Promptchan PPS), `eh_home_01` (eHentai revshare), `sa_home_01` (Secret.ai PPS), `dl_home_01` (Darlink PPS), `gh_home_01` (Getharder PPS) + inactive placeholders (`dt_home_01`, `ai_home_01`, `hn_home_01`, `jm_home_02`) + `default` fallback (skip in inventory).
- `urlTemplate` is **server-side only** — never send to clients; inventory output must be redacted (offer_ref/provider/offer_id/active/destination_available/execution_status, no URL). Numeric offer id parses from URL path segments `/NETWORK/OFFER/...`.

## Pitfalls hit during Lot 1 implementation

1. **Invalid-date parse order:** `new Date("bad-date").toISOString()` throws RangeError (Invalid time value) BEFORE any NaN check on the formatted string. Validate `Number.isNaN(date.getTime())` on the Date object FIRST, then `.toISOString()`, then throw `{status:400}`. Check `from >= to` after formatting.
2. **patch tool can consume a JSDoc opening `/**`:** when old_string starts just after a JSDoc comment, the fuzzy match may swallow the opener — the next `node --check` fails with `Unexpected token '*'`. After any replace touching a block adjacent to a JSDoc, verify the comment opener is intact.
3. **Fresh worktree needs npm install:** a new `git worktree add` has no `node_modules` → `Cannot find module 'express'` in tests. Run `npm install --no-audit --no-fund` before the first test run.
4. **Huge multi-lot missions can hit the per-turn tool-iteration cap mid-way.** Leave resumable state per lot: clean worktrees with uncommitted diffs, evidence JSON in `docs/mcporter/`, per-lot verdict lines in the transcript, and do NOT deploy/merge unless the lot gates actually passed. A fresh session resumes from the worktrees without re-discovery.
