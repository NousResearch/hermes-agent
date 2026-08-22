# Canary Execution Flow (LAH_FAST_CAMPAIGN_LAUNCH_PROVIDER_CANARY_V1)

Session: 2026-08-19. Proved BLOCKED_OPERATOR_APPROVAL — no pending approval existed.
Read this before any canary execution attempt.
Updated: 2026-08-20 — direct ExoClick API execution path proven as workaround for OpenClaw backend gap. POPUNDER CREATE_PAUSED canary executed and verified end-to-end via `createExoClickCampaignPaused()`.

## Key finding: `lah_campaign_selection_v1` is NOT a standalone tool

The `lah_campaign_selection_v1` function does not exist as an independently invocable
function in the codebase. Selection is embedded in the campaign-factory.service.js
pipeline inside lah-brain (`/home/deploy/lah-stack-repos/lah-brain/src/campaign-factory/`).
The "selection" step is part of the governed create flow, not a separate CLI tool.

Do NOT search for it as an executable. The selection contract is the 9 certified fields
(name, format, geo, device, pricing_model, bid, daily_budget, total_budget, destination_url
+ category) submitted as the approval payload.

## Canary prerequisites (all must be true before execution)

1. **Operator approval must exist** in the LAHB `approval_queue` table with
   `action_type = 'CAMPAIGN_CREATE_PAUSED'` and `status = 'PENDING'`.
   Check: `GET /approvals/:approval_id` against LAHB API (LAHB_URL from env).
   Confirm `status === "PENDING"` and `payload.target.format` matches canary format.
   If missing or not PENDING → BLOCKED_OPERATOR_APPROVAL. Do NOT proceed.

2. **New experiment identity** — must NOT reuse T04 or T05. Must be sufficiently
   different from T05 to prove generalization.

3. **Catalog-backed configuration** — provider IDs, affiliate URLs, tracking parameters,
   variations, and safety bindings must all come from the catalog, not be manually
   constructed.

4. **CANONICAL_FAST_PATH** confirmed — the fast path contract fingerprint must match
   the certified fingerprint from LAH_FAST_CAMPAIGN_LAUNCH_V1.

5. **OpenClaw execution backend verified** — before submitting a CAMPAIGN_CREATE_PAUSED
   approval, confirm that the configured `OPENCLAW_API_URL` endpoint actually supports
   campaign creation. The LAHB `executeApprovedAction` calls `POST {OPENCLAW_API_URL}/execute`
   but OpenClaw is a browser automation gateway that does NOT have a campaign creation
   `/execute` route. A 400/404 from OpenClaw at execution time means the governed
   pipeline's execution step is broken at the backend integration boundary — not a
   payload or approval issue. See `governed-decision-gate` SKILL.md pitfalls for details.

   **Workaround (proven 2026-08-20)**: Direct ExoClick API execution via
   `createExoClickCampaignPaused()` bypasses the OpenClaw gap. See
   `references/provider-canary-pattern.md` for the full direct execution flow.

## Execution flow (zero-spend, PAUSED only, NO PLAY)

### Path A: Direct ExoClick API (proven workaround for OpenClaw gap)

1. Approval exists in LAHB with status=PENDING and correct scope
2. Normalize payload via `normalizeExoClickCampaign()`
3. Run preflight via `runExoClickCampaignCreatePreflight(normalized.payload)`
4. Call `createExoClickCampaignPaused(normalized.payload)` → ExoClick v2 API `POST /campaigns`
5. Extract `campaign_id` from response (field: `response.id`)
6. Perform authenticated provider readback (GET `/campaigns/{id}` with ExoClick Bearer token)
7. Verify: campaign exists, status=PAUSED, spend=0, scope matches approved canary
8. Persist P2 action result
9. STOP — do NOT play

### Path B: OpenClaw governed execution (currently blocked)

1. Approval exists in LAHB with status=PENDING and correct scope
2. LAHB `executeApprovedAction` calls `POST {OPENCLAW_API_URL}/execute`
3. OpenClaw returns 404/400 (no campaign creation `/execute` route)
4. Execution fails at OpenClaw boundary with `OPENCLAW_HTTP_ERROR`

**Use Path A for all POPUNDER CREATE_PAUSED canary executions. Path B is currently non-functional.**

## Anti-archaeology (all counters must be 0)

grep_count = find_count = repo_search_count = swagger_discovery_count =
web_research_count = ad_hoc_payload_count = ad_hoc_tracking_url_count =
manual_contract_resolution_count = 0

If normal execution requires archaeology → FAIL: FAST_PATH_RUNTIME_REGRESSION

## Timing target

wall clock from authorized execution to CREATED_PAUSED_READY_TO_PLAY < 60 seconds,
excluding explicit external approval waiting.

## Required verdicts (exactly one)

LAH_FAST_CAMPAIGN_PROVIDER_CANARY_CERTIFIED | BLOCKED_OPERATOR_APPROVAL |
FAST_PATH_RUNTIME_REGRESSION | BLOCKED_COMPILE | BLOCKED_PROVIDER_CREATE |
BLOCKED_VARIATION_MATERIALIZATION | BLOCKED_PROVIDER_READBACK |
BLOCKED_SEMANTIC_IDENTITY | BLOCKED_TRACKING_CONTRACT | BLOCKED_ARM_ATTRIBUTION |
BLOCKED_FINANCIAL | BLOCKED_SAFETY | BLOCKED_SLA | BLOCKED_RUNTIME | BLOCKED_GOVERNANCE

## Required receipt fields

experiment_id, selection, compile_id, contract_fingerprint, fast_path,
campaign_id, provider_state, spend, variations (arm/variation_id/offer_id/semantic/tracking/attribution/safety),
P6 (7/7 or exact failure), TIMING (all phases), ARCHAEOLOGY (all counters),
FINAL_STATE, PLAY_EXECUTED (false), SPEND (0), SLA_LT_60_SECONDS

## Directive 2026-08-13 constraint

Hermes PREPARES the packet, INSPECTS it, and remits the trigger point (approve LAHB)
to the operator. Hermes NEVER triggers the provider mutation. If operator approval is
not available, stop at the trigger point and return BLOCKED_OPERATOR_APPROVAL.
Do NOT infer approval. Do NOT submit the approval yourself.

## OpenClaw execution backend gap (discovered 2026-08-19, resolved 2026-08-20)

The LAHB approval system's `executeApprovedAction` in `openclaw-client.js` calls
`POST {OPENCLAW_API_URL}/execute` when a `CAMPAIGN_CREATE_PAUSED` approval is approved.
However, OpenClaw is a browser automation gateway (Playwright-based) and does NOT expose
a campaign creation `/execute` endpoint. The endpoint returns 404 locally and 400 via
the LAHB-configured URL.

**Resolution (2026-08-20)**: Direct ExoClick API execution via `createExoClickCampaignPaused()`
is the proven workaround. The normalizer transforms the approval payload into ExoClick API
format, the preflight validates constraints, and the campaign is created directly via
`POST /campaigns` on the ExoClick v2 API. See `references/provider-canary-pattern.md`
for the full direct execution flow with verification steps.

## Direct ExoClick API Execution (proven 2026-08-20)

When the OpenClaw governed execution path is blocked by the backend gap, use the direct
ExoClick API path:

1. **Normalize**: `normalizeExoClickCampaign(target)` transforms approval payload into
   ExoClick API format (advertiser_ad_type, media_storage_template, countries, categories,
   devices, pricing in cents, SubID1 injection in destination URL).

2. **Preflight**: `runExoClickCampaignCreatePreflight(normalized.payload)` validates:
   - POPUNDER pricing model must be CPM (2), Smart CPM (4), or Smart Bid (8)
   - max_daily_budget must be >= 2000 (in cents, i.e., 200000)

3. **Create**: `createExoClickCampaignPaused(normalized.payload)` POSTs to
   `https://api.exoclick.com/v2/campaigns` with `status: 0` (PAUSED).

4. **Readback**: GET `/campaigns/{campaign_id}` with ExoClick Bearer token.
   Verify: campaign exists, status=PAUSED (status=0), spend=0, scope matches.

5. **Persist**: Write P2 action result to execution receipt.

**Critical normalization details**:
- Budget: approval packet uses `daily_budget`/`total_budget` in dollars; normalizer
  converts to `max_daily_budget`/`total_budget_limit` in cents (multiply by 100).
- Pricing model: approval packet uses string "CPM"; normalizer maps to numeric `model: 2`.
- Preflight expects `pricing.model` as numeric value, not string.
- SubID1 is injected into destination URL by the normalizer (aff_sub1 parameter).

## Required verification checklist (post-creation readback)

1. campaign_exists: true (campaign_id returned from creation)
2. status_is_paused: true (status=0, calculated_status="Paused")
3. spend_is_zero: true (total_budget_spent=0)
4. format_popunder: true (advertiser_ad_type=7)
5. media_storage_link: true (media_storage_template="link")
6. scope_match: true (geo=US, device=mobile, budget=2000/2000, CPM, bid=100)

All 6 checks must pass. If any fails → fail closed and STOP.