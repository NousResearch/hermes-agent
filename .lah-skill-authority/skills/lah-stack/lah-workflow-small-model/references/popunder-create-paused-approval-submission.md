# POPUNDER CREATE_PAUSED Approval Submission & Execution Flow

Session: 2026-08-20
Mission: LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3_SUBMIT_POPUNDER_CREATE_PAUSED_APPROVAL_V1
Verdict: POPUNDER_CREATE_PAUSED_APPROVAL_SUBMITTED_AND_PENDING_OPERATOR → POPUNDER_CREATE_PAUSED_EXECUTED_AND_VERIFIED

## Routing Resolution

- `lah-repo-router` (dry-run-route.sh) returned AMBIGUOUS initially (HISTORICAL_SIGNAL_CONFLICT between lah-core and openclaw-runtime/lah-brain)
- Resolved by providing `explicit_target: lah-brain` in the mission file
- Result: RESOLVED, execution_repo=openclaw-runtime, governance_repo=lah-brain
- Confidence: high after explicit_target

## LAHB API Configuration

- LAHB_URL: not set in .env, defaults to `https://leanframeworklab.com`
- LAHB_ADMIN_API_KEY: loaded from `/home/deploy/openclaw-runtime/lah-openclaw-mvp/.env`
- Secret loader: `getSecret('LAHB_ADMIN_API_KEY')` from `src/utils/secret-accessor.js`
- API endpoints:
  - GET `/approvals/:id` — readback
  - POST `/approvals/submit` — submission
- Auth header: `x-admin-api-key`

## Old Approval (FAILED, not reusable)

- ID: `approval_1787149585840_d8937b09`
- Status: FAILED
- Format: banner (wrong — only popunder allowed for this scope)
- Error: NORMALIZATION_FAILED / BLOCKED_INVALID_FIELDS / "Unknown format mapping"
- Error code: OPENCLAW_REJECTED

## New Approval (SUCCESS)

- ID: `approval_1787226539196_3362562f`
- Type: CAMPAIGN_CREATE_PAUSED
- Format: POPUNDER
- Status: PENDING → EXECUTED
- Created: 2026-08-20T11:48:59.196Z
- HTTP response: 201 Created

## Scope Verification (Readback)

- action_type: CAMPAIGN_CREATE_PAUSED ✅
- format: popunder ✅
- geo: US ✅
- device: mobile ✅
- budget: max_daily_budget=2000, total_budget=2000, spend_target=0 ✅
- play_authority: false ✅
- No scope mismatches detected

## Key Parameters (from certified canary state)

- provider: ExoClick
- format: POPUNDER
- experiment_id: canary_lah_fast_v3_20260819
- tracking_contract: CrakRevenue offer 10138, OURDREAM_EXOCLICK_TRACKING_CONTRACT
- attribution: deterministic POPUNDER CrakRevenue (SubID1 -> aff_sub1)
- financial_bounds: native_cap=2000, global_cap=2000, per_arm_cap=N/A
- safety: paused_required, paused_readback_required, spend_zero_required, risk_governor_required, stale_stats_fail_closed, unknown_outcome_verify_before_retry, p2_replay_protection

## Pitfalls Encountered

1. LAHB_URL not set in .env — the default `https://leanframeworklab.com` works but the code requires `LAHB_URL` env var for `lahb-state-client.js`. The `lahb-approval.js` also requires it. If LAHB_URL is missing, `submitApprovalRaw` throws `LAHB_URL_REQUIRED`.
2. The old approval's format "banner" caused NORMALIZATION_FAILED because the canary scope only allows "popunder" in the FORMAT_MAP for this experiment.
3. LAHB API GET `/approvals/:id` returns 200 with full approval object including error details. The `/api/approvals/` prefix returns 404 — use the root path.

## Execution Flow (proven 2026-08-20)

After operator approval, the campaign was created via direct ExoClick API:

1. **Normalize**: `normalizeExoClickCampaign(target)` transforms approval payload:
   - `daily_budget` (dollars) → `max_daily_budget` (cents, ×100)
   - `total_budget` (dollars) → `total_budget_limit` (cents, ×100)
   - `pricing_model` string → `pricing.model` numeric (CPM=2)
   - SubID1 injected into destination URL as `aff_sub1`
   - Format mapped to `advertiser_ad_type` (popunder=7) and `media_storage_template` (link)
   - Geo mapped to ISO3 country code (US→USA)
   - Device mapped to ExoClick device ID (mobile=2)
   - Category mapped to ExoClick category ID (general=511)

2. **Preflight**: `runExoClickCampaignCreatePreflight(normalized.payload)` validates:
   - POPUNDER pricing model must be CPM (2), Smart CPM (4), or Smart Bid (8)
   - max_daily_budget must be >= 2000 (in cents, i.e., 200000)

3. **Create**: `createExoClickCampaignPaused(normalized.payload)` POSTs to
   `https://api.exoclick.com/v2/campaigns` with `status: 0` (PAUSED).
   - Requires `EXOCLICK_LIVE_ENABLED=true` in .env
   - Requires `EXOCLICK_API_TOKEN` in .env (exchanged via `/v2/login` first)
   - Returns `{ ok: true, verdict: "EXOCLICK_CREATED_PAUSED", response: { id: campaign_id } }`

4. **Readback**: GET `/campaigns/{campaign_id}` with ExoClick Bearer token.
   - Verify: campaign exists, status=PAUSED (status=0), spend=0, scope matches
   - Check: advertiser_ad_type=7 (popunder), media_storage_template=link,
     max_daily_budget=200000, total_budget_limit=200000, pricing_model=2 (CPM), price=10000 (bid=100 cents)

5. **Persist**: Write P2 action result to execution receipt.

## Campaign Creation Result

- provider_campaign_id: 8557556
- status: Paused (status=0, calculated_status="Paused")
- total_budget_spent: 0
- max_daily_budget: 200000 (2000 in cents)
- total_budget_limit: 200000 (2000 in cents)
- advertiser_ad_type: 7 (popunder)
- media_storage_template: link
- pricing_model: 2 (CPM)
- price: 10000 (100 cents)
- destination_url: https://canary.lah-fast.test/observation?aff_sub1=canary_lah_fast_v3_20260819_popunder_us_mobile

## Direct ExoClick API Execution Notes

- The normalizer (`normalizeExoClickCampaign()`) must be called BEFORE `createExoClickCampaignPaused()`. The client function does NOT normalize the payload itself.
- The preflight check uses `payload.pricing.model` (numeric), not `payload.pricing_model` (string). Both must be present in the target payload for the normalizer and preflight to work correctly.
- Budget fields in the approval packet are in dollars; the ExoClick API expects cents. The normalizer handles this conversion automatically when `daily_budget` and `total_budget` are provided in the target payload.
- The ExoClick API returns `response.id` as the campaign_id (numeric, not string).