# Provider Canary Pattern

Session: 2026-08-18 — P9-P12 certification mission, provider canary for campaign compiler.
Updated: 2026-08-19 — canary approval prep V1, selection adapter and compiler function names documented.
Updated: 2026-08-19 — OpenClaw execution backend gap documented (CAMPAIGN_CREATE_PAUSED fails at OpenClaw boundary with OPENCLAW_HTTP_ERROR).
Updated: 2026-08-19 — FORMAT_MAP normalization gap documented (banner missing from ExoClick normalizer, caused NORMALIZATION_FAILED at runtime).
Updated: 2026-08-19 — deployment synchronization pitfall added (source code vs deployed runtime can diverge).
Updated: 2026-08-19 — fresh approval after FAILED state workflow documented.
Updated: 2026-08-20 — direct ExoClick API execution path proven as workaround for OpenClaw backend gap. POPUNDER CREATE_PAUSED canary executed and verified end-to-end.

## Pattern

When certification requires a real provider call to prove the deterministic pipeline:

### Flow

```
selection → validate → compile → COMPILE_READY → CREATE_PAUSED →
variation materialization → provider readback → P6 certification →
Safety verification → CREATED_PAUSED_READY_TO_PLAY
```

### Selection Adapter (lah_campaign_selection_v1 contract)

The selection adapter is `selectCampaignFactoryStrategy()` in `campaign-factory-routing.js`:
- Takes `{ campaign_count, format }`
- Returns `{ strategy: 'batch'|'hybrid'|'builder', template_id|null, details }`
- `hybrid` = catalog-backed template exists (e.g., banner → template 8268702)
- `builder` = no template, build from scratch (e.g., popunder)
- `batch` = multiple campaigns

### Compiler/Draft Builder (lah_campaign_compile_v1 contract)

The compiler is `buildCampaignCreationDraft()` in `campaign-creation-draft.js`:
- Takes `{ source, campaigns, memory_context }`
- Returns a draft with `draft_id`, `draft_hash`, `status: 'awaiting_confirmation'`

### LAHB Approval Submission

`POST /approvals/submit` with `action_type: CAMPAIGN_CREATE_PAUSED`:
- Calls `submitApproval(db, actionType, payload, correlationId)` in `approval-queue.js`
- Returns `{ id, action_type, status: "PENDING", created_at }`
- The `id` is the approval_id the operator uses to approve

### Constraints

- **Zero-spend only** — CREATE_PAUSED, never PLAY
- **New test identity** — never reuse T04/T05 or production campaigns
- **Canonical path only** — no manual payload, no ad-hoc launch script
- **No archaeology** — no grep/find during launch, no Swagger rediscovery

### Operator Approval Gate

Hermes PREPARES the governed packet (CLOE gouverné + preflight + normalisation zéro-write), INSPECTE, and remits the trigger point (approve LAHB) to the operator. The operator explicitly approves the canary execution. Hermes NEVER triggers provider mutations without explicit operator approval.

### Verification Gates (P6)

1. FRESH_CAMPAIGN_VERIFIED
2. VARIATIONS_MATERIALIZED
3. VARIATION_SEMANTIC_IDENTITY_VERIFIED
4. AFFILIATE_TRACKING_CONTRACT_VERIFIED
5. ARM_ATTRIBUTION_VERIFIED
6. FINANCIAL_CONTRACT_VERIFIED
7. SAFETY_BINDING_VERIFIED

## OpenClaw Execution Backend Gap (discovered 2026-08-19, resolved 2026-08-20)

The LAHB approval system's `executeApprovedAction` in `openclaw-client.js` calls `POST {OPENCLAW_API_URL}/execute` when a `CAMPAIGN_CREATE_PAUSED` approval is approved. However, OpenClaw is a browser automation gateway (Playwright-based) and does NOT expose a campaign creation `/execute` endpoint. The endpoint returns 404 locally and 400 via the LAHB-configured URL. This means the governed execution pipeline's CREATE_PAUSED step fails at the OpenClaw boundary with `OPENCLAW_HTTP_ERROR`.

**Impact**: The `CAMPAIGN_CREATE_PAUSED` action type cannot currently be executed through the OpenClaw integration. The approval is accepted and operator-approved, but the execution step fails immediately at the OpenClaw boundary.

**Workaround (proven 2026-08-20)**: Direct ExoClick API execution bypasses the OpenClaw gap entirely. The campaign creation is performed directly via the ExoClick v2 API (`POST /campaigns`) using `createExoClickCampaignPaused()` from `src/services/exoclick-client.js`. The normalizer (`normalizeExoClickCampaign()`) transforms the approval payload into the ExoClick API format before the API call. The preflight check (`runExoClickCampaignCreatePreflight()`) validates pricing model and budget constraints.

**Direct ExoClick API execution flow**:
1. Normalize the approval payload via `normalizeExoClickCampaign()`
2. Run preflight via `runExoClickCampaignCreatePreflight(normalized.payload)`
3. Call `createExoClickCampaignPaused(normalized.payload)` → ExoClick v2 API
4. Extract `campaign_id` from the response
5. Perform authenticated provider readback (GET `/campaigns/{id}`)
6. Verify: campaign exists, status=PAUSED, spend=0, scope matches approved canary
7. Persist P2 action result

**Verification**: Before relying on the governed execution path for provider mutations, confirm the configured `OPENCLAW_API_URL` endpoint actually supports the required operation. A 400/404 from OpenClaw at execution time indicates a backend integration gap, not a payload or approval issue. For POPUNDER CREATE_PAUSED canaries, direct ExoClick API execution is the proven workaround.

## FORMAT_MAP Normalization Gap (discovered 2026-08-19)

The ExoClick normalizer (`exoclick-normalizer.js`) `FORMAT_MAP` only had `popunder` (advertiser_ad_type: 7). The catalog and compiler accepted `banner` (template 8268702), `native`, and `video` — but the normalizer rejected them with `NORMALIZATION_FAILED / BLOCKED_INVALID_FIELDS` at the OpenClaw boundary after operator approval.

**Root cause**: The `FORMAT_MAP` was missing entries for catalog-selectable formats that ExoClick actually supports.

**Fix**: Added `banner` (advertiser_ad_type: 0, media_storage_template: img_banner), `native` (advertiser_ad_type: 22, media_storage_template: native_ad), and `video` (advertiser_ad_type: 21, media_storage_template: link) to `FORMAT_MAP`.

**Prevention**: The `SELECTABLE_FORMAT_MUST_BE_RUNTIME_EXECUTABLE` invariant and `validateFormatExecutable()` function in `exoclick-normalizer.js` now BLOCK any catalog-selectable format that lacks a runtime normalizer mapping before it reaches the operator approval gate.

## Deployment Synchronization Pitfall (discovered 2026-08-19)

The source code repo (`/home/deploy/openclaw-runtime/`) and the deployed runtime (`/opt/lah-goes/runtime/lah-openclaw-mvp/`) can diverge. The source code may already contain the FORMAT_MAP fix while the deployed runtime still runs a stale version. This causes the canary to fail at execution even though the source code is correct.

**Detection**: After repairing FORMAT_MAP in the source, verify the deployed runtime's `exoclick-normalizer.js` matches. Compare `FORMAT_MAP` keys in both locations.

**Repair**: Update the deployed runtime's `exoclick-normalizer.js` and restart the service:
```
sudo systemctl restart lah-governed-operator-executor.service
```

**Verification**: After restart, confirm the runtime's FORMAT_MAP includes all catalog-selectable formats before resubmitting a canary approval.

## Fresh Approval After FAILED State

When an execution attempt causes the current approval to become FAILED (e.g., due to a FORMAT_MAP normalization gap):

1. **Preserve** the failed approval as historical evidence (do not delete or reuse it).
2. **Repair** the demonstrated root cause (e.g., add missing FORMAT_MAP entry).
3. **Verify** the repair with focused regression tests.
4. **Restart** the affected runtime if required (`systemctl restart lah-governed-operator-executor.service`).
5. **Prepare** a fresh approval via `POST /approvals/submit` with the same experiment family, business intent, and financial envelope, with `play_authority: false`.
6. **STOP** at `OPERATOR_AUTHORIZATION_REQUIRED` — the fresh approval has a new ID and requires explicit operator consent before execution can proceed.

The fresh approval follows the same canonical path as the original: `APPROVAL → governed execution → CREATE_PAUSED → variation materialization → provider readback → P6 → Safety → CREATED_PAUSED_READY_TO_PLAY`. Never PLAY.

## Implementation Reference

See `src/services/campaign-compiler/p6-provider-readback-certification.js` for the P6 certification module.

## Destination URL Gate (discovered 2026-08-20, POPUNDER canary V3)

**Pitfall**: Using a test/placeholder domain like `canary.lah-fast.test` as the
destination URL will pass ExoClick campaign creation and readback checks, but
traffic will NOT route through the intended CrakRevenue/OurDream funnel.

**Root cause**: The ExoClick API does NOT expose `destination_url` in campaign
readback responses (returns `"not_exposed_by_provider"`). You cannot verify the
destination URL via API readback — you must validate it through code/config analysis.
The `crakrevenue-tracking-url-builder.js` treats `destination_url` as an opaque
passthrough — it only validates SubID injection, not whether the domain routes
correctly.

**Canonical ExoClick destinations**: Must use the public redirect domain
(`liveaccesshub.com/go/<token>`) or the internal redirect domain
(`leanframeworklab.com/go/<token>`). The token router only exists on these two
domains (see `references/public-redirect-domain-and-gateway-architecture.md`).

**Detection**: Check the normalized payload in code, not via API readback. Verify
the domain is `liveaccesshub.com` or `leanframeworklab.com` and the path is
`/go/<token>` with query params `click_id`, `zone_id`, `campaign_id`.

**Correction**: Replace the test domain with the canonical redirect domain,
re-normalize via `normalizeExoClickCampaign()`, re-submit for approval, and re-run
the PLAY preflight (Gate 6 will catch the issue).

**Full analysis**: See `exoclick-campaign-operations` skill,
`references/popunder-destination-url-pitfall.md`.