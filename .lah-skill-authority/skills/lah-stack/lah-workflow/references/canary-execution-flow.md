# Canary Execution Flow (LAH_FAST_CAMPAIGN_LAUNCH_PROVIDER_CANARY_V1)

Session: 2026-08-19. Proved BLOCKED_OPERATOR_APPROVAL — no pending approval existed.
Read this before any canary execution attempt.

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
   Check: `SELECT id, action_type, status, correlation_id FROM approval_queue
   WHERE action_type = 'CAMPAIGN_CREATE_PAUSED' AND status = 'PENDING';`
   If empty → BLOCKED_OPERATOR_APPROVAL. Do NOT proceed.

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

## Execution flow (zero-spend, PAUSED only, NO PLAY)

1. Selection (via campaign factory pipeline, not standalone tool)
2. Compile → COMPILE_READY with all invariants PASS
3. Record T0 immediately before governed launch
4. CREATE_PAUSED through governed launcher (requires operator approval)
5. Provider readback → fresh campaign identity + PAUSED state
6. Materialize variations using compiled packet only
7. Provider variation readback
8. P6 certification (7/7 automatic checks)
9. Safety binding verification
10. Final state: CREATED_PAUSED_READY_TO_PLAY
11. STOP — do NOT play

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

## OpenClaw execution backend gap (discovered 2026-08-19)

The LAHB approval system's `executeApprovedAction` in `openclaw-client.js`
calls `POST {OPENCLAW_API_URL}/execute` when a `CAMPAIGN_CREATE_PAUSED`
approval is approved. However, OpenClaw is a browser automation gateway
(Playwright-based) and does NOT expose a campaign creation `/execute`
endpoint. The endpoint returns 404 locally and 400 via the LAHB-configured
URL. This means the governed execution pipeline's CREATE_PAUSED step fails
at the OpenClaw boundary with `OPENCLAW_HTTP_ERROR`.

**Impact**: The `CAMPAIGN_CREATE_PAUSED` action type cannot currently be
executed through the OpenClaw integration. The approval is accepted and
operator-approved, but the execution step fails immediately.

**Workaround**: None within the current Fast Path. The OpenClaw gateway
needs a campaign creation route, or the LAHB `executeApprovedAction`
needs to route campaign creation to a different backend (e.g., direct
ExoClick API or a dedicated campaign creation service).

**Verification**: Before relying on the governed execution path for
provider mutations, confirm the configured `OPENCLAW_API_URL` endpoint
actually supports the required operation. A 400/404 from OpenClaw at
execution time indicates a backend integration gap, not a payload or
approval issue.