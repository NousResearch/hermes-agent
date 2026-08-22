# LAHB approval lifecycle + governed create flow (openclaw-runtime ↔ lah-brain)

Observed 2026-08-11 during the OurDream bounded-discovery + multi-vertical
extension missions. Read this before any governed provider action.

## Approval state machine (lah-brain `approval_queue`)

- `POST /approvals/submit` `{action_type, payload, correlation_id}` → 201,
  status `PENDING`. Put the spend envelope in `payload.spend_envelope`
  (this IS the governance record — "write the envelope into the governance
  mechanism" = submit here).
- `POST /approvals/:id/approve` — **executes the openclaw action immediately**
  (response carries `openclaw_result`). With `EXOCLICK_LIVE_ENABLED=false` the
  executor normalizes the payload then returns `LIVE_DISABLED` and the approval
  lands in **`BLOCKED_SAFE`** (perfect fail-closed demo; zero provider calls).
- **`BLOCKED_SAFE` is a STUCK state**: `approve` and `reject` both require
  status `PENDING` (400 "Cannot approve/reject: action is already ...").
- `POST /approvals/:id/reject` — only from PENDING.
- **Supersede/revoke a non-PENDING approval** via
  `PATCH /approvals/:id/outcome` `{operator_decision, operator_note}` — the
  ONLY way to mark a BLOCKED_SAFE approval as SUPERSEDED (governance-only, no
  provider mutation). This is the pattern for revoking a superseded spend
  envelope.
- `GET /approvals/:id` → `{ok, approval}` — used by openclaw's
  `verifyApprovalWithLAHB` (`src/services/lahb-verify-approval.js`):
  requires status APPROVED (not in EXECUTED/FAILED/REJECTED), action_type match,
  and campaign_id/zone_id match when present.

## Governed create flow (CAMPAIGN_CREATE_PAUSED)

1. Submit approval (payload.target = the 9 certified fields: name, format, geo,
   device, pricing_model, bid, daily_budget, total_budget, destination_url +
   category; payload.spend_envelope = caps/arms).
2. With live enabled, approve → executor runs
   `normalizeExoClickCampaign(target)` then `createExoClickCampaignPaused`:
   - budgets ×100 to cents (`max_daily_budget`/`total_budget_limit`),
   - bid ×100 (`pricing.price`, CPM),
   - device 'mobile' → the 35 mobile device ids (certified mapping),
   - category resolved via `data/refs/category_name_to_id.json`,
   - destination_url passed through; **do NOT pass subid1 when the gateway
     bakes aff_sub1 server-side** (subid1 would call buildCrakRevenueDestinationUrl
     on the gateway URL — wrong).
3. Live-gate semantics: `CAMPAIGN_CREATE_PAUSED` bypasses the global
   executor live gate but has its OWN internal live check
   (`createExoClickCampaignPaused` returns LIVE_DISABLED when
   `EXOCLICK_LIVE_ENABLED!==true`). Since the C104A bypass removal, governed
   CAMPAIGN_PAUSE also requires the live flag. So the create/pause window needs
   `EXOCLICK_LIVE_ENABLED=true` (record before value, restore `false` after —
   container env flip = compose recreate, see lah-stack-local-operator quirks).
4. Variations for arms B/C are added AFTER create via the provider variation
   API (documented in exoclick-campaign-operations); destination_url carries
   arm A; each variation carries the next arm's gateway URL.

## Router/attribution contract (multi-vertical capable)

- `/go/:token` + OFFER_MAP are **vertical-agnostic**: one gateway already routes
  cam (jm_home_01→8780), dating (wh_doi_01→8517), AI (sa_home_01→10381) tokens.
- The EXPERIMENT pipeline gates vertical per CONTEXT (`VERTICAL_MISMATCH` is a
  BLOCKING eligibility axis). Per-arm vertical required a bounded delta:
  - `first-bounded-discovery-live-readiness.js`: eligibility loop evaluates
    `{...context, vertical: candidate.vertical ?? context.vertical}` (per-arm).
  - `multi-offer-discovery-shadow-planner.js`: arm objects gain `vertical`
    (smartlink arm vertical null, DISCOVERY_ONLY).
  - `campaign-memory-schema.js`: `vertical:` tag prefix + normalized field.
  - `affiliate-traffic-identity-reconciliation.js`: optional `offer_vertical`
    map → `fact.vertical`; **identity_id hash stays STABLE** (backward compat).
- Track URL: `t.vlmai-1.com/406295/<offer>/<creative>` for direct offers,
  `t.vlmai-5.com/406295/7709` smartlink funnel (SF_... = account-level funnel
  token, not an arm). aff_sub1 = the only SubID slot (1-slot contract).

## Smartlink classification (CrakRevenue, 2026-08 snapshot)

VERTICAL_SPECIFIC per snapshot: AI smartlink (funnel 7709 / a.vfgtf.com...),
Cam smartlink (mosaic2.camshq.com), Gay smartlink (filf.com). All
DISCOVERY_ONLY + SELECTION_BIAS_POSSIBLE; not top-level vertical arms.

## Related

- Full-suite classification when touching these modules:
  `lah-stack-local-operator` → `references/full-suite-baseline-differential.md`.
- Create payload/provider mapping details: `exoclick-campaign-operations`.
