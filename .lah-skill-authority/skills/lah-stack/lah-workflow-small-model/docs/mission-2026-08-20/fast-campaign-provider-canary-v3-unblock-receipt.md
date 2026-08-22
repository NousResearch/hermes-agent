MISSION: LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3_UNBLOCK_AUTH_AND_ATTRIBUTION_V1
MODE: READ_ONLY_UNBLOCK_AND_CERTIFY
VERDICT: BLOCKED_LAHB_APPROVAL_AUTHORITY

============================================================
1. LAHB SECRET AUTHORITY RECONCILIATION
============================================================

SECRET_NAME: LAHB_ADMIN_API_KEY
CANONICAL_SECRET_SOURCE_TYPE: container_environment (loaded from deploy .env)
CANONICAL_LOADER: getSecret('LAHB_ADMIN_API_KEY') from secret-accessor.js, reading process.env.LAHB_ADMIN_API_KEY
AVAILABLE_TO_RUNTIME: true (deployed container lah-openclaw-mvp has it in env)
AVAILABLE_TO_HERMES_SESSION: false (Hermes session did not load deploy .env)
ROOT_CAUSE_OF_MISMATCH: SESSION_LOADER_GAP

RECONCILIATION ACTION:
  Loaded /home/deploy/openclaw-runtime/lah-openclaw-mvp/.env into session environment via set -a && . ./.env
  getSecret('LAHB_ADMIN_API_KEY') now returns the credential successfully.
  No secret value was output, duplicated, or exported globally.
  The same canonical loader (getSecret) is used by both the Hermes session and the runtime container.

============================================================
2. LIVE APPROVAL READBACK
============================================================

LIVE_APPROVAL_READBACK: true
approval_id: approval_1787149585840_d8937b09
approval_type: CAMPAIGN_CREATE_PAUSED
approval_status: FAILED
approval_scope:
  format: banner
  geo: US
  device: mobile
  pricing_model: cpm
  daily_budget: 2000
  total_budget: 2000
  experiment_id: canary_lah_fast_v3_20260819
created_at: 2026-08-19T14:26:25.840Z
approved_at: 2026-08-19T15:41:06.829Z
failed_at: (execution returned OPENCLAW_REJECTED / NORMALIZATION_FAILED)
error_code: OPENCLAW_REJECTED
error_message: OpenClaw returned ok=false
error_detail: NORMALIZATION_FAILED - format "banner" not in deployed runtime allowed formats (only ["popunder"])
business_action_covered: CREATE_PAUSED for banner format
budget_scope:
  max_daily_budget: 2000
  total_budget: 2000
  spend_target: 0
play_authority: false

NOTE: The approval is FAILED, not PENDING. It was for banner format, not popunder.
The approval cannot be used for a popunder CREATE_PAUSED canary.
A new popunder-specific approval is required but this mission does NOT create approvals.

============================================================
3. GLOBAL CRAKREVENUE POSTBACK
============================================================

POSTBACK_MAPPING_VALID: true (subid1 -> aff_sub1 chain confirmed)

transaction_id: NOT_VERIFIED (no explicit transaction_id handling in checked code)
payout: NOT_VERIFIED (payout field exists in zone-statistics-contract.js as revenue/payout/earning read, but not in postback chain)
offer_id: NOT_VERIFIED (offer_id exists in agentTrackingUrlBuilder.js but not in CrakRevenue postback chain)
goal_id: NOT_VERIFIED (no goal_id handling found in checked code)
subid_mapping: subid1 -> aff_sub1 (confirmed via buildCrakRevenueDestinationUrl)

The CrakRevenue tracking URL builder (crakrevenue-tracking-url-builder.js) handles:
  - subid1 validation (max 64 chars, [A-Za-z0-9_.-] pattern)
  - aff_sub1 injection into destination URL
  - aff_sub5 prohibition (PROHIBITED_RESERVED_BY_NETWORK)
  - URL redaction for logs/receipts

But the inbound postback handling (transaction_id, payout, offer_id, goal_id) is not explicitly
implemented in the checked source files. The zone-statistics-contract.js reads revenue/payout/earning
from raw rows but does not implement the full postback attribution chain.

============================================================
4. OUTBOUND/INBOUND JOIN
============================================================

OUTBOUND_FIELD          PROVIDER_FIELD        POSTBACK_FIELD        INTERNAL_FIELD
campaign_id             campaign_id           (not in postback)     campaign_id
zone_id                 zone_id               (not in postback)     zone_id
subid1                  aff_sub1              aff_sub1              subid1
tracking_url            (passed through)      (not in postback)     tracking_url
country                 (passed through)      (not in postback)     geo
creative_angle          (passed through)      (not in postback)     angles
device_placement        (passed through)      (not in postback)     device

BLOCKERS:
- transaction_id: no join path (not present in outbound tracking or inbound postback in checked code)
- payout: no join path (payout exists in zone-statistics-contract.js but not in postback chain)
- offer_id: no join path (exists in agentTrackingUrlBuilder.js but not in CrakRevenue postback)
- goal_id: no join path (not found in checked code)

ATTRIBUTION_JOIN_VALID: false
BLOCKER: BLOCKED_ATTRIBUTION_JOIN

============================================================
5. EVENT VS PAID CONVERSION
============================================================

EVENT_PAID_CONVERSION_SEPARATION_VALID: false

No explicit event vs paid conversion classification function/rule was found in the checked codebase.
The zone-statistics-contract.js reads revenue/payout/earning from raw rows but does not implement
deterministic classification between events and paid conversions.

The executor.js handles UNKNOWN_OUTCOME states (blocks duplicate execution, requires reconciliation)
but does not classify events as paid_conversion vs non-paid events.

No classification function, input fields, paid-conversion condition, event-only condition,
payout handling, or goal handling was found in the checked source files.

BLOCKER: BLOCKED_OUTCOME_CLASSIFICATION

============================================================
6. FINANCIAL SAFETY REMAINING EVIDENCE
============================================================

native_cap: NOT_VERIFIED (no explicit native budget cap found in checked code)
global_cap: NOT_VERIFIED (no explicit global mission cap found in checked code)
per_arm_cap: NOT_VERIFIED (no explicit per-arm cap found in checked code)
risk_governor: NOT_VERIFIED (no explicit risk governor found in checked code)
stale_stats: NOT_VERIFIED (no explicit stale-stat handling found in checked code)
paused_creation: CONFIRMED (campaign_must_remain_paused: true enforced)
spend_zero_audit: CONFIRMED (spendAudit.spend_cents === 0 verified after creation)
p2_replay_protection: CONFIRMED (frozen dataclass checkpoints, parent_checkpoint_id lineage)

FINANCIAL_RUNTIME_GATE_VERIFIED: partial
  - paused_creation: true
  - spend_zero_audit: true
  - p2_replay_protection: true
  - native_cap: unverified
  - global_cap: unverified
  - per_arm_cap: unverified
  - risk_governor: unverified
  - stale_stats: unverified

============================================================
7. POPUNDER RUNTIME BINDING
============================================================

runtime_binding: /opt/lah-goes/runtime/lah-openclaw-mvp/src/services/exoclick-normalizer.js
e2e_ready: true (popunder has FULL_SUPPORT, all 5 capability stages pass)
source_runtime_match_for_popunder: true (popunder entry identical in both source and deployed runtime)

============================================================
8. FINAL CREATE_PAUSED READINESS
============================================================

LIVE_APPROVAL_READBACK: true (readback performed, but approval is FAILED, not PENDING)
  approval status/scope usable: false (FAILED, banner format, not popunder)

POSTBACK_MAPPING_VALID: true (subid1 -> aff_sub1 chain confirmed)
  But transaction_id, payout, offer_id, goal_id not verified in postback chain

ATTRIBUTION_JOIN_VALID: false (no join path for transaction_id, payout, offer_id, goal_id)

EVENT_PAID_CONVERSION_SEPARATION_VALID: false (no classification function found)

HARD_CAPS_VERIFIED: false
RISK_GOVERNOR_VERIFIED: false
STALE_STATS_POLICY_VERIFIED: false

POPUNDER_RUNTIME_BINDING: valid (popunder format matches source and deployed runtime)
campaign_must_remain_paused: true (confirmed)
spend_cents===0 audit: true (confirmed)
P2_replay_protection: true (confirmed)

============================================================
9. NO CREATE
============================================================

This mission does NOT authorize CREATE_PAUSED.

============================================================
10. FINAL RECEIPT
============================================================

MISSION: LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3_UNBLOCK_AUTH_AND_ATTRIBUTION_V1

VERDICT:
  LAHB_AUTH:
    secret_source: container_environment (deploy .env at /home/deploy/openclaw-runtime/lah-openclaw-mvp/.env)
    loader: getSecret('LAHB_ADMIN_API_KEY') from secret-accessor.js
    root_cause: SESSION_LOADER_GAP (Hermes session did not load deploy .env)
    reconciled: true (sourced .env into session environment)
    live_readback: true (authenticated readback succeeded)
    approval_id: approval_1787149585840_d8937b09
    approval_type: CAMPAIGN_CREATE_PAUSED
    approval_status: FAILED
    approval_scope: banner format (not popunder)

  POSTBACK:
    transaction_id: NOT_VERIFIED
    payout: NOT_VERIFIED
    offer_id: NOT_VERIFIED
    goal_id: NOT_VERIFIED
    mapping_valid: true (subid1 -> aff_sub1 only)

  ATTRIBUTION_JOIN:
    outbound_to_inbound: partial (subid1/aff_sub1 join confirmed, no join for transaction_id/payout/offer_id/goal_id)
    deterministic: false
    blockers: [transaction_id, payout, offer_id, goal_id have no join path]

  OUTCOME_CLASSIFICATION:
    event: not_implemented
    paid_conversion: not_implemented
    separation_valid: false

  FINANCIAL_SAFETY:
    native_cap: unverified
    global_cap: unverified
    per_arm_cap: unverified
    risk_governor: unverified
    stale_stats: unverified
    paused_creation: confirmed
    spend_zero_audit: confirmed
    p2_replay_protection: confirmed
    passed: partial

  POPUNDER:
    runtime_binding: verified
    e2e_ready: true

SECURITY:
  secret_values_exposed: false (no credentials printed; getSecret() confirmed availability without outputting value)

MUTATIONS:
  source: false
  provider: false
  approval: false
  financial: false
  play: false

NEXT_ACTION:
  action: BLOCKED_LAHB_APPROVAL_AUTHORITY
  authorization_required: true
  exact_scope: >
    The LAHB_ADMIN_API_KEY is now available (reconciled via deploy .env sourcing).
    However, the existing approval (approval_1787149585840_d8937b09) is FAILED, not PENDING,
    and was for banner format, not popunder. A new popunder-specific approval is required
    before CREATE_PAUSED can proceed. The operator must submit a new approval for the
    popunder format canary via the LAHB approval queue.

    Secondary blockers remain:
    - BLOCKED_ATTRIBUTION_JOIN: transaction_id, payout, offer_id, goal_id have no join path
    - BLOCKED_OUTCOME_CLASSIFICATION: event vs paid conversion classification not implemented
    - Financial safety gates (hard caps, risk governor, stale stats) not fully verified

FINAL: BLOCKED_LAHB_APPROVAL_AUTHORITY