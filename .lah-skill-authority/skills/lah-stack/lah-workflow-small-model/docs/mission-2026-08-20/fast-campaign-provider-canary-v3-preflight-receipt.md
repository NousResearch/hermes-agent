MISSION: LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3_FINAL_PREFLIGHT_V1
MODE: READ_ONLY_REAL_RUNTIME_FINAL_GATE
VERDICT: BLOCKED_LAHB_APPROVAL_AUTHORITY

============================================================
GATE RESULTS
============================================================

APPROVAL:
  authenticated_live_readback: false
  reason: LAHB_ADMIN_API_KEY not available via canonical credential-loading mechanism (getSecret('LAHB_ADMIN_API_KEY') throws 'Missing required secret: LAHB_ADMIN_API_KEY')
  approval_id: approval_1787149585840_d8937b09
  approval_type: CAMPAIGN_CREATE_PAUSED
  status: PENDING (from resume packet, NOT authenticated live readback)
  scope: ExoClick popunder campaign creation
  campaign/provider intent: CREATE_PAUSED for popunder format only
  created_at: unknown (not read from live authority)
  expiry/validity: unknown
  LIVE_APPROVAL_READBACK: false
  BLOCKER: BLOCKED_LAHB_APPROVAL_AUTHORITY

POPUNDER:
  e2e_capability: FULL_SUPPORT (SELECTABLE -> CAMPAIGN_CREATABLE -> CREATIVE_MATERIALIZABLE -> PROVIDER_READBACK_CERTIFIABLE -> P6_ELIGIBLE)
  runtime_binding: /opt/lah-goes/runtime/lah-openclaw-mvp/src/services/exoclick-normalizer.js (deployed runtime)
  source_runtime_match_for_popunder: true (popunder entry identical: advertiser_ad_type=7, media_storage_template='link')
  FORMAT_MAP divergence (native/video): OUT OF SCOPE, not blocking popunder

TRACKING:
  outbound_valid: true
  country: passed through from campaign draft (geo field)
  zone_id: passed through from campaign draft (zone_id field)
  campaign_id: passed through from campaign draft (campaign_id field)
  creative_angle: passed through from campaign draft (angles field)
  device_placement: passed through from campaign draft (device field)
  subid1_mapping: subid1 -> aff_sub1 via buildCrakRevenueDestinationUrl()
  tracking_url: passed through from campaign draft (tracking_url field)
  OUTBOUND_TRACKING_VALID: true

POSTBACK:
  mapping_valid: true (subid1 -> aff_sub1 via CrakRevenue tracking URL builder)
  transaction_id: not explicitly verified in checked code
  payout: not explicitly verified in checked code
  offer_id: not explicitly verified in checked code
  goal_id: not explicitly verified in checked code
  POSTBACK_MAPPING_VALID: true (subid1/aff_sub1 chain confirmed)
  ATTRIBUTION_MAPPING_VALID: false (full attribution chain not verified)

ATTRIBUTION:
  mapping_valid: false
  event_paid_conversion_separation: NOT_VERIFIED (no explicit event vs paid conversion classification found in checked code)
  EVENT_PAID_CONVERSION_SEPARATION_VALID: false
  BLOCKER: BLOCKED_ATTRIBUTION (event vs paid conversion classification not verified)

FINANCIAL_SAFETY:
  paused_by_default: true (campaign_must_remain_paused: true enforced)
  native_cap: not_explicitly_verified
  global_cap: not_explicitly_verified
  per_arm_cap: not_explicitly_verified
  spend_audit: true (spendAudit.spend_cents === 0 verified after creation)
  pause: true (campaign_must_remain_paused enforced)
  readback: true (post_audit includes variation_audit, status_audit, spend_audit)
  stale_stats: not_explicitly_verified
  risk_governor: not_explicitly_verified
  unknown_outcome_policy: true (UNKNOWN_OUTCOME blocks duplicate execution, requires reconciliation)
  p2_replay_protection: true (frozen dataclass checkpoints, parent_checkpoint_id lineage)
  runtime_verified: true (popunder path bound to deployed runtime at /opt/lah-goes/runtime/)
  FINANCIAL_RUNTIME_GATE_VERIFIED: partial (spend audit, pause, unknown outcome policy confirmed; caps not fully verified)

============================================================
OUT OF SCOPE
============================================================
  native_video_format_drift: DEFERRED (known FORMAT_MAP divergence in deployed runtime, native+video missing; out of scope for this POPUNDER-only mission)
  action: DEFERRED

============================================================
SECURITY
============================================================
  secret_values_exposed: false (no credentials, API keys, or tokens printed; LAHB_ADMIN_API_KEY confirmed absent via getSecret() call, not by reading env)

============================================================
MUTATIONS
============================================================
  source: false
  provider: false
  financial: false
  approval: false
  play: false

============================================================
NEXT_ACTION
============================================================
  action: BLOCKED_LAHB_APPROVAL_AUTHORITY
  authorization_required: true
  exact_scope: >
    The LAHB_ADMIN_API_KEY is not available in the current session.
    The canonical credential-loading mechanism (getSecret('LAHB_ADMIN_API_KEY')
    from secret-accessor.js) throws 'Missing required secret: LAHB_ADMIN_API_KEY'.
    The live approval readback for approval_1787149585840_d8937b09 cannot be
    obtained. Per the mission STOP contract, this is BLOCKED_LAHB_APPROVAL_AUTHORITY.
    The operator must ensure LAHB_ADMIN_API_KEY is available in the session
    environment before the live approval readback can be performed.

============================================================
SECONDARY BLOCKERS (identified but not primary)
============================================================
  - BLOCKED_ATTRIBUTION: event vs paid conversion classification not verified
  - PARTIAL_FINANCIAL_GATE: spend audit and pause enforcement confirmed; caps/risk governor not fully verified

============================================================
FINAL
============================================================
  BLOCKED_LAHB_APPROVAL_AUTHORITY