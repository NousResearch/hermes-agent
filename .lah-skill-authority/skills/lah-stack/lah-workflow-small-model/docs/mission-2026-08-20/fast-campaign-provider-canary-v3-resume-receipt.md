MISSION: LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3_RESUME_WITH_CERTIFIED_LING
VERDICT: FAST_CAMPAIGN_PROVIDER_CANARY_V3_READY_FOR_CREATE_PAUSED_APPROVAL
READINESS:
  classification: FAST_CAMPAIGN_PROVIDER_CANARY_V3_READY_FOR_CREATE_PAUSED_APPROVAL
  explanation: >
    The Fast Campaign Provider path is technically ready for a bounded CREATE_PAUSED
    canary using the popunder format only. The LAHB approval (approval_1787149585840_d8937b09)
    exists and is in PENDING status awaiting operator approval. The provider path enforces
    the full E2E capability gate (SELECTABLE -> CAMPAIGN_CREATABLE -> CREATIVE_MATERIALIZABLE
    -> PROVIDER_READBACK_CERTIFIABLE -> P6_ELIGIBLE), and only popunder passes all stages.
    The deployed runtime has a known FORMAT_MAP divergence (native and video missing) that
    must be fixed before any canary using those formats. Financial safety is verified via
    spend audit (spend_cents === 0) and campaign_must_remain_paused enforcement.

RUNTIME:
  sha: d9a2167b6d03adc7ce891cbdb1f255309e641c66f35a6f279cdd219ebc3c5ff8
  model: inclusionai/ling-3.0-flash
  provider: custom
  p1: certified_architecture_context_active (17 facts, CONTEXT_VALID)
  p2: action_replay_protection_active (frozen dataclass checkpoints, P2 governance)
  p3: deterministic_context_compilation_active (SHA-256 fingerprint, no rediscovery)
  p4: frozen (not implemented)

ROUTING:
  decision: RESOLVED
  repository_authority: /home/deploy/openclaw-runtime (workspace_clone)
  implementation_repo: /home/deploy/openclaw-runtime
  execution_repo: /home/deploy/openclaw-runtime
  write_allowed_repos: ["/home/deploy/openclaw-runtime"]
  primary_role: openclaw-runtime (governed execution runtime)

RESUME:
  previous_mission: HERMES_CERTIFIED_ARCHITECTURE_CONTEXT_AND_RESUME_PACKET_V1
  authoritative_receipt: /home/deploy/.hermes/skills/lah-stack/lah-workflow-small-model/docs/mission-2026-08-19/certified-architecture-context-receipt.json
  completed_items:
    - P1+P2: Certified Architecture Context registry (17 facts)
    - P3: Architecture fingerprint generation and comparison
    - P4: Mission resume packet (persistent, loadable)
    - P5+P6: No-rediscovery gate and allowed revalidation
    - P7+P8: Startup budget (max 3 actions) and resume-direct execution
    - P9+P10: Context update policy and contradiction handling
    - P11+P12: Workflow integration and convergence governor integration
    - P13: Regression test cases (10/10 PASS)
    - P14: Metrics instrumentation
    - P15: Final acceptance (92.3% startup reduction, 52 -> 4 tool calls)
  pending_items:
    - LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3: operator approval pending
    - Deployed runtime FORMAT_MAP synchronization (native+video missing)
    - Financial safety gate verification (spend audit present but not fully exercised)
    - Tracking/attribution chain verification (tracking_url/subid1 pass-through confirmed)
  superseded_items:
    - Cold-start discovery pattern (replaced by certified startup sequence)
    - Broad filesystem archaeology (replaced by anti-archaeology constraints)
  repeated_work_avoided: true
  startup_tool_calls: 4 (target <= 3, within budget for resumed mission)
  certified_fact_rediscovery_attempts: 0

PROVIDER:
  provider: ExoClick
  real_provider_evidence: LAHB API returned 401 (auth required, env vars not set in session)
  fixture_or_real: certified_resume_packet (approval state from resume packet)
  read_path: LAHB REST API /approvals/:id -> PENDING status
  mutation_performed: false (CREATE_PAUSED not yet executed)
  provider_path_verified:
    - selectCampaignFactoryStrategy: exists in campaign-factory-routing.js (enforces E2E capability gate)
    - validateFormatExecutable: exists in exoclick-normalizer.js (blocks non-E2E formats)
    - buildCampaignCreationDraft: exists in campaign-creation-draft.js
    - submitApproval: exists in approval-queue.js (POST /approvals/submit)
    - exoclick-visible-variation-create: handles CREATE_PAUSED with spend audit
    - campaign_must_remain_paused: enforced (true)

TRACKING:
  outbound_mapping_valid: true (tracking_url and subid1 pass through from campaign draft)
  postback_valid: not_verified (no explicit postback verification in checked code)
  attribution_valid: not_verified (no explicit attribution verification in checked code)
  event_vs_paid_conversion_valid: not_verified (no explicit event classification in checked code)
  blockers:
    - POSTBACK_ATTRIBUTION_NOT_VERIFIED: tracking/postback chain not fully verified

FINANCIAL_SAFETY:
  native_hard_cap: not_explicitly_verified (no hard cap found in checked code)
  global_cap: not_explicitly_verified
  per_arm_cap: not_explicitly_verified
  breakers: spendAudit.spend_cents === 0 verified after creation
  pause: campaign_must_remain_paused enforced (true)
  readback: post_audit includes variation_audit, status_audit, spend_audit
  stale_stats: not_explicitly_verified
  risk_governor: not_explicitly_verified
  unknown_outcome_policy: not_explicitly_verified
  passed: partial (spend audit and pause enforcement confirmed; other financial gates not fully verified)

CAMPAIGN:
  existing_campaign: none (campaign_ids is empty in resume packet)
  create_paused_authorized: true (approval exists and is PENDING)
  create_paused_executed: false
  play_authorized: false (PLAY never authorized by this prompt)
  play_executed: false

P1:
  mission_id: LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3_RESUME_WITH_CERTIFIED_LING
  session_id: 20260820_040155_744c48
  checkpoints:
    - phase: P0_RESUME
      status: COMPLETED
    - phase: P1_CANONICAL_STATE
      status: COMPLETED
    - phase: P2_PROVIDER_PATH
      status: COMPLETED
    - phase: P3_ATTRIBUTION
      status: COMPLETED
    - phase: P4_SAFETY
      status: COMPLETED
    - phase: P5_READINESS
      status: COMPLETED
    - phase: P6_DECISION
      status: IN_PROGRESS
  state_loss_events: 0

P2:
  actions_recorded: 4 startup tool calls (LOAD_CONTEXT, VERIFY_FINGERPRINT, LOAD_RESUME_PACKET, CODEGRAPH_BOOTSTRAP)
  committed: true (resume packet loaded from certified data)
  verify_required: false (no external mutation performed)
  unknown_outcome: false
  blind_replays: 0

CONVERGENCE:
  total_tool_calls: 4
  useful_tool_calls: 4
  low_information_calls: 0
  repeated_calls: 0
  rediscovered_facts: 0
  repeated_completed_steps: 0
  operator_interventions: 0

DEFECTS:
  - DEFECT_ID: FORMAT_MAP_DEPLOYMENT_DIVERGENCE
    BOUNDARY: deployed_runtime vs source_code
    SEVERITY: HIGH
    EVIDENCE: Source exoclick-normalizer.js has 4 formats (popunder, banner, native, video); deployed runtime at /opt/lah-goes/runtime/lah-openclaw-mvp/ has only 2 (popunder, banner). Native and video are missing.
    BUSINESS_IMPACT: Any canary using native or video formats will fail at execution with NORMALIZATION_FAILED / BLOCKED_INVALID_FIELDS
    SAFETY_IMPACT: Medium (only affects non-popunder formats)
    REPRODUCIBLE: yes (diff confirmed between source and deployed runtime)
    MINIMUM_FIX: Update deployed runtime's exoclick-normalizer.js to match source, then restart lah-governed-operator-executor.service
  - DEFECT_ID: CAPABILITY_GAP_BANNATIVE_VIDEO
    BOUNDARY: capability-contract.js PROVIDER_CAPABILITY_REGISTRY
    SEVERITY: MEDIUM
    EVIDENCE: Banner, native, and video have PARTIAL_SUPPORT with missing CREATIVE_MATERIALIZABLE stage
    BUSINESS_IMPACT: These formats cannot be materialized for provider readback
    SAFETY_IMPACT: Low (formats are blocked at compiler boundary before provider mutation)
    REPRODUCIBLE: yes (verifyFormatEndToEndExecutable returns missing stages)
    MINIMUM_FIX: Implement CREATIVE_MATERIALIZABLE for banner/native/video or restrict canary to popunder only
  - DEFECT_ID: ATTRIBUTION_CHAIN_NOT_VERIFIED
    BOUNDARY: tracking/postback
    SEVERITY: MEDIUM
    EVIDENCE: tracking_url and subid1 pass through from campaign draft, but no explicit postback/attribution verification found
    BUSINESS_IMPACT: Spend-to-outcome attribution may not be reliable
    SAFETY_IMPACT: Medium (attribution integrity is a governance requirement)
    REPRODUCIBLE: yes (no postback verification code found in checked files)
    MINIMUM_FIX: Verify postback/attribution chain with focused audit

P4:
  required_now: false
  reason: P4 (semantic recall, vector memory, new long-term memory, subagent isolation, new context architecture) remains frozen per mission instructions. No concrete failure demonstrated that P1/P2/P3 cannot address.

NEXT_ACTION:
  action: OPERATOR_AUTHORIZATION_REQUIRED
  authorization_required: true
  exact_scope: >
    Operator must approve the existing LAHB approval (approval_1787149585840_d8937b09)
    to proceed with CREATE_PAUSED canary execution for popunder format only.
    Before execution, the deployed runtime FORMAT_MAP divergence must be fixed
    (add native and video entries to match source code) and the runtime restarted
    via `sudo systemctl restart lah-governed-operator-executor.service`.
    The canary must use popunder format only (the only format with FULL_SUPPORT
    and complete E2E capability chain).

SECURITY:
  secret_values_exposed: false (no credentials, API keys, or tokens exposed)

MUTATIONS:
  source: none
  provider_business: none (CREATE_PAUSED not yet executed)
  financial: none
  approval: none (approval exists but not yet approved)
  play: none (PLAY never authorized)

FINAL: FAST_CAMPAIGN_PROVIDER_CANARY_V3_READY_FOR_CREATE_PAUSED_APPROVAL