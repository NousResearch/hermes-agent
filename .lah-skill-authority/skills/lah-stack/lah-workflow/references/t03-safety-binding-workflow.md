# T03 Safety Binding Workflow

Gated 14-phase workflow for postmortem activation and memory refresh of a single-campaign direct provider execution.

## Hard Constraints (non-negotiable)

- Do NOT discover routes, scan ports, grep source for runtime readiness, inspect historical worktrees unless explicitly requested, create manual routing overrides, or use direct provider readers when a canonical governed surface exists.
- Unknown canonical surface → `BLOCKED_SAFE`.
- Never replace an unavailable canonical surface with exploratory curls.
- Live actions (CREATE_PAUSED, bind, pre_play, PLAY) require explicit operator consent even if mission is pre-authorized.

## Phase Sequence

| # | Phase | Step | Gate |
|---|-------|------|------|
| 1 | Platform Preflight | `platform_preflight` | PASS → continue |
| 2 | Provider Reconciliation | `provider_reconciliation` | PASS → continue |
| 3 | Business Eligibility/Scope | `cloe_intent` | PASS → continue |
| 4 | Tracking/Readiness/Coverage/NBT | (derived from phases 1-3) | PASS → continue |
| 5 | Final Packet | (compile all phase results) | PASS → continue |
| 6 | Cloé Intent | `lahb_authorization` | PASS → continue |
| 7 | LAHB Authorization | `safety_precreate` | PASS → continue |
| 8 | Safety Pre-Create Gate | (9 launch safety gates) | 9/9 PASS → continue |
| 9 | Create Paused Only | `CREATE_PAUSED` | **Operator consent required** |
| 10 | Safety Binding | `bind` | PASS → continue |
| 11 | Pre-Play Hard Gate | `pre_play` | PASS → continue |
| 12 | Governed Play | `PLAY` | PASS → continue |
| 13 | Initial Live Observation | `observe` | PASS → continue |
| 14 | Final Receipt | (compile final receipt) | DONE |

## Phase 1-8: Read-Only (no provider mutations)

Phases 1-8 are read-only checks. No provider writes, no campaign mutations.

### Phase 1: Platform Preflight
- Verify GOES preflight checks pass
- Verify provider reader module exists and is functional
- Verify EXOCLICK_API_TOKEN available in systemd env
- No live mutations

### Phase 2: Provider Reconciliation
- Read live campaign state from canonical provider reader
- Read CrakRevenue public offer page via canonical offer reader
- Verify canonical offer identity uniqueness
- Verify tracking requirements (PPS_SALE conversion event)
- Verify geo/device eligibility from public page

### Phase 3: Business Eligibility/Scope (cloe_intent)
- Determine Cloé intent from mission definition
- Verify business eligibility (PROBING state, public offer, valid proposition)
- Determine business scope (mode, topology, envelope, risk level)

### Phase 4: Tracking/Readiness/Coverage/NBT
- Verify tracking configuration (conversion events)
- Check readiness via fastpath S4/S5/S6 scenarios
- Verify coverage (39/39 cells, all arms PROBING)
- Check NBT (spend < native cap, spend < budget)

### Phase 5: Final Packet
- Compile all phase results into structured packet
- Write to `/home/deploy/t03-packets/t03-final-packet.json`

### Phase 6: Cloé Intent (lahb_authorization)
- Check LAHB facade for approval status
- Verify approval scope matches mission
- Confirm `provider_write_allowed: false` for financial_authorization_only scope

### Phase 7: LAHB Authorization (safety_precreate)
- Evaluate precreate safety state via `evaluatePreCreateSafetyState()`
- Conditions: binding_state=TO_BE_BOUND_AFTER_CREATE, governed_campaign_ids=[], readerHealthy=true
- Result: `live_window_state: SAFE_PRELIVE_LIVE_WINDOW_CLOSED`

### Phase 8: Safety Pre-Create Gate (9 launch safety gates)
All 9 gates must PASS:
1. `CAMPAIGN_CREATED_PAUSED` — campaign is in Paused status
2. `ARMS_MATERIALIZED` — all 3 arms present (8529830, 8536570, 8539232)
3. `PROVIDER_READBACK_GREEN` — provider reader available, freshness=FRESH
4. `SAFETY_CAMPAIGN_BINDING_VALID` — binding_state=TO_BE_BOUND_AFTER_CREATE
5. `SAFETY_SPEND_READER_FRESH` — provider_freshness=FRESH
6. `ZONE_OBSERVABILITY_READY` — zone monitor state exists
7. `STOP_RULE_RUNTIME_READINESS` — triggered_rules=[], action=CONTINUE
8. `CAP_MATCH` — authorized_total_cents=6000, soft_stop_cents=540
9. `AUTHORITY_VALID` — LAHB approval exists and is not in terminal status

## Phase 9+: Live Mutation (operator consent required)

Phase 9 (CREATE_PAUSED) and beyond are live mutations that modify provider state. Explicit operator consent is required before proceeding.

## Key Files

- Safety provider reader: `lah-openclaw-mvp/src/services/safety/safety-provider-reader.js`
- CrakRevenue offer reader: `lah-openclaw-mvp/src/services/crakrevenue-public-offer-reader.js`
- Campaign role config: `lah-openclaw-mvp/src/services/safety/campaign-role-config.js` (PRE_CREATE_BINDING = 'TO_BE_BOUND_AFTER_CREATE')
- Launch safety gates: `lah-openclaw-mvp/src/services/safety/launch-safety-gates.js` (9 gates)
- Governor state: `lah-openclaw-mvp/data/safety/governor-state.json`
- Supervisor state: `lah-openclaw-mvp/data/safety/supervisor-state.json`
- Monitored campaigns: `lah-openclaw-mvp/data/safety/monitored-campaigns.json`
- Zone monitor state: `lah-openclaw-mvp/data/zone-monitor/zone-monitor-state.json`
- LAHB facade: `leanframeworklab.com` (LAHB_URL env)
- LAHB verify approval: `GET /approvals/:id` with x-admin-api-key

## Canonical Surface Priority

When a canonical governed surface exists, it MUST be used instead of direct provider readers or exploratory curls:
- ExoClick data → `safety-provider-reader.js` (canonical governed surface)
- CrakRevenue public offer data → `crakrevenue-public-offer-reader.js` (canonical governed surface)
- Business context → `cloe-canonical-business-context.js` (canonical governed surface)
- LAHB authorization → LAHB facade at `leanframeworklab.com` (canonical governed surface)
- Unknown canonical surface → `BLOCKED_SAFE`