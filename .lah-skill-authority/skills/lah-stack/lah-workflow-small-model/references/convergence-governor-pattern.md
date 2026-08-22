# Convergence Governor Pattern Reference

Reusable component for preventing Hermes from continuing repository/runtime
discovery after sufficient evidence exists to answer the current engineering
question.

## Overview

The convergence governor implements P1-P16 of the
HERMES_CONVERGENCE_GOVERNOR_AND_DISCOVERY_LOOP_GUARD_V1 mission.

## Components

### P1 — Command Fingerprinter

Normalizes tool/terminal actions into semantic fingerprints.

**Fingerprint fields:**
- `exact_command_hash` — SHA-256 of the exact command string
- `semantic_command_hash` — SHA-256 of the normalized command
- `tool_type` — terminal, search_files, read_file, write_file, patch, browser, etc.
- `operation_type` — grep, find, docker_inspect, process_inspect, env_inspect, etc.
- `search_target` — the search concept (e.g., "OPENCLAW_API_URL")
- `scope` — the directory/path scope
- `filters` — extracted flags and filters
- `mutation` / `read_only` — whether the action mutates state
- `execution_count` — how many times this exact command has run
- `new_evidence_count` — how many new evidence items this produced

**Normalization rules:**
- `grep -rn` → `grep -r`
- `head -10` / `head -20` → `head`
- `find | xargs grep` → `grep -r`
- Path aliases normalized where determinable

### P2 — Discovery Budget

Per-phase ceilings with `NEW_EVIDENCE_JUSTIFICATION` requirement for overages.

| Category | Default Max |
|----------|-------------|
| exact_same_command | 1 |
| same_semantic_search | 2 |
| broad_repo_search | 3 |
| global_filesystem_search | 2 |
| environment_archaeology | 3 |
| docker_process_topology | 3 |
| swagger_api_discovery | 2 |

Anti-archaeology counters (BUILD_AND_CERTIFY):
- `grep_count`, `find_count`, `repo_search_count`
- `swagger_discovery_count`, `web_research_count`
- `ad_hoc_payload_count`, `ad_hoc_tracking_url_count`
- `manual_provider_contract_resolution_count`

All must be zero for certification to pass `FAST_PATH_NO_ARCHAEOLOGY`.

### P3 — Evidence Ledger

Structured hypothesis tracking:
- `hypothesis_id`, `question`, `evidence_for`, `evidence_against`
- `unknowns`, `blocking_unknowns`, `confidence`
- `canonical_sources_found`, `next_information_needed`

When `blocking_unknowns = 0`: `EVIDENCE_SUFFICIENT = true`.

### P4 — Evidence Sufficiency Gate

`evaluateEvidenceSufficiency()` returns one of:
- `INSUFFICIENT` — not enough evidence
- `SUFFICIENT_TO_FORM_HYPOTHESIS` — can form a hypothesis
- `SUFFICIENT_TO_IMPLEMENT` — enough to implement (repair mode)
- `SUFFICIENT_TO_REPORT` — enough to report (diagnostic mode)

### P5 — Repeated Command Detection

- `EXACT_REPEAT_DETECTED` — same exact command returns materially identical result twice → third execution forbidden
- `SEMANTIC_REPEAT_DETECTED` — same semantic search family produces no new evidence twice → further execution forbidden

Emits `DISCOVERY_LOOP_DETECTED` with command family, repeat count, new evidence produced, and recommended transition.

### P6 — Information Gain Scorer

Scores each discovery action: `HIGH`, `MEDIUM`, `LOW`, `ZERO`.

`ZERO` when: repeats already-recorded evidence, returns same files/processes/configuration, does not reduce a blocking unknown, broadens scope without mission relevance.

After 3 consecutive LOW/ZERO: `FORCE_CONVERGENCE_REVIEW`.

### P7 — Side-Quest Detector

Each new discovery family must map to a current `BLOCKING_UNKNOWN_ID`. If no mapping exists: `SIDE_QUEST_BLOCKED`.

### P8 — Mutation Escalation Guard

In `DIAGNOSTIC`, `PREPARE`, `READ_ONLY`, `OFFLINE_CERTIFICATION` modes: prohibit new live/provider mutation probes without explicit authorization.

Forbidden patterns:
- `POST /execute` (hand-built)
- `POST /campaigns` (direct provider creation)
- `fake campaign` payloads
- `manual payload` construction
- `affiliate URL` mutation
- `direct provider write`

Emits `UNAUTHORIZED_DISCOVERY_MUTATION_BLOCKED`. Secrets are redacted in logs.

### P9 — Context Compaction Continuity

Before compaction: persist evidence ledger, resolved hypotheses, canonical authorities, commands executed, semantic fingerprints, blocking unknowns, phase state.

After compaction: restore them. `NO_POST_COMPACTION_REDISCOVERY` unless source fingerprint changed or evidence became stale.

### P10 — Workflow State Machine

States: `ORIENT → DISCOVER → HYPOTHESIS_READY → IMPLEMENT → VERIFY → REPORT`

For diagnostic missions: `HYPOTHESIS_READY → REPORT` (shortcut).

Once `state >= IMPLEMENT`: broad discovery prohibited unless verification exposes a new blocker → `VERIFY → DISCOVER_BLOCKER` (not full mission reset).

### P11 — Force Convergence

When loop detection triggers, automatically output `CONVERGENCE_CHECK`:
1. What do I already know?
2. What exact unknown is blocking progress?
3. Is that unknown actually required?
4. What single narrow action resolves it?

If blocking unknown = NONE: transition immediately. Do not ask the operator. Do not continue searching.

#### Negative Evidence Evaluation (T05 Fix)

`generateConvergenceCheck()` now evaluates snapshot data for clear negative evidence before defaulting to PROCEED_WITH_CAUTION. When a `decisionContext` (snapshot data) is provided, the convergence check may produce a stronger verdict:

| Verdict | When |
|---------|------|
| TERMINATE | spend > 0, revenue = 0, conversions = 0, no positive signal, information_readiness = READY |
| BLOCKED_CANONICAL_DATA | data_quality.status = FAIL |
| PROCEED_WITH_CAUTION | blocking unknown exists but no decisive negative evidence |
| TRANSITION_IMMEDIATELY | no blocking unknown |

The `_evaluateNegativeEvidence(decisionContext, knownFacts)` method inspects the snapshot's economics, funnel, zones, and decision_inputs to determine whether the blocking unknown is actually blocking a decision or whether the data supports a stronger verdict.

**Pitfall: PARTIAL ≠ Insufficient for Termination**

A PARTIAL data_quality status with SPEND_WITHOUT_REVENUE warning is observed negative evidence, not missing data. Do not automatically suppress business decisions (TERMINATE) when the snapshot has clear negative evidence and information readiness is READY. Only BLOCKED_CANONICAL_DATA when data_quality is FAIL.

### P12 — Stop Conditions

STOP DISCOVERY when ANY is true:
1. Canonical authority identified AND exact defect localized
2. Smallest repair is clear
3. Remaining unknowns do not affect repair correctness
4. Mission acceptance criteria can already be evaluated
5. Repeated searches provide zero new evidence

### P15 — Hard Safety Limit

Emergency discovery ceiling: 50 discovery actions without transitioning to IMPLEMENT or REPORT → `HARD_CONVERGENCE_TRIGGER`.

Threshold is configurable via `hardCeiling` option.

### P16 — Convergence Governor Enforcement Proof

The convergence governor must distinguish between three distinct enforcement states. These are NOT interchangeable.

| State | Meaning | Satisfies Runtime Enforcement? |
|-------|---------|-------------------------------|
| `RUNTIME_ENFORCED` | An actual tool-dispatch gate is active and blocking unauthorized actions | YES |
| `BEHAVIORAL_ONLY` | The LLM is following instructions voluntarily | NO |
| `NOT_ACTIVE` | No governor is active | NO |

**Machine-verifiable proof is required for RUNTIME_ENFORCED.** The proof must include:

- `enforcement_module` — Name of the enforcement module (e.g., `model_tools.py:handle_function_call`)
- `dispatch_boundary` — The exact dispatch boundary where enforcement occurs
- `session_id` or `mission_id` — The session this proof applies to
- `bootstrap_state` — The CodeGraph bootstrap state at proof time
- `blocked_action_count` — Count of actions blocked by the governor
- `runtime_gate_version` — Version of the runtime gate
- `runtime_gate_fingerprint` — SHA-256 fingerprint of the gate code

**Pitfall: BEHAVIORAL_ONLY ≠ RUNTIME_ENFORCED**

A governor that is merely followed voluntarily (BEHAVIORAL_ONLY) is NOT the same as a governor that blocks at the tool-dispatch boundary (RUNTIME_ENFORCED). Hermes must never report `CONVERGENCE_GOVERNOR_ACTIVE=true` or `isRuntimeEnforced=true` unless the actual enforcement boundary provides machine-verifiable evidence that the governor is active for the current mission.

**Pitfall: Objective Satisfaction Must Block Discovery at Runtime**

After the mission objective becomes OBJECTIVE_SATISFIED or OPERATOR_AUTHORIZATION_REQUIRED, discovery tools must be rejected at the runtime boundary with `BLOCKED_OBJECTIVE_ALREADY_SATISFIED`. This is not an LLM instruction — it is a tool-dispatch gate. The denial must include:

- `mission_state` — The current objective state
- `objective` — What the objective was
- `satisfied_by` — What achieved satisfaction
- `blocked_action` — The discovery action that was blocked
- `reason` — Why it was blocked

**Pitfall: Router Ambiguity Must Be Machine-Blocking**

The repo router must produce a machine-consumable result with `status` field. Allowed statuses: `RESOLVED`, `BLOCKED_AMBIGUOUS`, `BLOCKED_UNKNOWN`. If `status != RESOLVED`, mission execution must STOP. The LLM must not repair routing ambiguity by reading mapping files manually, guessing from directory names, or choosing the "most likely" repository. The router ambiguity gate must be enforced at the model_tools.py dispatch boundary, not at the instruction level.

**Verification:** Run the runtime enforcement regression corpus (`tests/runtime-enforcement-regression-tests.js`). All 16 tests (T01-T16) must pass.

## Usage

```javascript
const { ConvergenceGovernor } = require("./convergence-governor.js");

const gov = new ConvergenceGovernor({ missionMode: "REPAIR" });

// Record a hypothesis
gov.recordHypothesis("H1", {
  question: "What executes CAMPAIGN_CREATE_PAUSED after LAHB approval?",
  evidence_for: ["executor.js found"],
  blocking_unknowns: ["transport binding"],
  confidence: 0.8,
  canonical_sources_found: ["executor.js"],
});

// Record a discovery action
const result = gov.recordAction({
  command: 'grep -r "transport" /path/to/scripts/',
  evidence: { transport: "http", endpoint: "/execute" },
  justification: "Adds transport binding scope not covered by previous search",
});

if (!result.allowed) {
  console.log(`Blocked: ${result.reason}`);
  console.log(`Convergence check:`, result.convergence_check);
}

// Transition state
gov.transitionState("IMPLEMENT");

// Generate receipt
const receipt = gov.generateReceipt();
console.log(JSON.stringify(receipt, null, 2));
```

## Metrics

Exposed in mission receipt:
- `total_tool_calls`
- `discovery_tool_calls`
- `exact_repeat_count` (target: 0)
- `semantic_repeat_count` (target: <= 1)
- `zero_information_calls` (target: <= 2)
- `post_compaction_rediscovery_count` (target: 0)
- `unauthorized_probes_block_count`
- `forced_convergence_count`
- `side_quest_block_count`
- `hard_convergence_trigger_count`