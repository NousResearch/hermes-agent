# CLOE Batch B + Full Assistant Loop + Design Gate Patterns

Extension to `openclaw-runtime-mission-pattern.md` covering patterns that emerged during missions 7-9.

---

## 1. Batch B Pattern: 4 slices (beyond the 3-slice limit)

Unlike Batch A (3 slices), Batch B has **4 slices** that exceed `max_concurrent_children=3`. Execution strategy:

### Parallel execution order

```
Lot 1 (3 sub-agents):
  Slice 10 — Friction Detection
  Slice 11 — Knowledge Gap Intelligence  
  Slice 12 — Recommendation Layer

Lot 2 (1 sub-agent, after Lot 1 completes):
  Slice 13 — Executive Reasoning (depends on knowledge gap + friction detection interfaces)
```

### Slice 10 — Friction Detection (`cloe-friction-detection.js`)

11 friction signals: repeated_manual_step, repeated_clarification, repeated_governance_exception, repeated_failed_tool, repeated_ci_billing_blocker, repeated_memory_context_loss, repeated_direct_main_bypass, repeated_prompt_handoff, repeated_branch_pr_reconciliation, repeated_missing_policy, repeated_local_remote_drift.

Each signal has: friction_id, category, signal, evidence, frequency, severity, scope, detected_at, source_refs, related_workflows, related_decisions, recommended_action, confidence, governance_required.

Methods: `detectFriction(events)`, `rankFriction()`, `proposeImprovement(frictionId)`, `distinguishPattern(frictionId)`, `getFrictionById(id)`, `listFriction(filter)`, `getStatus()`.

Fail-closed when evidence is insufficient (frequency < threshold). Never mutates memory automatically.

### Slice 11 — Knowledge Gap Intelligence (`cloe-knowledge-gap-intelligence.js`)

11 gap categories: missing_policy, missing_workflow, missing_tool, missing_memory_wiring, missing_context, stale_assumption, conflicting_policy, undocumented_decision, unverified_capability, unowned_blocker, missing_test_coverage.

Gap fields: gap_id, category, description, evidence, impact, urgency, affected_components, source_refs, proposed_resolution, confidence, requires_operator_decision, fail_closed_recommendation.

Methods: `detectGaps(snapshot)`, `detectContradictions()`, `detectMem0ClaimMismatch()`, `detectToolMismatch()`, `proposeResolution(gapId)`, `listGaps(filter)`, `getGapById(id)`, `getStatus()`.

**Pitfall — `***` placeholders:** Sub-agents may leave `***` (three asterisks) as placeholder field names. These cannot be fixed via `patch` or normal string replacement because both old and new strings are identical byte sequences (`***`). **Fix: Python sentinel approach**:
```python
content = content.replace('***', 'UNIQUE_SENTINEL_XYZ')
content = content.replace('UNIQUE_SENTINEL_XYZ', '***')
```
After replacement, always re-run the test suite to verify no corruption.

### Slice 12 — Recommendation Layer (`cloe-recommendation-layer.js`)

Input: candidate options, mission state, blockers, risks, costs, expected impact, governance constraints, dependencies, confidence.

Output: recommended_option, ranked_options, rationale, tradeoffs, rejected_options, confidence, required_approvals, stop_conditions, suggested_handoff.

Scoring: rank by impact/risk/cost/dependencies. Prefers stabilization when blocker exists. Prefers local CI while GitHub Actions unavailable. Prefers docs/spec when architecture uncertain. "I do not recommend" when evidence insufficient.

Methods: `evaluate(options, context)`, `getRecommendation()`, `explainTradeoffs(optionId)`, `rejectUnsafeOption(option)`, `getStatus()`.

### Slice 13 — Executive Reasoning (`cloe-executive-reasoning.js`)

10 reasoning modes: proceed, pause_and_stabilize, ask_operator, split_scope, defer, audit_first, policy_first, memory_first, rollback_first, fail_closed.

Methods: `evaluateSituation(context)`, `challengePrematureWork(proposal)`, `detectGovernanceDebt(snapshot)`, `detectArchitectureNotReady(snapshot)`, `recommendSplitMission(proposal)`, `recommendAuditFirst(snapshot)`, `requestOperatorApproval(action)`, `authorizeAction(action)`, `getStatus()`.

Accepts optional DI: frictionDetection, knowledgeGapIntelligence, recommendationLayer, nextMoveEngine, workflowMemory, decisionMemory, operationalMemory.

**Key behavior:** `authorizeAction` must never authorize forbidden actions (provider_write, ungoverned, bypass_governance, etc.). `detectGovernanceDebt` looks for repeated exception markers in snapshots.

---

## 2. Full Assistant Loop Pattern

The final integration layer (CLOE_FULL_ASSISTANT_LOOP_V1) that orchestrates ALL 12 cognitive modules into a single governed pipeline.

### Module: `cloe-full-assistant-loop.js`

Exports `createCloeAssistantLoop(options)` returning:
- `runAssistantLoop(input)` — main entry point
- `buildAssistantContext(input)` — structured context from string or `{text, query}` objects
- `classifyAssistantRequest(input)` — deterministic keyword matching (NO LLM)
- `selectAssistantResponseMode(input, classification)` — maps to 10 behavioral modes
- `composeAssistantResponse(input, classification, context)` — orchestrates modules
- `explainAssistantLoopState()` — returns loop state

### 14 request types (deterministic classification)

| Query | Detected by keywords |
|-------|---------------------|
| NEXT_MOVE_QUERY | next move, prochain, next step, what's next |
| STATUS_QUERY | status, where are we, state, resume, where we at |
| BLOCKER_QUERY | blocker, blocked, stuck, bloque, bloqué |
| WORKFLOW_QUERY | workflow, procédure, which workflow |
| TOOL_QUERY | tool, which tool, what can I use |
| SELF_KNOWLEDGE_QUERY | who are you, what can you do, capabilities |
| MEMORY_QUERY | memory, what do you remember, layers |
| DECISION_QUERY | decision, why was it decided |
| FRICTION_QUERY | friction, repeated, pattern |
| KNOWLEDGE_GAP_QUERY | gap, missing, unknown |
| RECOMMENDATION_QUERY | recommend, what should I |
| HANDOFF_QUERY | handoff, prompt, prepare, generate |
| EXECUTIVE_REVIEW_QUERY | review, evaluate, is it safe |
| GENERAL_QUERY | (fallback) |

### 10 response modes (from behavior model)

answer, diagnose, recommend, challenge, handoff, summarize, resume, plan, ask-for-approval, stop/fail-closed.

### Loop output envelope (21 fields)

```javascript
{
  ok, request_type, response_mode, answer, recommendation,
  next_move, blockers, active_modes, active_workflows,
  decisions, handoff, friction, knowledge_gaps, executive_review,
  safety, provenance, limitations,
  mem0_status: { installed: false, enabled: false, wired: false },
  writes_performed: false,
  tools_executed: false
}
```

### Module delegation map

| Request type | Delegates to |
|-------------|-------------|
| NEXT_MOVE_QUERY | `NextMoveEngine.evaluateNextMove()` |
| STATUS_QUERY | `SessionContinuity.getActiveMissionSummary()` |
| BLOCKER_QUERY | `SessionContinuity.getCurrentBlocker()` |
| WORKFLOW_QUERY | `WorkflowMemory.retrieveBestWorkflow()` |
| TOOL_QUERY | `ToolAwareness.recommendToolForTask()` |
| SELF_KNOWLEDGE_QUERY | `SelfKnowledge.ask()` |
| MEMORY_QUERY | `SelfKnowledge.ask('memory')` |
| DECISION_QUERY | `DecisionMemory.searchDecisions()` |
| FRICTION_QUERY | `FrictionDetection.listFriction()` |
| KNOWLEDGE_GAP_QUERY | `KnowledgeGapIntelligence.listGaps()` |
| RECOMMENDATION_QUERY | `RecommendationLayer.evaluate()` |
| HANDOFF_QUERY | `GovernedActionHandoff.generateHandoff()` |
| EXECUTIVE_REVIEW_QUERY | `ExecutiveReasoning.evaluateSituation()` |
| GENERAL_QUERY | `SelfKnowledge.ask()` |

### Testing pattern

73 tests covering: all 14 request types, mem0 deferred, writes_performed=false, tools_executed=false, unknown state not invented, unsafe action returns fail-closed, loop delegates to modules. Run all existing cognitive suites afterward for regression.

---

## 3. Design Gate Pattern (docs-only)

A design gate (e.g. CLOE_MEM0_ADAPTER_DESIGN_GATE_V1) is a **docs-only** mission that produces an analysis document answering specific yes/no questions about whether to proceed with implementation.

### Structure

1. Current status audit
2. Decision matrix (3-4 options with risk/value/effort assessments)
3. Recommendation with rationale
4. Allowed/forbidden categories
5. Source-of-truth boundaries
6. Adapter safety model
7. Data flow diagrams
8. Failure modes
9. Stop conditions (10+ conditions that block implementation)
10. Future implementation phases
11. Security/cost/migration risk analysis

### Validation

- git diff --check: PASS
- JSON.parse receipt: PASS
- No dependency changes: confirmed
- No runtime code: confirmed (docs only)

### Deliverable

Single architecture doc at `docs/architecture/CLOE_<TOPIC>_DESIGN_GATE_V1.md` plus operator packet, receipt, and continuity JSON. No source code changes. No test files.
