# CLOE Reasoning Context Roadmap Missions

## Purpose

Documents the 8-mission sequence that followed the LAH Cognitive Quality
Assurance Roadmap (9 phases). These missions transformed CLOE from a
template-generating assistant into a structured reasoning pipeline.

## Mission Sequence (8 missions, 10 PRs)

### Mission 0: CLOE_GATEWAY_CLOE_INTENT_ROUTING_FIX_V1 (PR #578)
**Problem**: 6/9 diagnostic prompts were routed to non-CLOE intents
(identity, self-evolution, unknown) instead of cloe_assistant.

**Fix**: Added a CLOE-specific pre-check block in `classifyReadonlyConversationIntent`
that catches CLOE-relevant prompts (identity, capabilities, roadmap, devil advocate,
memory, policy) BEFORE generic intent classification.

**Key lesson**: The order of intent checks matters — cloe_assistant was last,
so generic intents (identity, capabilities, self-evolution) caught CLOE-relevant
prompts first. The fix added a pre-check between the safety gate and generic intents.

**Result**: 9/9 prompts now reach cloe_assistant (up from 3/9).

### Mission 1: CLOE_REASONING_CONTEXT_CONTRACT_V1 (PR #579)
**Deliverable**: `src/cognitive/cloe-reasoning-context-contract.js`
- 24-field ReasoningContextEnvelope V1
- Validator rejects `answer` field (no canned text)
- `fromCloeEnvelope()` adapter for existing CLOE output
- Safety markers: `_no_canned_answer`, `_no_template_fragments`, `_no_provider_text_as_truth`
- 19 tests

### Mission 2: CLOE_BRAIN_CONTEXT_BUILDER_REASONING_CONTEXT_V1 (PR #580)
**Deliverable**: `src/brain/cloe-brain-context-bridge.js`
- `reasoningContextToBrainPack()` — formats ReasoningContextEnvelope into 15
  LLM-ready context sections with provenance, governance, uncertainty
- Returns `cognitiveContextPack` consumed by `buildOpenClawBrainContext()`
- No answer text, no templates
- 14 tests

### Mission 3: CLOE_COGNITIVE_MODULES_STRUCTURED_OUTPUT_V1 (PR #581)
**Deliverable**: `toReasoningContextFacts()` adapter in composer
- Extracts structured facts from module output envelopes (with provenance)
- Replaces `mergeModuleOutputs` template generation (50+ lines of canned answers removed)
- Sets `answer: null` — LLM generates from facts
- Provenance flows through: `mergeModuleOutputs` → `composeBehavior` → `runAssistantLoop` → `fromCloeEnvelope`
- 170/170 tests pass

**Pitfall**: answer:null causes behavioral validation score drops because
V1 checks answer length. All behavioral domains added to accepted regressions.

### Mission 4: CLOE_BEHAVIOR_COMPOSER_ENVELOPE_COMPOSITION_V1 (PR #582)
**Deliverable**: `composeEnvelopes()` in composer
- Merges multiple ReasoningContextEnvelopes
- Dedup by fact type+value hash
- Conflict detection (same type, different values)
- Strict mode fails on conflict; normal mode includes conflicts list
- 13 tests

### Mission 5: CLOE_RUNTIME_REASONING_CONTEXT_PIPELINE_V1 (PR #583)
**Deliverable**: Updated `buildCloeAssistantResponse()` in gateway router
- Full 6-step pipeline: classify → CLOE → fromCloeEnvelope → reasoningContextToBrainPack → buildOpenClawBrainContext → return
- No template answers, no module prose
- Structured reasoning_context metadata in response
- 183/183 tests pass

### Mission 6: CLOE_REAL_RUNTIME_BEHAVIOR_VALIDATION_V2 (PR #584)
**Deliverable**: Behavioral validation V2 report
- Global Score: 69/100
- Release Gate: 7/7 PASS
- CLOE Routing: 50%
- Gateway failure no longer primary root cause
- Evidence + Comparison need reviewer calibration

### Mission 7: CLOE_BEHAVIOR_VALIDATION_V2_CALIBRATION_V1 (PR #585)
**Deliverable**: Mapping + comparison reporter
- `test/fixtures/lah-domain-capability-mapping-v1.json` — 12 V1 domains → 17 V2 capabilities
- `tools/ci/lah-behavior-comparison-reporter.mjs` — normalized comparison
- Evidence reviewer: 100% completeness
- Behaviour reviewer: avg 62
- Continuous learning: 31 artifacts
- Comparison: V1 avg 57 → V2 avg 54 (delta -3, within expected)

### Mission 8: CLOE_REASONING_CONTEXT_REGRESSION_GUARD_V1 (PR #586)
**Deliverable**: 10-guard regression guard
- Gateway Routing, ReasoningContextEnvelope, No Canned Answers, Provenance,
  Governance, Brain Context Builder, V2 Reviewer Alignment, Continuous Learning,
  Resource Governance, Release Gate Integration
- Wired into release gate as Gate 8 (8/8 gates PASS)
- `tools/ci/cloe-reasoning-context-regression-guard.mjs`
- 12 tests, all guards PASS on current pipeline

### Mission 9: CLOE_COGNITIVE_ASSISTANT_STABILIZATION_V1 (PR #587)
**Deliverable**: Official V1 certification
- 195/195 tests, 8/8 release gates, 10/10 regression guards, 20/20 certification checks
- Architecture frozen as CLOE_COGNITIVE_ASSISTANT_V1
- Certification report, baseline metrics, memory lock chain (21 missions)

### Mission 10: CLOE_COGNITIVE_ASSISTANT_V1_ARCHITECTURAL_REVIEW (PR #588)
**Deliverable**: Canonical architectural review document
- 8 canonical architecture decisions (AD-001 through AD-008)
- 10 architectural invariants
- 21 missions across PRs #567–#589
- Complete pipeline data flow diagram

### Mission 11: CLOE_USER_FACING_ANSWER_RENDERER_V1 (PR #589)
**Deliverable**: `src/services/cloe/operator-answer-renderer.js`
- Transforms Brain Context output into natural operator responses
- NORMAL mode: strips system prompts, safety envelopes, context packs, metadata
- DEBUG mode: natural answer + reasoning metadata (requires explicit flag)
- 9-question smoke test: 9/9 natural, 0 internal artifacts exposed
- Wired into Gateway via `buildCloeAssistantResponse()`
- 197/197 tests, 10/10 regression guards, 8/8 release gates PASS

See `references/cloe-user-facing-answer-renderer-pattern.md` for full pattern details.

## Key Architecture Decisions

### Pipeline
```
User → Gateway → CLOE Runtime → Modules → ReasoningContextEnvelope
  → composeEnvelopes() → reasoningContextToBrainPack()
  → buildOpenClawBrainContext(cognitiveContextPack) → LLM prompt
```

### No Canned Answers
The composer sets `answer: null` when structured facts exist. Error paths
(fail-closed) still produce user-facing messages since there's no structured
output to generate from.

### Provenance
Every fact/recommendation/blocker/decision carries `{ source, layer, confidence,
uncertain, timestamp }`. This survives composition and brain pack conversion.

## Common Pitfalls

1. **Patch tool backslash escalation**: When writing regex patterns like `\b`
   (word boundary) using the `patch` tool, `\\b` in the input becomes `\\\\b`
   (double backslash = literal `\b`, NOT word boundary) in the file. Fix:
   run `python3 -c "c=open('f').read(); c=c.replace('\\\\b', '\\b'); open('f','w').write(c)"`.

2. **ES module shorthand gotcha**: `{ total_turns }` in object shorthand requires
   a variable named `total_turns` in scope. If the variable is `totalTurns`
   (camelCase), write `{ total_turns: totalTurns }`.

3. **Behavioral validation V1 vs V2 scoring**: Review engine scores and behavioral
   validation runner scores use DIFFERENT measurement systems. Never compare them
   directly. Always use the behavioral validation runner for regression checking.

4. **Known regression acceptance**: Switching to structured output (answer: null)
   causes V1 behavioral validation scores to drop because answer-length checks fail.
   Add affected domains to `acceptedRegressions` in the release gate.

5. **Resource governance first**: The release gate's resource check must run BEFORE
   heavy validation. On 8GB VPS, stop with RESOURCE_PRESSURE if < 1 GiB available.
