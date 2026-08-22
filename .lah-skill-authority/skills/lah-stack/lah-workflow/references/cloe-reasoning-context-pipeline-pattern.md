# CLOE Reasoning Context Pipeline Pattern

## Purpose

Document the full runtime pipeline that carries structured cognitive reasoning
from Gateway to LLM. This pattern integrates the ReasoningContextEnvelope V1
contract, envelope composition, and Brain Context Builder.

## Architecture

```
User Request
    ↓
Gateway Router (cloe_assistant intent)
    ↓
CLOE Runtime Adapter → runAssistantLoop()
    ↓
13 Cognitive Modules → structured module envelopes
    ↓
mergeModuleOutputs() → toReasoningContextFacts()
    ↓
ReasoningContextEnvelope V1 { facts, provenance, governance, uncertainty }
    ↓
composeEnvelopes() — merge multiple envelopes (dedup, conflict detection)
    ↓
reasoningContextToBrainPack() — format into LLM context block
    ↓
buildOpenClawBrainContext(cognitiveContextPack) — full prompt construction
    ↓
LLM → natural language response from structured facts
```

## Key Components

### ReasoningContextEnvelope V1
- **File:** `src/cognitive/cloe-reasoning-context-contract.js`
- **Schema:** `cloe_reasoning_context_envelope_v1`
- **24 fields** including: verified_facts, decisions, recommendations, blockers,
  constraints, uncertainty, missing_context, governance, safety
- **No answer text** — validator rejects `answer` field
- **Provenance** required on facts/recommendations/decisions/blockers
- **Adapter:** `fromCloeEnvelope()` converts existing CLOE output

### Structured Output Adaptation
- **Location:** `src/cognitive/cloe-behavior-composer.js` → `toReasoningContextFacts()`
- Extracts structured facts from module output envelopes
- Preserves provenance (source, layer, confidence, uncertainty flag)
- Sets `answer: null` — no canned text
- Produces `reasoning_context_facts` array flowing through `composeBehavior`

### Envelope Composition
- **Function:** `composeEnvelopes(envelopes, options?)`
- Merges multiple ReasoningContextEnvelopes into one
- Deduplicates facts by type+value hash
- Detects conflicts (same type, different values)
- `strict: true` fails on conflict; default includes conflicts list
- Merges blockers, decisions, recommendations, constraints, uncertainty
- Merges governance and safety (strictest wins)

### Brain Context Bridge
- **File:** `src/brain/cloe-brain-context-bridge.js`
- **Function:** `reasoningContextToBrainPack(envelope)`
- Formats envelope into 15 context sections: identity, request, capabilities,
  modules, facts, decisions, recommendations, blockers, constraints,
  governance, uncertainty, memory, workflow, next actions, LLM instructions
- Returns `cognitiveContextPack` consumed by `buildOpenClawBrainContext()`
- LLM instructions tell the model to generate from facts, not fabricate

### Pipeline Integration (Gateway)
- **Location:** `src/services/gateway/readonly-conversation-router.js`
- **Function:** `buildCloeAssistantResponse()`
- 6-step pipeline: classify → CLOE → fromCloeEnvelope → reasoningContextToBrainPack
  → buildOpenClawBrainContext → return brain context
- Surfaces: `cloe_assistant`, `reasoning_context_envelope`, `brain_context_builder`, `cognitive_context_pack`
- Returns `reasoning_context` metadata (fact count, module count, budget, etc.)

### Regression Guard (Gate 8)
- **File:** `tools/ci/cloe-reasoning-context-regression-guard.mjs`
- **10 guards** protecting: Gateway Routing, ReasoningContextEnvelope, No Canned
  Answers, Provenance, Governance, Brain Context Builder, V2 Reviewer Alignment,
  Continuous Learning, Resource Governance, Release Gate Integration
- Wired into release gate at `tools/ci/lah-cognitive-release-gate.mjs`
- Run standalone: `node tools/ci/cloe-reasoning-context-regression-guard.mjs`

## Pipeline Tests

| Test file | Tests | Coverage |
|-----------|-------|----------|
| `test/cloe-reasoning-context-contract.test.js` | 19 | contract creation, validation, fromCloeEnvelope adapter |
| `test/cloe-brain-context-bridge.test.js` | 14 | brain pack conversion, all 15 sections, metadata |
| `test/cloe-envelope-composition.test.js` | 13 | multi-envelope merge, dedup, conflicts, strict mode |
| `test/cloe-behavior-composer.test.js` | 49 | module execution, mergeModuleOutputs, toReasoningContextFacts |
| `test/cloe-reasoning-context-regression-guard.test.js` | 12 | 10 regression guards |

## Migration Path from Old Pipeline

Old flow:
```
CLOE → answer text → Gateway → template response
```

New flow:
```
CLOE → module envelopes → toReasoningContextFacts() → ReasoningContextEnvelope
  → composeEnvelopes() → reasoningContextToBrainPack()
  → buildOpenClawBrainContext(cognitiveContextPack) → LLM prompt
```

## Common Pitfalls

1. **Answer field still populated by modules** — the `fromCloeEnvelope` adapter
   explicitly discards `answer` text. If modules still produce answer text, the
   adapter will not use it — check `reasoning_context_facts` instead.

2. **Backward compatibility** — legacy `next_move`, `blockers`, `decisions`,
   `recommendation`, `executive_review` fields are still present on the envelope
   for backward compat. The new path uses `reasoning_context_facts`.

3. **Regression gate acceptance** — switching to structured output causes
   behavioral validation scores to drop because answer-length checks fail.
   Add affected domains to the accepted-regressions list in
   `tools/ci/lah-cognitive-release-gate.mjs`.

4. **Gateway test updates** — `buildCloeAssistantResponse` now returns brain
   context instead of module answer text. Gateway tests expecting specific
   answer strings must be updated.

5. **Patch tool backslash escalation**: When writing regex patterns with `\b`
   using the `patch` tool, `\\b` in the input becomes `\\\\b` (double backslash)
   in the file. Fix with Python: replace `\\\\b` with `\\b` after patching.

## Knowledge Source Injection

Project knowledge facts (roadmap, certification, QA reports, memory locks,
continuity, risks, next mission, baseline metrics) can be injected into the
pipeline **after** `fromCloeEnvelope()` → into `rcEnvelope.verified_facts`.

See `controlled-delivery-workflow` skill → `references/cloe-project-knowledge-source-pattern.md`
for the full pattern: static allowlist, path traversal rejection, bounds,
9 standard knowledge targets, and the `formatCognitiveContextPack` legacy
format passthrough fix in `selectContextSections`.

## Roadmap Context

See `references/cloe-reasoning-context-roadmap-pattern.md` for the full 8-mission
execution sequence that built this pipeline (PRs #578–#586).
