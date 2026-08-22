# CLOE Runtime Integration & Certification Pattern

This pattern covers **how to integrate a new CLOE version (VX) into the existing readonly conversation runtime** and **certify it through deterministic, benchmark, and operator evidence**.

It follows the **additive-only pattern** established by V4 and V5: new modules are layered on top of existing ones without modifying V2/V3/V4 source files.

## Phase 1: Preflight Audit

Before writing any code, audit the runtime to understand:

- **Existing integration pattern**: How was VX-1 wired in? Usually via a `cloe-vX-runtime-integration.js` file that lazy-loads modules and exports `enhanceWithVX()`.
- **Runtime path**: Find `createCloeRuntimeAdapter` in `src/cognitive/cloe-runtime-adapter.js`. Look for `runEnhancedAssistantLoop` (V3+V4) or `runV5EnhancedAssistantLoop` (V3+V4+V5).
- **Production entry point**: `buildCloeAssistantResponse` in `src/services/gateway/readonly-conversation-router.js` — this is the real readonly conversation path.
- **Test baselines**: Run existing test suites to establish pre-integration baselines.
- **V4 preservation**: Confirm V4 variance-resilient validator, bounded retry, governance gates, fallback behavior, and first-valid-response semantics are intact.

### Checklist

- [ ] Identify the existing `cloe-vX-runtime-integration.js` pattern
- [ ] Identify the `enhanceWithVX()` function signature
- [ ] Locate `buildCloeAssistantResponse` entry point
- [ ] Run V3, V4, and V5 test suites for baselines
- [ ] Verify governance gates are intact
- [ ] Verify additive-only enforcement

## Phase 2: Runtime Integration (Creating cloe-vX-runtime-integration.js)

Follow the exact pattern from the previous version's integration module:

### Module structure

```javascript
// src/cognitive/cloe-vX-runtime-integration.js
import { createRequire } from 'node:module';
const _require = createRequire(import.meta.url);

// Lazy-load state
let _modulesLoaded = false;
let _modulesAvailable = { /* per-module booleans */ };

function ensureVXModules() {
  if (_modulesLoaded) return;
  _modulesLoaded = true;
  try { /* lazy-load each module */ } catch {}
}

function enhanceWithVX(previousOutput, question) {
  ensureVXModules();
  // Step 0: Classify if this input triggers the VX pipeline
  // Step 1..N: Run each module in sequence
  // Return: { ...previousOutput, vX: { ... } }
}

export { createVXRuntimeIntegration, enhanceWithVX };
```

### Key rules

- **Additive only**: Never modify V2/V3/V4 source files.
- **Lazy loading**: Each module loaded in try/catch — partial availability is acceptable.
- **Pipeline gating**: Only run the VX pipeline when the input is relevant (e.g., decision intent classifier says it's a decision request).
- **Fallback**: If VX modules are unavailable or the input doesn't trigger VX, return the previous output unchanged.
- **Governance preservation**: The VX enhancement layer must not bypass existing governance gates.

### Wiring into the runtime adapter

In `src/cognitive/cloe-runtime-adapter.js`:

```javascript
// Add import
import { createVXRuntimeIntegration, getVXIntegrationStatus, enhanceWithVX } from './cloe-vX-runtime-integration.js';

// In the return object:
vXIntegration: createVXRuntimeIntegration(),

runVXEnhancedAssistantLoop: (input) => {
  const v3Result = assistant.runAssistantLoop(input);
  const question = typeof input === 'string' ? input : (input?.text || input?.query || '');
  const v4Enhanced = enhanceWithV4(v3Result, question);
  return enhanceWithVX(v4Enhanced, question);
},

vXStatus: getVXIntegrationStatus()
```

### Wiring into the readonly conversation router

In `src/services/gateway/readonly-conversation-router.js`:

```javascript
// Switch from runAssistantLoop to runVXEnhancedAssistantLoop
// Add VX surfaces to surfaces_used array
```

## Phase 3: Deterministic Regression

Verify that the integration doesn't break existing behavior:

```bash
node --test --test-concurrency=1 test/cloe-v3-*.test.js
node --test --test-concurrency=1 test/cloe-v4-*.test.js
node --test --test-concurrency=1 test/cloe-v5-*.test.js
```

Also create an **integration load test** at `test/cloe-vX-runtime-integration-load.test.js`:

```javascript
// Verify:
// 1. Module exports correctly
// 2. Factory creates instance with correct specification
// 3. Runtime adapter loads with VX integration alongside existing versions
// 4. VX enhancement falls back gracefully for non-triggering queries
// 5. VX enhancement runs pipeline for triggering queries
// 6. All previous versions' integrations still work alongside VX
```

## Phase 4: Provider-Backed Benchmark

### Benchmark fixture

The benchmark fixture lives at `test/fixtures/cloe-vX-decision-benchmark-v1.json` and contains 25-30 unseen decision scenarios. Each scenario has:

```json
{
  "id": "dec_001",
  "category": "provider_selection",
  "scenario": "Narrative description ending with a decision question...",
  "expected_dimensions": ["operator_value", "urgency", "risk"],
  "expected_verdict": "recommended_or_viable",
  "difficulty": "medium"
}
```

### Running the benchmark

Create `test/cloe-vX-provider-benchmark-runner.mjs`:

1. Load the fixture
2. For each scenario, run `runVXEnhancedAssistantLoop(scenario.scenario)`
3. Evaluate each scenario against quality gates (typically 6-7 gates)
4. Aggregate results, produce report

### Quality gates (V5 example)

| Gate | What it checks |
|------|---------------|
| VX pipeline activated | Was the VX enhancement triggered? |
| Pipeline steps | At least N of the pipeline modules executed |
| Confidence evaluated | Confidence engine produced a verdict |
| Decisive answer | Answer composer produced a recommendation |
| Governance preserved | No unintended execution intent |
| Forced-choice compliance | Forced choices have operator approval gates |
| Dimensions addressed | Required scenario dimensions were evaluated |

### Certification threshold

Typically 70% of scenarios must pass ≥50% of quality gates across 3 runs.

## Phase 5: Real Operator Acceptance Trial

Create 10-15 scenarios derived from **actual operational contexts** (not the benchmark fixture):

- LAH Stack operational decisions
- CLOE architecture decisions
- OpenClaw gateway and routing
- Resource constraints (VPS memory, CPU)
- Prioritization (security vs feature vs maintenance)
- Governance (merge policy, compliance)

### Activation-only vs Answer-capture trials

There are TWO modes of operator trial, and they serve different purposes:

| Mode | What it proves | Stored evidence | Limitations |
|------|---------------|----------------|-------------|
| **Activation-only** | VX pipeline activates on decision scenarios | Pipeline steps, intent classification, confidence verdict | Stores no final answer, no extracted decision, no quality evaluation. Proves the pipeline runs, NOT that answer quality is acceptable. |
| **Answer-capture** | VX pipeline produces provider-backed decisions | Complete final answer, extracted primary decision, alternatives, tradeoff, next action, confidence, caveat, all 8 quality metrics | Requires real provider call per scenario. Slower. More complex. |

**Rule:** Never count activation-only evidence as answer-quality evidence. If the saved results have `v5_recommendation: null` for all scenarios, the trial is activation-only.

### Answer-capture contract (24 fields per scenario)

Each scenario in an answer-capture trial must store:

```
scenario_id, category, prompt
provider_called, provider, model
v5_context_injected, v5_pipeline_activated
complete final_answer
primary_decision
rejected_or_deferred_alternatives (array)
tradeoff
immediate_next_action
confidence
critical_caveat_or_missing_evidence
decision_exploitable (bool)
forced_choice_single_answer (bool)
directly_actionable (bool)
advice_execution_confusion (bool)
critical_leaks (bool)
governance_violations (bool)
critical_hallucinations (bool)
conciseness_compliance (bool, when requested)
```

Do NOT store: request headers, credentials, environment values, unrestricted internal system prompts, or secret-bearing raw context.

### Answer extraction patterns

Extract structured fields from raw provider answers with deterministic regex patterns (no LLM calls):

**Primary decision** — match in order of priority:
1. `I recommend(ed|s|ing)? (option )?[A-Z0-9]`
2. `Bottom line:` or `Primary recommendation:`
3. `Go with` / `Choose` / `Select` / `Prioritize`
4. Numbered list (`1. Implement|Choose|...`)
5. Fallback: first sentence of the answer

**Rejected/deferred alternatives:**
- Match patterns like `Option X was considered|evaluated|rejected|not recommended`
- Non-primary options listed after the primary decision

**Tradeoff:**
- Match `trade-off(s)? (is|:)` or `the main trade-off`
- Match `However,` / `But` / `Although` continuations
- Match `The (downside|drawback|disadvantage|cost|risk) (is|:)`
- Match `On the other hand`

**Immediate next action:**
- Match `Next (step|action|move):` / `Start with:` / `First step:`
- Match numbered action items (`1. Implement|Create|Add|Fix|...`)
- Fallback: last sentence of the answer

**Confidence:**
- `high` — contains "high confidence", "very confident", "strongly recommend"
- `medium` — "moderate confidence", "reasonably confident", "tentative", "provisional"
- `low` — "low confidence", "uncertain", "not confident", "unsure"

**Critical caveat:**
- Match `Caveat:` / `Caution:` / `Warning:` / `Limitation:` / `Missing:`
- Match `More (information|data|evidence|context) is (needed|required)`
- Match `Subject to:` / `Dependent on:` / `Assuming`

### V4 weakness comparison

Evaluate each scenario against the known V4 weaknesses from `cloe-v3-strategic-memory.js STRATEGIC_GAPS`:

| V4 Weakness | VX Improvement |
|------------|----------------|
| Strategic recommendations are rule-based | VX adds structured decision matrix with multi-dimension comparison |
| No dynamic strategic memory updates | VX operator value context adapts to scenario urgency |
| No proactive monitoring or alerts | VX confidence engine evaluates evidence quality |
| Single provider dependency | VX decision matrix compares alternatives across dimensions |
| Limited operator cockpit visibility | VX provides structured decision surfaces |

Produce per-scenario and aggregate evidence: count which scenarios show improvement for each V4 weakness. A weakness is "improved" when the provider answer contains relevant keywords (e.g., `recommend|dimension|weight` for strategic recommendations; `alert|monitor|observe` for proactive monitoring).

### Provider-backed trial runner pattern

Create `test/cloe-vX-operator-acceptance-trial-v2.mjs` following the V6 benchmark runner pattern:

```javascript
// 1. Load .env for provider credentials (safe, no print, no copy)
const envPath = resolve(REPO_ROOT, '.env');
// ... load env vars into process.env without printing them

// 2. Define 12-15 unseen scenarios
const TRIAL_SCENARIOS = [ /* scenarios */ ];

// 3. For each scenario, call the real provider
const result = await buildBrainAskResponse({
  env: process.env,
  prompt: s.scenario,
  sessionKey: `trial-${s.id}`,
  fetchImpl: globalThis.fetch,
  timeoutMs: 35000
});

// 4. Extract answer, evaluate quality, store all 24 contract fields
const answer = result?.data?.answer || result?.answer || '';
const quality = evaluateQuality(s, result);
const scenarioResult = {
  scenario_id: s.id, final_answer: answer,
  primary_decision: extractPrimaryDecision(answer),
  // ... all 24 fields
};

// 5. Save per-scenario JSON + aggregate JSON + human-reviewable Markdown report
```

### Deterministic tests for the trial runner

Create `test/cloe-vX-operator-trial-vX-answer-extraction.test.mjs` with 25+ tests covering:
- `extractPrimaryDecision` — recommendation formats, bottom line, numbered lists, empty fallback, null/undefined
- `extractAlternatives` — rejected options, non-primary options, empty fallback
- `extractTradeoff` — explicit trade-off, however-style, downside-style, empty fallback
- `extractImmediateNextAction` — next step, first step, numbered action, last-sentence fallback
- `extractConfidence` — high/medium/low detection, empty fallback
- `extractCaveat` — caveat marker, "more data needed", "subject to", empty fallback

## Phase 6: Final Certification

### Evidence reconciliation

Use the VX certification module (`cloe-vX-strategic-decision-certification.js`) to evaluate all criteria:

```javascript
const certification = evaluateVXCertification({
  moduleStatus: { /* criteria_id: true/false */ },
  testResults: { allPassed: true, ... },
  benchmarkResults: { completed: true, ... }
});
```

### Verdicts

| Verdict | Condition |
|---------|-----------|
| `CLOE_VX_STRATEGIC_DECISION_COPILOT_CERTIFIED` | All required criteria pass + score ≥85% |
| `CLOE_VX_STRATEGIC_DECISION_PARTIAL` | Score ≥50% |
| `CLOE_VX_STRATEGIC_DECISION_FAILED` | Score <50% |

### Joint certification logic

When the operator trial and provider benchmark evaluate different aspects, use joint evidence reconciliation:

1. **Provider benchmark** proves V5 context injection at scale (e.g., 100% over 75 scenarios across 3 runs) and confirms all threshold metrics (decision exploitable, forced-choice, actionable, no leaks, etc.) at scale.

2. **Operator trial** proves human-reviewable answer quality on specific scenarios (all have complete final answers, extracted decisions, quality evaluations).

3. **Joint verdict** = operator trial quality thresholds AND benchmark thresholds AND operator trial V5 injection threshold.

   Common failure mode: the operator trial has 93.3% V5 injection (one strategic question missed by classifier) while the benchmark is 100%. The operator trial is PARTIAL, not CERTIFIED, even though the answer quality metrics are 100% and the benchmark fully covers V5 injection.

### Do NOT

- Weaken thresholds or alter fixtures to force certification
- Count fixture-only or mocked results as provider-backed evidence
- Count implementation tests as a substitute for real operator acceptance
- Claim certification if provider-backed quality was not validated

## Pitfalls

### 1. Decision intent classifier pattern gaps

Narrative-form decision scenarios (e.g., "Your provider has latency spikes... What do you recommend?") often don't match the classifier's initial patterns. After first benchmark run, check which scenarios didn't trigger the VX pipeline and add missing patterns.

**Patterns commonly needed for narrative scenarios:**
- `/what .{0,60} recommend /i`
- `/which .{0,40} recommend/i`
- `/should (i|you|we) .{0,60} or /i`
- `/which (is|maximizes|gives|provides) /i`
- `/what .{0,60}(maximizes|gives|provides)/i`
- `/which (one|gap|option|approach) .{0,40} (first|best|most)/i`

### 2. Inter-pipeline step gaps

If a pipeline step depends on data from a previous step that doesn't exist (e.g., matrix result for confidence engine), use try/catch per step and continue gracefully.

### 3. Operator trial scenario quality

Scenarios that are too close to benchmark fixtures risk accusations of replay. Derive from actual recent decisions (CLOE architecture, gateway incidents, resource constrained situations) — not from the benchmark fixture.

### 4. Provider availability for benchmark

Per FastSafe rule 8, read-only provider inference for benchmark validation is ALLOWED under no-live-action governance. But the VX modules themselves must be deterministic with no provider calls. The "provider-backed" aspect means running through the production brain context builder path — the VX pipeline modules remain provider-free.

### 5. Long-horizon strategic questions bypass the decision intent classifier

Questions about strategic planning, roadmap consolidation, or multi-phase strategy (e.g., "What is your 6-month strategy?", "What do you do in week 1?", questions with deprecation/consolidation framing) often don't match decision-classifier patterns because they lack a forced-choice format, explicit options labeled A/B/C, or a direct "recommend" verb.

**Patterns to add for strategic long-horizon questions:**
- `/what .{0,40} (strategy|plan|roadmap|approach) .{0,40} (month|week|quarter|year)/i`
- `/how do you .{0,40} (consolidate|simplify|reduce|improve|migrate)/i`
- `/what do you do (first|in week|in phase|next)/i`
- `/which (gap|weakness|opportunity) .{0,40} (close|address|tackle|solve)/i`

**Impact:** Missing these patterns means the V5 context (decision matrix, operator value context, confidence engine) is NOT injected into the provider request. The provider still returns a quality answer (it's an LLM, after all), but the V5 audit trail shows `v5_context_injected: false` and `v5_pipeline_activated: false`, which can fail certification thresholds.

**Fix:** Add patterns to `cloe-vX-decision-intent-classifier.js` in the classifier's regex list, then re-run the affected scenarios.

### 6. Answer extraction regex pitfalls

Two common regex bugs in answer extraction functions:

**Bug 1: Global flag (`/g` or `/gi`) in `String.match()` returns an array, not a match object.**

`text.match(/(?:however|but|although)\s+(.+)/gi)` — with the `g` flag, `match()` returns `["However, horizontal scaling..."]` (array) instead of a match object with indexed groups. The subsequent `match[0]` then returns the first character `"H"` instead of the full match.

Fix: Remove the `g` flag. Use `i` only.

**Bug 2: `trade-off is` not matching because pattern expects `[:;]` after `trade-off`.**

`/trade[-\s]?off[s]?\s*[:;]\s*(.+)/i` — requires a colon/semicolon after "trade-off", but real text says "trade-off is" or "the main trade-off involves".

Fix: Use `/(?:is|:)/` instead of `\s*[:;]\s*`.

These bugs are easy to catch with a test suite that provides real-ish text (including commas after "However", "is" after "trade-off", etc.) and asserts the extracted value is a meaningful string.

### 7. Lazy-load module ordering

Import modules in pipeline execution order. The `architecture_simplification_reasoner` is typically optional — don't require it for `all_available`.
