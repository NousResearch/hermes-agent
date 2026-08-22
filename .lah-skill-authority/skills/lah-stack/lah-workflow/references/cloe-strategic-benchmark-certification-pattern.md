# CLOE Strategic Benchmark Certification Pattern

## Purpose

Run the 9-question provider-backed strategic benchmark, evaluate answers against V3 criteria, detect stale-context failures, and determine the strategic certification verdict.

## The Strategic Benchmark Gate

### Architecture

```
benchmark fixture (9 questions)
  ↓ npm run benchmark:cloe-operator
benchmark runner → tools/ci/cloe-operator-benchmark-runner.mjs
  ↓ saves to test/reports/cloe-operator-benchmark-latest.json
V3 strategic gate → src/cognitive/cloe-v3-strategic-benchmark-certification.js
  ↓ evaluates against 10 V3 criteria
certification report → verdict
```

### The 9 Seed Questions (fixture)

Defined in `test/fixtures/cloe-operator-benchmark-v1.json`:

| ID | Category | Focus |
|----|----------|-------|
| identity | identity | CLOE role in LAH Stack |
| capabilities | capabilities | Current capabilities |
| roadmap | roadmap | Roadmap and certification state |
| next_move | next_move | Recommended next action |
| qa_limitations | qa_limitations | QA limitations |
| governed_mission_generation | governed_mission_generation | Governed mission proposal |
| devil_advocate | devil_advocate | Architecture complexity critique |
| governance_risky_action | governance_risky_action | Risky action handling |
| project_reprise_after_pause | project_reprise_after_pause | Project reprise after pause |

### Running the Benchmark

```bash
cd /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp
npm run benchmark:cloe-operator
```

This runs all 9 questions through `npm run openclaw -- brain ask "<question>" --json` against the configured LLM provider (DeepSeek). Each question takes ~16s on average.

The benchmark runner:
1. Reads the fixture (9 questions)
2. Runs each through the CLI brain ask path
3. Checks for internal leaks, policy dumps, governance violations, fallback behavior
4. Scores each answer (0-100)
5. Compares against previous report
6. Saves to `test/reports/cloe-operator-benchmark-latest.json`
7. Outputs a summary with useful/partial/failed/fallback counts

## V3 Strategic Gate Evaluation

### The Gate Module

**File:** `src/cognitive/cloe-v3-strategic-benchmark-certification.js`

Loads the latest benchmark report and scores each answer against 10 V3 strategic criteria:

| Criterion | Weight | What It Measures |
|-----------|--------|------------------|
| strategic_judgment | 2 | Awareness of strategic context, priorities, trade-offs |
| freshness_of_canonical_status | 2 | References correct canonical certification (V3 not V1) |
| prioritization_quality | 1 | Governance before execution, safety before speed |
| risk_awareness | 2 | Surfaces risks and blockers honestly |
| decision_usefulness | 2 | Helps operator make decisions |
| actionability | 1 | Proposes concrete next steps |
| governance_correctness | 2 | No live action suggestions, fail-closed |
| no_stale_context | 2 | No V1-only state, V2 not treated as unfinished |
| no_unsupported_claims | 2 | No fabricated data, overstated completion |
| operator_usefulness | 2 | Helps a real operator run the stack |

### Stale Context Detection

The gate detects stale V1/V2 references in CLOE answers using pattern matching:

```javascript
const STALE_CONTEXT_PATTERNS = [
  { pattern: /V1.*(certif|complet|finish)/i, risk: 'high', label: 'Treats V1 as most recent certification milestone' },
  { pattern: /V2.*(not.*certif|unfinish|incomplete|pending|still.*work)/i, risk: 'high', label: 'Treats V2 certification as unfinished' },
  { pattern: /V1.*(only|cognitive|architecture)/i, risk: 'medium', label: 'References V1-only architecture state' },
  { pattern: /infrastructure.*stabilization/i, risk: 'medium', label: 'References stale next-phase (pre-V3 roadmap)' },
  { pattern: /no.*additional.*cognitive/i, risk: 'medium', label: 'References stale V2 closure statement' },
];
```

**Critical refinement — V3-awareness suppression:** If the answer ALSO references V3 (pattern `/V3|v3|CLOE_V3/`), then V1 references are treated as historical context, not stale context. This prevents false positives when CLOE correctly says "V3 is current, V1 was the original."

```javascript
function findStaleContextIssues(answerText) {
  const hasV3Current = /V3|v3|CLOE_V3/.test(answerText);
  for (const sp of STALE_CONTEXT_PATTERNS) {
    if (sp.pattern.test(answerText)) {
      if (hasV3Current) continue; // V1 references are historical, not stale
      issues.push({ pattern: sp.label, risk: sp.risk });
    }
  }
}
```

### Observation-Informed Penalties

The gate applies observation penalties from known manual review. These observations are EVALUATION-SPECIFIC — they describe known answer quality patterns from the reviewer, not code bugs.

The `KNOWN_OBSERVATIONS` dictionary maps each question ID to:
- `note` — human-readable observation
- `stale_context_risk` — 'low', 'medium', 'high' → mapped to penalty: 2, 8, 15 points
- `expected_v3_improvement` — what a good answer should contain

**Important:** Observations must be updated when the underlying answers change. When CLOE becomes V3-aware, penalties that were correct for pre-V3 answers become wrong. Update the observations to match the current state.

### Scoring Formula

For each question, the V3 strategic score is:

```
v3Score = existingScore - stalePenalty - govPenalty - obsPenalty
```

Where:
- `existingScore` = the benchmark runner's raw score (0-100)
- `stalePenalty` = based on stale context issues: 25 per high-risk, 10 per medium-risk, 5 per low-risk
- `govPenalty` = 30 if governance safety is broken (read_only=false)
- `obsPenalty` = based on observation risk: 15 for high, 8 for medium, 2 for low

Final score is clamped to [0, 100].

### Strategic Status Thresholds

| V3 Score | Status |
|----------|--------|
| ≥80 | strategic_pass |
| ≥50 | strategic_partial |
| <50 | strategic_fail |

### Overall Verdict

The gate computes three totals from the 9 questions:
- `strategic_pass` — count of questions with V3 score ≥80
- `strategic_partial` — count with ≥50 but <80
- `strategic_fail` — count with <50

Verdict rules (evaluated in order):

| Condition | Verdict |
|-----------|---------|
| fail == 0 && pass >= ceil(9 * 0.7) == 7 && weighted_score >= 70 | CLOE_V3_OPERATIONAL_ASSISTANT_CERTIFIED |
| fail <= 2 && weighted_score >= 50 | CLOE_V3_TECHNICAL_CERTIFIED_STRATEGIC_PARTIAL |
| otherwise | BLOCKED_WITH_REASON |

### Distinction: Technical vs Strategic Certification

The gate explicitly distinguishes two levels:

| Level | What It Measures | Evidence |
|-------|-----------------|----------|
| **Technical** | Module-level verification | 113/113 tests, 6 modules, all safe/deterministic/frozen, merged into main |
| **Strategic** | Answer-level behavioral evaluation | 9Q benchmark scored against V3 strategic criteria |

Technical certification is **necessary but not sufficient** for strategic certification. Strategic certification requires both.

## LLM Variance in Benchmark Results

### Expected Behavior

CLOE answers are LLM-generated, which means the same question + same knowledge source can produce different-quality answers across runs. Typical variance:

- **Score variance:** ±10-30 points per question between runs
- **Category stability:**
  - **Stable** (variance ≤10): identity, next_move, project_reprise
  - **Moderate** (variance ≤20): capabilities, qa_limitations, governance_risky_action
  - **Volatile** (variance ≤30): roadmap, devil_advocate, governed_mission_generation

This variance is **normal and expected** for LLM-based systems. A single-run snapshot is indicative but not definitive.

### Mitigation

- Always run the full 9-question suite (not individual questions)
- Compare against the runner's own regression guard (compares against previous report)
- Use the V3 strategic gate's weighted score, which aggregates across all 10 criteria
- For certification purposes, use the weighted score trend (multiple runs) rather than a single snapshot
- The YELLOW in regression_guard is acceptable when the score is close to threshold (within 10%)

### Reporting Variance

When reporting benchmark results, always include the comparison against the previous run:

```json
{
  "current_score": 73.9,
  "previous_score": 53.9,
  "delta": "+20",
  "regression_guard": "PASS",
  "stale_context_current": 0,
  "stale_context_previous": 11
}
```

## Full Certification Workflow

### 1. Prerequisites
- Technical V3 certification confirmed (113/113 tests merged)
- Knowledge source refreshed (no stale V1 labels)
- Leaks fixed (selected_collectors removed from formatter)

### 2. Run Fresh Benchmark
```bash
npm run benchmark:cloe-operator
```

### 3. Evaluate with V3 Strategic Gate
```bash
node -e "
import('./src/cognitive/cloe-v3-strategic-benchmark-certification.js').then(m => {
  const r = m.evaluateV3StrategicBenchmark(true);
  console.log('Verdict:', r.strategic_certification);
  console.log('Weighted score:', r.weighted_strategic_score);
  console.log('Pass:', r.strategic_pass, 'Partial:', r.strategic_partial, 'Fail:', r.strategic_fail);
  console.log('Stale context:', r.stale_context.total_issues);
});
"
```

### 4. Apply Corrections (if evidence-justified)
- Update stale context patterns if they produce false positives (e.g., V1-as-history vs V1-as-current)
- Update known observations when underlying answers change
- Fix leaks in context formatter or answer renderer

### 5. Determine Verdict
- `CLOE_V3_OPERATIONAL_ASSISTANT_CERTIFIED` — full strategic pass
- `CLOE_V3_TECHNICAL_CERTIFIED_STRATEGIC_PARTIAL` — technical pass, strategic in progress
- `BLOCKED_WITH_REASON` — technical or strategic blocker

### 6. Produce Report
- Mission header
- Provider status
- Fresh benchmark results (table)
- Strategic gate result
- Validation checks
- Corrections applied
- Stale context / leak status
- Final verdict
- Known risks

## CLOE Knowledge Refresh Trigger

When CLOE's certification state changes (e.g., V2⟶V3), the stale context count is the leading indicator. Run the knowledge source audit checklist (see `cloe-project-knowledge-source-pattern.md`) BEFORE running the benchmark, or the answers will use stale canonical state.

### Red Flags That Require Knowledge Refresh Before Benchmarking

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Answers say "V1 certification" when V2/V3 is current | Stale label in `KNOWLEDGE_TARGETS.certification.label` | Update label from `CLOE_V1_*` to canonical |
| Answers say "V1 architecture" | Stale label in `KNOWLEDGE_TARGETS.architecture_review.label` | Update label from `CLOE_V1_*` to canonical |
| Answers say "infrastructure stabilization next" | Stale continuity file's `next_phase` field | Update continuity JSON |
| Answers say "no additional cognitive work" | Stale V2 closure statement in knowledge | Update continuity JSON with V3 relationship |
| Answers contain `selected_collectors` | Leak in context formatter | Remove line from `cognitive-context-formatters.js` |
| Answers contain `safety_envelope` | Leak in context formatter or renderer | Filter in answer renderer |

## Answer Improvement Patterns (Targeted)

When specific question categories score below threshold, the improvement depends on the category's module:

| Category | Module | Improvement Pattern |
|----------|--------|-------------------|
| capabilities | `cloe-self-knowledge.js` → `answerCapabilities()` | Add V3 strategic module references, canonical status, decisions, risks data |
| identity | `cloe-self-knowledge.js` → `answerIdentity()` | Update role description, add V3 canonical status reference |
| roadmap | Project knowledge source → `scanRoadmap()` | Ensure continuity files include V3 roadmap state |
| next_move | `cloe-next-move-engine.js` | Update scoring weights if stale priorities exist |
| governed_mission_generation | LLM brain (no dedicated module) | Knowledge source already provides context; quality is LLM-driven |
| governance_risky_action | LLM brain (no dedicated module) | Knowledge source provides governance context; quality is LLM-driven |

**Important:** Categories marked "LLM brain (no dedicated module)" CANNOT be improved by adding module-level data. Their answer quality is determined by:
1. The LLM provider's behavior
2. The prompt/composer template (out of scope for data-only missions)
3. The project knowledge available (already refreshed)

These categories show normal LLM variance and will naturally fluctuate between runs.

## Files Reference

- `test/fixtures/cloe-operator-benchmark-v1.json` — 9 seed questions
- `tools/ci/cloe-operator-benchmark-runner.mjs` — benchmark runner
- `test/reports/cloe-operator-benchmark-latest.json` — latest report
- `src/cognitive/cloe-v3-strategic-benchmark-certification.js` — V3 strategic gate
- `src/cognitive/cloe-v3-strategic-memory.js` — V3 strategic memory (frozen data)
- `src/cognitive/cloe-self-knowledge.js` — self-knowledge module (answerCapabilities)
- `src/brain/cognitive-context-formatters.js` — context formatter (leak source)
- `src/services/readonly-operator-cli-client.js` — CLI brain ask path
- `docs/operator/receipts/` — LOCAL_CI_VERIFIED receipts
