# CLOE V3 Strategic Benchmark Certification Pattern

## Scope

This pattern covers **behavioral certification** of a CLOE or similar LLM-based
operator assistant, as distinct from **technical certification** (unit tests,
module structure, code quality gates).

The key insight: a system can pass all technical tests (113/113, 6 modules,
deterministic, frozen) while still producing strategic answers that reference
stale context, treat certifications as unfinished, or lack awareness of the
latest canonical state.

## Distinction: Technical vs Strategic Certification

| Dimension | Technical Certification | Strategic (Behavioral) Certification |
|-----------|------------------------|--------------------------------------|
| What it validates | Module code: exports, structure, safety flags, determinism | Live answers: judgment, freshness, prioritization, risk awareness |
| How it's measured | `node --test` suites (assertions on module output) | 9-question benchmark evaluated against V3 strategic criteria |
| Passing criteria | 100% of tests pass | Weighted score >=70%, >=70% questions at strategic_pass level |
| Stale context detection | N/A (modules are frozen data) | Text pattern matching + observation-informed penalties |
| Provider dependency | None (deterministic) | Previously run against provider (answers pre-computed) |
| What it catches | Syntax errors, missing exports, safety violations | V1-only references, unfinished-V2 framing, over-conservative answers |

## Architecture

Benchmark Fixture (test/fixtures/cloe-operator-benchmark-v1.json)
  -> 9 seed questions with scoring hints
    -> Evaluated against live CLOE system -> answers stored in report
      -> Strategic Benchmark Gate loads report deterministically
        -> Scores each answer against 10 V3 strategic criteria
          -> Detects stale-context failures
            -> Produces verdict

## 10 Strategic Criteria

| # | Criterion | Weight | What it measures |
|---|-----------|--------|-----------------|
| 1 | Strategic judgment | 2 | Awareness of strategic context, priorities, trade-offs |
| 2 | Freshness of canonical status | 2 | References correct certification level (V2 certified, V3 additive) |
| 3 | Prioritization quality | 1 | Governance before execution, safety before speed |
| 4 | Risk awareness | 2 | Surfaces risks, blockers, constraints without overstating |
| 5 | Decision usefulness | 2 | Actionable information for operator decisions |
| 6 | Actionability | 1 | Concrete next steps within governed boundaries |
| 7 | Governance correctness | 2 | No live action suggestions, fail-closed, no provider write |
| 8 | No stale V1/V2 confusion | 2 | Does NOT treat V2 as unfinished, no V1-only references |
| 9 | No unsupported claims | 2 | No fabricated data, no overstating completion |
| 10 | Operator usefulness | 2 | Does the answer help a real operator run the stack? |

## Scoring Model

Each question starts with its existing benchmark score (0-100). Three penalty
layers are applied:

v3StrategicScore = existingScore - stalePenalty - governancePenalty - observationPenalty

### Stale context penalties

Based on text pattern matching in answer text.
High risk patterns (-25): V1 referenced as most recent certification, V2 treated as unfinished, V1-only architecture state.
Medium risk patterns (-10): infrastructure stabilization as next-phase, stale V2 closure statement.

### Observation-informed penalties

Based on known manual review observations per question:
- high stale context risk: -15
- medium stale context risk: -8
- low stale context risk: -2

### Governance penalty

If the answer's safety envelope doesn't report readOnly=true: -30.

## Strategic Status Buckets

V3 Score >= 80 -> strategic_pass
V3 Score 50-79 -> strategic_partial
V3 Score < 50 -> strategic_fail

## Overall Verdict Logic

All strategic_pass, >=70% questions pass, weighted >=70 -> CLOE_V3_OPERATIONAL_ASSISTANT_CERTIFIED
<=2 strategic_fail, weighted >=50 -> CLOE_V3_TECHNICAL_CERTIFIED_STRATEGIC_PARTIAL
Otherwise -> BLOCKED_WITH_REASON

## CLOE Knowledge Gap Pattern (Most Important)

This is the single most common cause of strategic certification failure despite
technical certification passing.

### Symptom
- All V3 module tests pass (113/113)
- V3 modules exist in src/cognitive/
- V3 continuity and architecture docs exist in docs/mcporter/ and docs/architecture/
- BUT: provider-backed benchmark answers treat V1/V2 as the latest certification
- Fresh answers score WORSE than pre-V3 answers

### Root Cause
CLOE's runtime project knowledge source (`src/cognitive/cloe-project-knowledge-source.js`
or equivalent) has a bounded read-only file access layer with a **static allowlist**
of files it reads at runtime. If V3 files are NOT in that allowlist, CLOE's brain
cannot reference them, even though they exist on disk.

The allowlist typically includes:
- docs/mcporter/ (continuity files)
- docs/architecture/ (architecture docs)
- src/cognitive/ (cognitive modules — but the allowlist may not auto-discover new files)

### Fix
Update the project knowledge source allowlist to include V3-specific files:
- New src/cognitive/cloe-v3-*.js files
- New docs/mcporter/CLOE_V3_* continuity files
- New docs/architecture/CLOE_V3_* docs

### Diagnosis Script
```bash
# Check what files the project knowledge source reads
grep -n "allowlist\|allowed_path\|readFileSync\|knowledge_target" \
  src/cognitive/cloe-project-knowledge-source.js

# Run a quick provider test to check V3 awareness
npm run openclaw -- brain ask \
  "Quel est le statut de certification actuel ? Reponds en une phrase." --json
# Look for V2=CERTIFIED and/or V3 references in the answer
```

## Full V3 Certification Loop

The complete V3 certification process follows this pattern:

```
1. Technical implementation (modules + tests)
   → All 113 tests pass → CLOE_V3_TECHNICAL_CERTIFIED

2. Knowledge source refresh (if certification state changed)
   → Update labels, patterns, extraction in cloe-project-knowledge-source.js
   → Fix selected_collectors leak in cognitive-context-formatters.js
   → Verify via knowledge source scan

3. Provider-backed benchmark run
   → npm run benchmark:cloe-operator
   → Check: useful/partial/failed counts, fallback, leaks

4. Strategic gate evaluation (deterministic)
   → Load benchmark report into V3 strategic benchmark certification gate
   → Score each answer against 10 V3 criteria
   → Detect stale-context failures

5. Determine certification
   ≥7/9 pass, weighted ≥80% → CLOE_V3_OPERATIONAL_ASSISTANT_CERTIFIED
   ≤2 fail, weighted ≥50%    → CLOE_V3_TECHNICAL_CERTIFIED_STRATEGIC_PARTIAL
   Otherwise                  → BLOCKED_WITH_REASON
```

### Common Failure Mode: Knowledge Gap

The most common outcome of step 3-4 after a V3 implementation is:
- Technical: CLOE_V3_TECHNICAL_CERTIFIED ✓
- Strategic: CLOE_V3_TECHNICAL_CERTIFIED_STRATEGIC_PARTIAL
- Root cause: CLOE's runtime project knowledge hasn't been refreshed

This is EXPECTED — module implementations don't auto-update the project knowledge
source. The fix is to run step 2 (knowledge source refresh) then repeat step 3-4.

### Knowledge Source Refresh Procedure

When refreshing the project knowledge source to include V3 canonical status:

1. **Fix stale labels**: The `KNOWLEDGE_TARGETS` entries may have hardcoded `V1` labels
   (e.g., `CLOE_V1_CERTIFICATION`). Replace with canonical labels like
   `CLOE_CANONICAL_CERTIFICATION_STATUS`.

2. **Add V3 patterns**: Extend `patterns` arrays in each target to include
   `docs/mcporter/*CLOE_V3*`, `docs/architecture/*CLOE_V3*`, etc.

3. **Extract V3 fields**: In `scanCertification()`, after reading the continuity JSON,
   check for V3-specific fields like `baseline_freeze`, `v3_relationship`, `pr.merge_commit`,
   and `verdict` that contain `CLOE_V3`. Append these to the extracted summary.

4. **Verify**: Run `node -e "import('./src/cognitive/cloe-project-knowledge-source.js').then(m => m.scanProjectKnowledge(process.cwd()).then(r => r.facts.forEach(f => console.log(f.type, f.value.slice(0,200)))))"` and confirm `CLOE_CANONICAL_CERTIFICATION_STATUS` includes `V3_CERTIFICATION=`.

### Expected outcomes after each fix

| State | Raw Score | Strategic Verdict | Action |
|-------|-----------|-------------------|--------|
| Pre-V3 (no V3 modules) | 80 avg | PARTIAL | Implement V3 modules |
| V3 modules but stale knowledge | 54 avg, 1 leak | PARTIAL | Refresh knowledge source |
| V3 modules + fresh knowledge | ~74 avg, 0 leaks | PARTIAL (if 3+ weak answers) | Improve answer quality or re-benchmark |
| After knowledge refresh + stale detection fix | 78% weighted, 5/9 pass | PARTIAL (correct) | Re-run if provider had transient fallback |

### Stale Context Detection: Historical vs. Current

**Critical fix**: The stale context detector must distinguish between:
- **Stale**: References V1 as the CURRENT certification state (no V3 awareness)
- **Historical**: References V1 as PAST context while correctly stating V3 as current

Implementation:
```javascript
function findStaleContextIssues(answerText) {
  const hasV3Current = /V3|v3|CLOE_V3/.test(answerText);
  // If answer references V3 as current, V1 references are historical, not stale
  if (hasV3Current) return [];
  // ... original stale detection logic
}
```

Without this fix, the gate incorrectly penalizes answers that correctly reference
V1 as history while stating V3 as current — producing false positives.

### Observation Penalty Lifecycle

The `KNOWN_OBSERVATIONS` dictionary contains per-question notes and stale_context_risk
levels from manual review. These observations MUST be updated when the system under
evaluation changes:

| Event | Action |
|-------|--------|
| Knowledge source refreshed | Re-evaluate all observations: some stale_context risks drop from high/medium to low |
| selected_collectors leak fixed | Update devil_advocate observation (was stale/leaking, now correct) |
| Provider-backed answers change | Adjust observation notes to match current answer quality — old notes about pre-V3 answers no longer apply |

Without updating observations, the gate applies outdated penalties that penalize
the system for problems it no longer has.

### selected_collectors Leak: Origin and Fix

The leak originates from `src/brain/cognitive-context-formatters.js`:

```javascript
// Line 43 (BEFORE fix):
lines.push(`selected_collectors: ${formatList(pack.selected_collectors)}`);

// Line 43 (AFTER fix): remove the line entirely
```

This line outputs the list of selected knowledge collectors into the LLM context
prompt. The LLM occasionally reproduces this metadata in its user-facing answer,
triggering the benchmark runner's `internal_leak` detector.

Fix: remove the `selected_collectors` line from the formatter. The information
is still available in the cognitive context pack for the LLM to use implicitly,
but it no longer appears as a labeled field that the LLM can echo back.

### Transient Provider Fallback Handling

When running the provider-backed benchmark, the provider may be temporarily
unavailable for some questions. The benchmark runner handles this:

- Each question is independently attempted through the provider
- If provider fails for a question, it uses `local_fallback`
- The runner sets `provider_available=false` globally even if only 1/9 failed

**If you get 1 fallback**, check:
1. Was it the same question every time? → Possible timeout issue (increase `--timeout`)
2. Was it a different question each time? → Transient network/provider blip
3. Was it always the last question? → Possible cumulative timeout

**Recommended action**: if only 1 question fell back and it appears transient,
re-run the benchmark before scoring. A single transient fallback can lower the
average score enough to miss the 80% threshold.

### Scoring Model Refinements

The weighted strategic score aggregates across all 10 criteria:

```
totalWeightedScore = Σ(questionScore × questionWeight) / Σ(questionWeight)
```

Each criterion contributes:
```
criteriaScore = Σ(questionCriteriaScore × criteriaWeight) / Σ(criteriaWeight)
```

If the weighted score is 78% (below 80% threshold) but stale context is 0 and
leaks are 0, the gap is legitimate answer quality in specific questions.
Do not lower scoring thresholds — the threshold is intentionally high to ensure
behavioral quality matches technical quality.

## Running the Provider-Backed Benchmark

### One-shot run (recommended for certification)
```bash
cd lah-openclaw-mvp
npm run benchmark:cloe-operator
```

This runs all 9 questions sequentially through the CLI brain ask path.
Output includes JSON summary at the end with useful/partial/failed counts.

### Reading the results
```bash
python3 -c "
import json
with open('test/reports/cloe-operator-benchmark-latest.json') as f:
    d = json.load(f)
print('useful:', d.get('summary',{}).get('useful_count'))
print('partial:', d.get('summary',{}).get('partial_count'))
print('failed:', d.get('summary',{}).get('failed_count'))
print('fallback:', d.get('summary',{}).get('fallback_count'))
print('avg_score:', d.get('summary',{}).get('average_score'))
"
```

### Per-question details
```bash
python3 -c "
import json
with open('test/reports/cloe-operator-benchmark-latest.json') as f:
    d = json.load(f)
for q in d['questions']:
    ans = q.get('parsed_response',{}).get('data',{}).get('answer','')
    leaks = q.get('signals',{}).get('internal_leak',[])
    print(q['id'], 'score=' + str(q.get('score','?')), 'status=' + q.get('status','?'),
          'leaks=' + str(len(leaks)))
"
```

### Check provider availability without full benchmark
```bash
timeout 25 npm run openclaw -- brain ask \
  "Qui es-tu ? Reponds en une phrase." --json 2>/dev/null
```

### Handling npm preamble in JSON output
The `brain ask --json` command emits npm header lines before the JSON payload.
Parse by scanning for the JSON object:
```javascript
// In Node:
const lines = output.split('\n');
let jsonStr = '', braceCount = 0, inJson = false;
for (const ch of output) {
  if (ch === '{') { inJson = true; braceCount++; jsonStr += ch; }
  else if (ch === '}') { braceCount--; jsonStr += ch; if (braceCount === 0) break; }
  else if (inJson) jsonStr += ch;
}
const parsed = JSON.parse(jsonStr);
```

## Benchmark Internal Leak Patterns

The benchmark runner detects these answer-quality issues automatically:

| Pattern | What it detects | Severity |
|---------|----------------|----------|
| `selected_collectors` | Internal cognitive context pack field leaked in answer | critical |
| `safety_envelope` | Safety structure exposed to user | medium |
| `ReasoningContextEnvelope` | Reasoning internals leaked | critical |
| `system prompt` or `brain pack` | System prompt framing leaked | medium |
| `mutation_hardstop` or `governed_action_block` | Policy dump in answer | medium |

If a question scores 0 with internal_leak, fix the answer renderer
(`cloe-user-facing-answer-renderer.js` or equivalent) to strip
internal field names before presenting to the user.

## Pitfalls

### import.meta.url path resolution from src/cognitive/

The openclaw-runtime repo has git root at the repo level but source files in
lah-openclaw-mvp/src/cognitive/. When a module in src/cognitive/ loads a
data file via new URL(path, import.meta.url):

- import.meta.url = file:///.../src/cognitive/module.js
- ../ resolves within src/, not to the repo root
- To reach test/reports/, use ../../test/reports/ (up from src/cognitive/ -> src/ -> repo root)
- Using ../test/reports/ resolves to src/test/reports/ which doesn't exist

### Module vs Data parameter confusion

When building multi-module evaluation chains, track carefully whether each
function receives a module (with getters like .getV3StrategicMemory()) or
data (with fields like .canonical_status).

Top-level certification functions receive modules and call getters internally.
Utility/diagnostic functions receive data directly.
Passing the wrong type causes silent failures on undefined field access.

### Diagnosing which check fails

When a diagnostic returns ok=false without indicating which check failed,
run each check individually:
```
diag.diagnostics.forEach(d => console.log((d.passed ? 'PASS' : 'FAIL') + ' | ' + d.check));
```
