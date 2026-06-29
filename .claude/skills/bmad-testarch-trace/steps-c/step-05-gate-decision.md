---
name: 'step-05-gate-decision'
description: 'Phase 2: Apply gate decision logic and generate outputs'
outputFile: '{test_artifacts}/traceability-matrix.md'
---

# Step 5: Phase 2 - Gate Decision

## STEP GOAL

**Phase 2:** Read coverage matrix from Phase 1, apply deterministic gate decision logic when gate-eligible, and generate the traceability report plus machine-readable outputs.

---

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`
- ✅ Read coverage matrix from Phase 1 temp file
- ✅ Resolve collection status and gate eligibility before applying gate decision logic
- ❌ Do NOT regenerate coverage matrix (use Phase 1 output)

---

## EXECUTION PROTOCOLS:

- 🎯 Follow the MANDATORY SEQUENCE exactly
- 💾 Record outputs before proceeding
- 📖 This is the FINAL step

## CONTEXT BOUNDARIES:

- Available context: Coverage matrix from Phase 1 temp file
- Focus: gate decision logic only
- Dependencies: Phase 1 complete (coverage matrix exists)

---

## MANDATORY SEQUENCE

### 1. Read Phase 1 Coverage Matrix

Read `{outputFile}` frontmatter for `tempCoverageMatrixPath`. Halt when missing — the fallback timestamp cannot be reconstructed reliably in a different execution context:

```javascript
const progressDoc = fs.readFileSync('{outputFile}', 'utf8');
const frontmatterMatch = progressDoc.match(/^---\n([\s\S]*?)\n---/);
const frontmatter = frontmatterMatch ? yaml.parse(frontmatterMatch[1]) : {};

const matrixPath = frontmatter.tempCoverageMatrixPath;
if (!matrixPath) {
  throw new Error(
    '❌ tempCoverageMatrixPath not found in progress frontmatter. ' +
      'Step 4 must record the resolved temp file path before Step 5 can proceed.',
  );
}
const coverageMatrix = JSON.parse(fs.readFileSync(matrixPath, 'utf8'));

console.log('✅ Phase 1 coverage matrix loaded');
```

**Verify Phase 1 complete:**

```javascript
if (coverageMatrix.phase !== 'PHASE_1_COMPLETE') {
  throw new Error('Phase 1 not complete - cannot proceed to gate decision');
}
```

---

### 2. Apply Gate Decision Logic

**Decision Tree:**

```javascript
const stats = coverageMatrix.coverage_statistics;
if (
  !stats ||
  typeof stats !== 'object' ||
  !stats.priority_breakdown ||
  !stats.priority_breakdown.P0 ||
  !stats.priority_breakdown.P1 ||
  !stats.priority_breakdown.P2 ||
  !stats.priority_breakdown.P3
) {
  throw new Error(
    'Phase 1 coverage_statistics.priority_breakdown is missing or incomplete. ' +
      'Step 4 must emit P0-P3 totals and coverage percentages before Step 5 can proceed.',
  );
}
const priorityBreakdown = stats.priority_breakdown;
const p0Coverage = priorityBreakdown.P0.percentage;
const p1Coverage = priorityBreakdown.P1.percentage;
const hasP1Requirements = (priorityBreakdown.P1.total || 0) > 0;
const effectiveP1Coverage = hasP1Requirements ? p1Coverage : 100;
const overallCoverage = stats.overall_coverage_percentage;
const criticalGaps = (coverageMatrix.gap_analysis?.critical_gaps || []).length;
const isUnresolved = (value) => typeof value === 'string' && value.startsWith('{') && value.endsWith('}');
const normalizeResolvedToken = (value) => {
  if (value === undefined || value === null) return null;
  const normalized = String(value).trim().toLowerCase();
  if (!normalized || normalized === 'auto' || isUnresolved(normalized)) return null;
  return normalized;
};
const oracleResolutionMode = normalizeResolvedToken(coverageMatrix.oracle?.resolution_mode) || 'formal_requirements';
const coverageBasis =
  normalizeResolvedToken(coverageMatrix.coverage_basis) ||
  {
    formal_requirements: 'acceptance_criteria',
    spec_artifact: 'openapi_endpoints',
    external_pointer: 'acceptance_criteria',
    synthetic_source: 'user_journeys',
  }[oracleResolutionMode] ||
  'acceptance_criteria';
const oracleConfidence =
  normalizeResolvedToken(coverageMatrix.oracle?.confidence || coverageMatrix.summary_confidence) ||
  {
    formal_requirements: 'high',
    spec_artifact: 'high',
    external_pointer: 'medium',
    synthetic_source: 'medium',
  }[oracleResolutionMode] ||
  'medium';
const syntheticOracle = coverageMatrix.oracle?.synthetic === true || ['synthetic_requirements', 'user_journeys'].includes(coverageBasis);
const deriveActiveTestCasesFromRequirements = (requirements) => {
  const uniqueTests = new Map();

  (requirements || []).forEach((req) => {
    (req.tests || []).forEach((test) => {
      const stableId =
        test.id ||
        [test.file, test.title || test.name, test.line]
          .filter((value) => value !== undefined && value !== null && value !== '')
          .join(':') ||
        null;

      if (stableId === null || uniqueTests.has(stableId)) return;

      const explicitStatus = String(test.status || '')
        .trim()
        .toLowerCase();
      const status = ['skipped', 'pending', 'fixme'].includes(explicitStatus)
        ? explicitStatus
        : test.fixme === true
          ? 'fixme'
          : test.pending === true
            ? 'pending'
            : test.skipped === true
              ? 'skipped'
              : 'active';

      uniqueTests.set(stableId, status);
    });
  });

  return [...uniqueTests.values()].filter((status) => status === 'active').length;
};
const summarizedTestInventory = coverageMatrix.test_inventory?.summary || null;
const activeTestCases =
  summarizedTestInventory === null
    ? deriveActiveTestCasesFromRequirements(coverageMatrix.requirements)
    : Math.max(
        0,
        (summarizedTestInventory.cases || 0) -
          (summarizedTestInventory.skipped_cases || 0) -
          (summarizedTestInventory.fixme_cases || 0) -
          (summarizedTestInventory.pending_cases || 0),
      );
let effectiveOracleConfidence = oracleConfidence;
if (effectiveOracleConfidence === 'high' && activeTestCases === 0) {
  effectiveOracleConfidence = 'medium';
}

const normalizeBoolean = (value, defaultValue = true) => {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['false', '0', 'off', 'no'].includes(normalized)) return false;
    if (['true', '1', 'on', 'yes'].includes(normalized)) return true;
  }
  if (value === undefined || value === null) return defaultValue;
  return Boolean(value);
};

const collectionMode = String(!isUnresolved(coverageMatrix.collection_mode) ? coverageMatrix.collection_mode : 'contract_static')
  .trim()
  .toLowerCase();
const rawAllowGate = !isUnresolved(coverageMatrix.allow_gate) ? coverageMatrix.allow_gate : true;
const allowGate = normalizeBoolean(rawAllowGate, true);
const rawCollectionStatus =
  coverageMatrix.collection_status ||
  {
    waived: 'WAIVED',
    restricted: 'RESTRICTED',
    inaccessible: 'INACCESSIBLE',
    deferred_shared: 'DEFERRED_SHARED',
  }[collectionMode] ||
  'COLLECTED';
// Normalize to UPPER_CASE + trimmed so comparisons are whitespace/case-safe.
const collectionStatus = String(rawCollectionStatus).trim().toUpperCase();
const gateEligible = allowGate && collectionStatus === 'COLLECTED';

let gateDecision = 'NOT_EVALUATED'; // default; overwritten when gateEligible
let rationale;

if (!gateEligible) {
  rationale = `Gate decision skipped because allow_gate=${allowGate} and collection_status=${collectionStatus}.`;
} else {
  // Rule 1: P0 coverage must be 100%
  if (p0Coverage < 100) {
    gateDecision = 'FAIL';
    rationale = `P0 coverage is ${p0Coverage}% (required: 100%). ${criticalGaps} critical requirements uncovered.`;
  }
  // Rule 2: Overall coverage must be >= 80%
  else if (overallCoverage < 80) {
    gateDecision = 'FAIL';
    rationale = `Overall coverage is ${overallCoverage}% (minimum: 80%). Significant gaps exist.`;
  }
  // Rule 3: P1 coverage < 80% → FAIL
  else if (effectiveP1Coverage < 80) {
    gateDecision = 'FAIL';
    rationale = hasP1Requirements
      ? `P1 coverage is ${effectiveP1Coverage}% (minimum: 80%). High-priority gaps must be addressed.`
      : `P1 requirements are not present; continuing with remaining gate criteria.`;
  }
  // Rule 4: P1 coverage >= 90% and overall >= 80% with P0 at 100% → PASS
  else if (effectiveP1Coverage >= 90) {
    gateDecision = 'PASS';
    rationale = hasP1Requirements
      ? `P0 coverage is 100%, P1 coverage is ${effectiveP1Coverage}% (target: 90%), and overall coverage is ${overallCoverage}% (minimum: 80%).`
      : `P0 coverage is 100% and overall coverage is ${overallCoverage}% (minimum: 80%). No P1 requirements detected.`;
  }
  // Rule 5: P1 coverage 80-89% with P0 at 100% and overall >= 80% → CONCERNS
  else if (effectiveP1Coverage >= 80) {
    gateDecision = 'CONCERNS';
    rationale = hasP1Requirements
      ? `P0 coverage is 100% and overall coverage is ${overallCoverage}% (minimum: 80%), but P1 coverage is ${effectiveP1Coverage}% (target: 90%).`
      : `P0 coverage is 100% and overall coverage is ${overallCoverage}% (minimum: 80%), but additional non-P1 gaps need mitigation.`;
  }

  // Rule 6: Manual waiver — set gateDecision = 'WAIVED' and update rationale here
  // if a stakeholder-approved waiver applies (wired through config or user input upstream).

  // Oracle confidence overlay
  if (syntheticOracle && gateDecision === 'PASS' && effectiveOracleConfidence !== 'high') {
    gateDecision = 'CONCERNS';
    rationale =
      `Coverage traced against inferred ${coverageBasis.replace('_', ' ')} with ${effectiveOracleConfidence} confidence. ` +
      `Base coverage meets PASS thresholds, but confidence is not high enough for an unconditional PASS.`;
  } else if (syntheticOracle && effectiveOracleConfidence === 'low' && gateDecision === 'NOT_EVALUATED') {
    gateDecision = 'CONCERNS';
    rationale =
      `Coverage traced against inferred ${coverageBasis.replace('_', ' ')} with low confidence. ` +
      `Treat this result as advisory until the inferred journeys are confirmed or formalized.`;
  }
}
```

---

### 3. Generate Gate Report

```javascript
const gateReport = {
  gate_eligible: gateEligible,
  collection_status: collectionStatus,
  decision: gateEligible ? gateDecision : 'NOT_EVALUATED',
  rationale: rationale,
  decision_date: new Date().toISOString(),

  coverage_matrix: coverageMatrix,

  gate_criteria: gateEligible
    ? {
        p0_coverage_required: '100%',
        p0_coverage_actual: `${p0Coverage}%`,
        p0_status: p0Coverage === 100 ? 'MET' : 'NOT_MET',

        p1_coverage_target: '90%',
        p1_coverage_minimum: '80%',
        p1_coverage_actual: `${effectiveP1Coverage}%`,
        p1_status: effectiveP1Coverage >= 90 ? 'MET' : effectiveP1Coverage >= 80 ? 'PARTIAL' : 'NOT_MET',

        overall_coverage_minimum: '80%',
        overall_coverage_actual: `${overallCoverage}%`,
        overall_status: overallCoverage >= 80 ? 'MET' : 'NOT_MET',
      }
    : null,

  uncovered_requirements: (coverageMatrix.gap_analysis?.critical_gaps || []).concat(coverageMatrix.gap_analysis?.high_gaps || []),

  recommendations: coverageMatrix.recommendations,
};
```

---

### 3b. Emit `e2e-trace-summary.json`

**After the gate report is assembled, write the machine-readable summary to `{e2e_trace_summary_output}`.**

This file is the portable, automation-friendly companion to the markdown report. Any CI/CD pipeline, reporting dashboard, or LLM agent can consume it without parsing markdown.

```javascript
const buildFallbackInventory = () => {
  const byLevel = {
    e2e: { tests: 0, criteria_covered: 0 },
    api: { tests: 0, criteria_covered: 0 },
    component: { tests: 0, criteria_covered: 0 },
    unit: { tests: 0, criteria_covered: 0 },
    other: { tests: 0, criteria_covered: 0 }, // captures tests with unrecognized or empty level
  };
  const coverageEligibleStatuses = new Set(['FULL', 'PARTIAL', 'UNIT-ONLY', 'INTEGRATION-ONLY']);
  const uniqueTests = new Map();

  (coverageMatrix.requirements || []).forEach((req) => {
    (req.tests || []).forEach((test) => {
      const stableId =
        test.id ||
        [test.file, test.title || test.name, test.line]
          .filter((value) => value !== undefined && value !== null && value !== '')
          .join(':') ||
        null; // unresolvable — skip rather than manufacture a key

      if (stableId === null || uniqueTests.has(stableId)) return;
      const explicitStatus = String(test.status || '')
        .trim()
        .toLowerCase();
      const status = ['skipped', 'pending', 'fixme'].includes(explicitStatus)
        ? explicitStatus
        : test.fixme === true
          ? 'fixme'
          : test.pending === true
            ? 'pending'
            : test.skipped === true
              ? 'skipped'
              : 'active';

      uniqueTests.set(stableId, {
        id: stableId,
        file: test.file || '',
        title: test.title || test.name || stableId,
        level: String(test.level || '')
          .trim()
          .toLowerCase(),
        skipped: status === 'skipped',
        fixme: status === 'fixme',
        pending: status === 'pending',
        status: status,
        blocker_reason: test.skip_reason || test.blocker_reason || test.fixme_reason || test.pending_reason || '',
      });
    });

    if (!coverageEligibleStatuses.has(req.coverage)) return;
    const requirementLevels = new Set(
      (req.tests || []).map((test) => {
        const level = String(test.level || '')
          .trim()
          .toLowerCase();
        return byLevel[level] ? level : 'other';
      }),
    );
    requirementLevels.forEach((level) => {
      byLevel[level].criteria_covered += 1;
    });
  });

  const deduplicatedTests = [...uniqueTests.values()];
  deduplicatedTests.forEach((test) => {
    const bucket = byLevel[test.level] ? test.level : 'other';
    byLevel[bucket].tests += 1;
  });

  return {
    summary: {
      files: [...new Set(deduplicatedTests.map((test) => test.file).filter(Boolean))].length,
      cases: deduplicatedTests.length,
      skipped_cases: deduplicatedTests.filter((test) => test.skipped).length,
      fixme_cases: deduplicatedTests.filter((test) => test.fixme).length,
      pending_cases: deduplicatedTests.filter((test) => test.pending).length,
      by_level: byLevel,
    },
    blockers: deduplicatedTests
      .filter((test) => ['skipped', 'pending', 'fixme'].includes(test.status))
      .map((test) => ({
        id: test.id,
        severity: test.status === 'skipped' ? 'high' : 'medium',
        reason: test.blocker_reason || `Test marked ${test.status} during trace collection`,
        test_file: test.file,
        test_title: test.title,
      })),
  };
};

const fallbackInventory = buildFallbackInventory();
const testInventory = coverageMatrix.test_inventory?.summary || fallbackInventory.summary;
const blockers = coverageMatrix.blockers || coverageMatrix.test_inventory?.blockers || fallbackInventory.blockers;

const heuristicCounts = coverageMatrix.coverage_heuristics?.counts || {};
const endpointGapCount = heuristicCounts.endpoints_without_tests ?? 0;
const authGapCount = heuristicCounts.auth_missing_negative_paths ?? 0;
const errorPathGapCount = heuristicCounts.happy_path_only_criteria ?? 0;
const uiJourneyGapCount = heuristicCounts.ui_journeys_without_e2e;
const uiStateGapCount = heuristicCounts.ui_states_missing_coverage;
const sourceSha = process.env.GITHUB_SHA || runtime.getSourceSha?.() || '';
const mapOptionalHeuristicStatus = (count, applicable) => {
  if (!applicable) return 'not_applicable';
  if (typeof count !== 'number' || Number.isNaN(count)) return 'unknown';
  if (count === 0) return 'present';
  return count <= 2 ? 'partial' : 'none';
};
const gateBasis = gateEligible ? 'priority_thresholds' : 'none';

const e2eTraceSummary = {
  schema_version: '0.1.0',
  snapshot_at: new Date().toISOString(),
  repo: '{project_name}',
  collection_mode: collectionMode,
  collection_status: collectionStatus,
  inventory_basis: coverageBasis,
  gate_basis: gateBasis,
  source_sha: sourceSha || '',
  target: coverageMatrix.trace_target || { type: '{gate_type}', id: null, label: null },
  decision_mode: '{decision_mode}',
  evaluator: '{user_name}',
  confidence: effectiveOracleConfidence,
  oracle: {
    resolution_mode: oracleResolutionMode,
    confidence: effectiveOracleConfidence,
    sources: coverageMatrix.oracle?.sources || [],
    external_pointer_status: coverageMatrix.oracle?.external_pointer_status || 'not_used',
    synthetic: syntheticOracle,
  },

  coverage: {
    inventory: {
      covered: stats.fully_covered,
      total: stats.total_requirements,
      pct: stats.overall_coverage_percentage,
    },
    priority_breakdown: {
      P0: {
        total: priorityBreakdown.P0.total,
        covered: priorityBreakdown.P0.covered,
        pct: priorityBreakdown.P0.percentage,
      },
      P1: {
        total: priorityBreakdown.P1.total,
        covered: priorityBreakdown.P1.covered,
        pct: priorityBreakdown.P1.percentage,
      },
      P2: {
        total: priorityBreakdown.P2.total,
        covered: priorityBreakdown.P2.covered,
        pct: priorityBreakdown.P2.percentage,
      },
      P3: {
        total: priorityBreakdown.P3.total,
        covered: priorityBreakdown.P3.covered,
        pct: priorityBreakdown.P3.percentage,
      },
    },
    by_level: testInventory.by_level,
  },

  tests: {
    files: testInventory.files || 0,
    cases: testInventory.cases || 0,
    skipped_cases: testInventory.skipped_cases || 0,
    fixme_cases: testInventory.fixme_cases || 0,
    pending_cases: testInventory.pending_cases || 0,
  },

  risk_summary: {
    critical_open: (coverageMatrix.gap_analysis?.critical_gaps || []).length,
    high_open: (coverageMatrix.gap_analysis?.high_gaps || []).length,
    medium_open: (coverageMatrix.gap_analysis?.medium_gaps || []).length,
    low_open: (coverageMatrix.gap_analysis?.low_gaps || []).length,
  },

  heuristics: {
    endpoint_gaps: endpointGapCount,
    auth_negative_path_status: authGapCount === 0 ? 'present' : authGapCount <= 2 ? 'partial' : 'none',
    error_path_status: errorPathGapCount === 0 ? 'present' : errorPathGapCount <= 2 ? 'partial' : 'none',
    ui_journey_status: mapOptionalHeuristicStatus(uiJourneyGapCount, syntheticOracle),
    ui_state_status: mapOptionalHeuristicStatus(uiStateGapCount, syntheticOracle),
  },

  blockers: blockers,
  recommendations: coverageMatrix.recommendations,

  links: {
    trace_report_path: '{outputFile}',
    trace_report_url: '', // populated by CI/CD runner after artifact upload
    artifact_url: '',
    journey_evidence_url: '',
  },
};

if (gateEligible) {
  e2eTraceSummary.gate_status = gateDecision;
  e2eTraceSummary.gate_criteria = {
    p0_coverage_required: '100%',
    p0_coverage_actual: `${p0Coverage}%`,
    p0_status: p0Coverage === 100 ? 'MET' : 'NOT_MET',
    p1_coverage_target: '90%',
    p1_coverage_minimum: '80%',
    p1_coverage_actual: `${effectiveP1Coverage}%`,
    p1_status: effectiveP1Coverage >= 90 ? 'MET' : effectiveP1Coverage >= 80 ? 'PARTIAL' : 'NOT_MET',
    overall_coverage_minimum: '80%',
    overall_coverage_actual: `${overallCoverage}%`,
    overall_status: overallCoverage >= 80 ? 'MET' : 'NOT_MET',
  };
}

fs.writeFileSync('{e2e_trace_summary_output}', JSON.stringify(e2eTraceSummary, null, 2), 'utf8');
console.log(`✅ e2e-trace-summary.json written to {e2e_trace_summary_output}`);
```

**Optional: emit `gate-decision.json`** for pipelines that only need the gate signal without the full summary:

```javascript
// Construct and write only when gate evaluation was performed and produced a meaningful decision.
// gateDecisionSlim is intentionally inside this guard: e2eTraceSummary.gate_criteria is only
// populated when gateEligible is true, so constructing it outside would throw when !gateEligible.
if (gateEligible && ['PASS', 'CONCERNS', 'FAIL', 'WAIVED'].includes(gateDecision)) {
  const gateDecisionSlim = {
    schema_version: '0.1.0',
    evaluated_at: e2eTraceSummary.snapshot_at,
    repo: e2eTraceSummary.repo,
    target: e2eTraceSummary.target,
    collection_status: e2eTraceSummary.collection_status,
    gate_basis: e2eTraceSummary.gate_basis,
    gate_status: gateDecision,
    rationale: rationale,
    p0_status: e2eTraceSummary.gate_criteria.p0_status,
    p1_status: e2eTraceSummary.gate_criteria.p1_status,
    overall_status: e2eTraceSummary.gate_criteria.overall_status,
    critical_open: e2eTraceSummary.risk_summary.critical_open,
    links: e2eTraceSummary.links,
  };
  fs.writeFileSync('{gate_decision_output}', JSON.stringify(gateDecisionSlim, null, 2), 'utf8');
  console.log(`✅ gate-decision.json written to {gate_decision_output}`);
}
```

---

### 4. Generate Traceability Report

**Use trace-template.md to generate:**

```markdown
# Traceability Report

## Gate Decision: {gateDecision}

**Rationale:** {rationale}

## Coverage Summary

- Total Requirements: {totalRequirements}
- Covered: {fullyCovered} ({coveragePercentage}%)
- P0 Coverage: {p0CoveragePercentage}%

## Traceability Matrix

[Full matrix with requirement → test mappings]

## Gaps & Recommendations

[List of uncovered requirements with recommended actions]

## Next Actions

{recommendations}
```

**Save to:**

```javascript
fs.writeFileSync('{outputFile}', reportContent, 'utf8');
```

---

### 5. Display Gate Decision

```
🚨 GATE DECISION: {gateDecision}

📊 Coverage Analysis:
- P0 Coverage: {p0Coverage}% (Required: 100%) → {p0_status}
- P1 Coverage: {effectiveP1Coverage}% (PASS target: 90%, minimum: 80%) → {p1_status}
- Overall Coverage: {overallCoverage}% (Minimum: 80%) → {overall_status}

✅ Decision Rationale:
{rationale}

⚠️ Critical Gaps: {criticalGaps.length}

📝 Recommended Actions:
{list top 3 recommendations}

📂 Full Report: {outputFile}

{if !gateEligible}
ℹ️ GATE: NOT EVALUATED - collection status is {collectionStatus}; machine-readable summary still emitted
{endif}

{if FAIL}
🚫 GATE: FAIL - Release BLOCKED until coverage improves
{endif}

{if CONCERNS}
⚠️ GATE: CONCERNS - Proceed with caution, address gaps soon
{endif}

{if PASS}
✅ GATE: PASS - Release approved, coverage meets standards
{endif}
```

---

### 6. Save Progress

**Update the YAML frontmatter in `{outputFile}` to mark this final step complete.**

Since step 4 (Generate Traceability Report) already wrote the report content to `{outputFile}`, do NOT overwrite it. Instead, update only the frontmatter at the top of the existing file:

- Add `'step-05-gate-decision'` to `stepsCompleted` array (only if not already present)
- Set `lastStep: 'step-05-gate-decision'`
- Set `lastSaved: '{date}'`

Then append the gate decision summary (from section 5 above) to the end of the existing report content.

---

## EXIT CONDITION

**WORKFLOW COMPLETE when:**

- ✅ Phase 1 coverage matrix read successfully
- ✅ Collection status resolved and gate decision logic applied when eligible
- ✅ `e2e-trace-summary.json` written to `{e2e_trace_summary_output}`
- ✅ `gate-decision.json` written to `{gate_decision_output}` (when gate-eligible)
- ✅ Traceability report generated
- ✅ Gate decision displayed

**Workflow terminates here.**

---

## 🚨 PHASE 2 SUCCESS METRICS

### ✅ SUCCESS:

- Coverage matrix read from Phase 1
- Gate decision made with clear rationale when gate-eligible
- `e2e-trace-summary.json` written and valid
- `gate-decision.json` written when gate-eligible
- Report generated and saved
- Decision communicated clearly

### ❌ FAILURE:

- Could not read Phase 1 matrix
- Gate eligibility or gate decision logic incorrect
- `e2e-trace-summary.json` missing or invalid JSON
- Report missing or incomplete

**Master Rule:** Gate decision MUST be deterministic based on clear criteria (P0 100%, P1 90/80, overall >=80) whenever `allow_gate` is true and `collection_status` is `COLLECTED`. `e2e-trace-summary.json` MUST be written before the workflow terminates.

## On Complete

Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow.on_complete`

If the resolver succeeds and returns a non-empty `workflow.on_complete`, execute that value as the final terminal instruction before exiting.

If the resolver fails, returns no output, or resolves an empty value, skip the hook and exit normally.
