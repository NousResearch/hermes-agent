---
name: 'step-04-analyze-gaps'
description: 'Complete Phase 1 with adaptive orchestration (agent-team, subagent, or sequential)'
nextStepFile: '{skill-root}/steps-c/step-05-gate-decision.md'
outputFile: '{test_artifacts}/traceability-matrix.md'
tempOutputFile: '/tmp/tea-trace-coverage-matrix-{{timestamp}}.json'
---

# Step 4: Complete Phase 1 - Coverage Matrix Generation

## STEP GOAL

**Phase 1 Final Step:** Analyze coverage gaps (including endpoint/auth/error-path blind spots), generate recommendations, and output complete coverage matrix to temp file for Phase 2 (gate decision).

---

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`
- ✅ Output coverage matrix to temp file
- ✅ Resolve execution mode from explicit user request first, then config
- ✅ Apply fallback rules deterministically when requested mode is unsupported
- ❌ Do NOT make gate decision (that's Phase 2 - Step 5)

---

## EXECUTION PROTOCOLS:

- 🎯 Follow the MANDATORY SEQUENCE exactly
- 💾 Record outputs before proceeding
- 📖 Load the next step only when instructed

## CONTEXT BOUNDARIES:

- Available context: resolved oracle items from Step 1, tests from Step 2, traceability matrix from Step 3
- Focus: gap analysis and matrix completion
- Limits: do not make gate decision (Phase 2 responsibility)

---

## MANDATORY SEQUENCE

### 0. Resolve Execution Mode (User Override First)

```javascript
const parseBooleanFlag = (value, defaultValue = true) => {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['false', '0', 'off', 'no'].includes(normalized)) return false;
    if (['true', '1', 'on', 'yes'].includes(normalized)) return true;
  }
  if (value === undefined || value === null) return defaultValue;
  return Boolean(value);
};

const orchestrationContext = {
  config: {
    execution_mode: config.tea_execution_mode || 'auto', // "auto" | "subagent" | "agent-team" | "sequential"
    capability_probe: parseBooleanFlag(config.tea_capability_probe, true), // supports booleans and "false"/"true" strings
  },
  timestamp: new Date().toISOString().replace(/[:.]/g, '-'),
};

const normalizeUserExecutionMode = (mode) => {
  if (typeof mode !== 'string') return null;
  const normalized = mode.trim().toLowerCase().replace(/[-_]/g, ' ').replace(/\s+/g, ' ');

  if (normalized === 'auto') return 'auto';
  if (normalized === 'sequential') return 'sequential';
  if (normalized === 'subagent' || normalized === 'sub agent' || normalized === 'subagents' || normalized === 'sub agents') {
    return 'subagent';
  }
  if (normalized === 'agent team' || normalized === 'agent teams' || normalized === 'agentteam') {
    return 'agent-team';
  }

  return null;
};

const normalizeConfigExecutionMode = (mode) => {
  if (mode === 'subagent') return 'subagent';
  if (mode === 'auto' || mode === 'sequential' || mode === 'subagent' || mode === 'agent-team') {
    return mode;
  }
  return null;
};

// Explicit user instruction in the active run takes priority over config.
const explicitModeFromUser = normalizeUserExecutionMode(runtime.getExplicitExecutionModeHint?.() || null);

const requestedMode = explicitModeFromUser || normalizeConfigExecutionMode(orchestrationContext.config.execution_mode) || 'auto';
const probeEnabled = orchestrationContext.config.capability_probe;

const supports = { subagent: false, agentTeam: false };
if (probeEnabled) {
  supports.subagent = runtime.canLaunchSubagents?.() === true;
  supports.agentTeam = runtime.canLaunchAgentTeams?.() === true;
}

let resolvedMode = requestedMode;
if (requestedMode === 'auto') {
  if (supports.agentTeam) resolvedMode = 'agent-team';
  else if (supports.subagent) resolvedMode = 'subagent';
  else resolvedMode = 'sequential';
} else if (probeEnabled && requestedMode === 'agent-team' && !supports.agentTeam) {
  resolvedMode = supports.subagent ? 'subagent' : 'sequential';
} else if (probeEnabled && requestedMode === 'subagent' && !supports.subagent) {
  resolvedMode = 'sequential';
}
```

Resolution precedence:

1. Explicit user request in this run (`agent team` => `agent-team`; `subagent` => `subagent`; `sequential`; `auto`)
2. `tea_execution_mode` from config
3. Runtime capability fallback (when probing enabled)

### 1. Gap Analysis

**Identify uncovered requirements:**

```javascript
const uncoveredRequirements = traceabilityMatrix.filter((req) => req.coverage === 'NONE');
const partialCoverage = traceabilityMatrix.filter((req) => req.coverage === 'PARTIAL');
const unitOnlyCoverage = traceabilityMatrix.filter((req) => req.coverage === 'UNIT-ONLY');
```

**Prioritize gaps by risk:**

```javascript
const criticalGaps = uncoveredRequirements.filter((req) => req.priority === 'P0');
const highGaps = uncoveredRequirements.filter((req) => req.priority === 'P1');
const mediumGaps = uncoveredRequirements.filter((req) => req.priority === 'P2');
const lowGaps = uncoveredRequirements.filter((req) => req.priority === 'P3');
```

---

### 2. Coverage Heuristics Checks

Use the heuristics inventory from Step 2 and mapped criteria from Step 3 to flag common coverage blind spots:

```javascript
const endpointCoverageGaps = coverageHeuristics?.endpoints_without_tests || [];
const authCoverageGaps = coverageHeuristics?.auth_missing_negative_paths || [];
const errorPathGaps = coverageHeuristics?.criteria_happy_path_only || [];
const uiJourneyGaps = coverageHeuristics?.ui_journeys_without_e2e || [];
const uiStateGaps = coverageHeuristics?.ui_states_missing_coverage || [];

const heuristicGapCounts = {
  endpoints_without_tests: endpointCoverageGaps.length,
  auth_missing_negative_paths: authCoverageGaps.length,
  happy_path_only_criteria: errorPathGaps.length,
  ui_journeys_without_e2e: uiJourneyGaps.length,
  ui_states_missing_coverage: uiStateGaps.length,
};
```

Heuristics are advisory but must influence gap severity and recommendations, especially for P0/P1 criteria.

---

### 3. Generate Recommendations

**Based on gap analysis:**

```javascript
const progressDoc = fs.existsSync('{outputFile}') ? fs.readFileSync('{outputFile}', 'utf8') : '';
const progressFrontmatterMatch = progressDoc.match(/^---\n([\s\S]*?)\n---/);
const progressFrontmatter = progressFrontmatterMatch ? yaml.parse(progressFrontmatterMatch[1]) : {};

const isUnresolved = (value) => typeof value === 'string' && value.startsWith('{') && value.endsWith('}');
const normalizeResolvedToken = (value) => {
  if (value === undefined || value === null) return null;
  const normalized = String(value).trim().toLowerCase();
  if (!normalized || normalized === 'auto' || isUnresolved(normalized)) return null;
  return normalized;
};
const firstResolvedToken = (...values) => {
  for (const value of values) {
    const normalized = normalizeResolvedToken(value);
    if (normalized) return normalized;
  }
  return null;
};

const oracleResolutionMode =
  firstResolvedToken(runtime.getOracleResolutionMode?.(), progressFrontmatter.oracleResolutionMode) || 'formal_requirements';
const resolvedCoverageBasis =
  firstResolvedToken(runtime.getResolvedCoverageBasis?.(), progressFrontmatter.coverageBasis) ||
  {
    formal_requirements: 'acceptance_criteria',
    spec_artifact: 'openapi_endpoints',
    external_pointer: 'acceptance_criteria',
    synthetic_source: 'user_journeys',
  }[oracleResolutionMode] ||
  'acceptance_criteria';
const resolvedOracleConfidence =
  firstResolvedToken(runtime.getResolvedOracleConfidence?.(), progressFrontmatter.oracleConfidence) ||
  {
    formal_requirements: 'high',
    spec_artifact: 'high',
    external_pointer: 'medium',
    synthetic_source: 'medium',
  }[oracleResolutionMode] ||
  'medium';
const oracleSources = runtime.getOracleSources?.() || progressFrontmatter.oracleSources || [];
const externalPointerStatus =
  firstResolvedToken(runtime.getExternalPointerStatus?.(), progressFrontmatter.externalPointerStatus) || 'not_used';
const recommendations = [];

// Critical gaps (P0)
if (criticalGaps.length > 0) {
  recommendations.push({
    priority: 'URGENT',
    action: `Run /bmad:tea:atdd for ${criticalGaps.length} P0 requirements`,
    requirements: criticalGaps.map((r) => r.id),
  });
}

// High priority gaps (P1)
if (highGaps.length > 0) {
  recommendations.push({
    priority: 'HIGH',
    action: `Run /bmad:tea:automate to expand coverage for ${highGaps.length} P1 requirements`,
    requirements: highGaps.map((r) => r.id),
  });
}

// Partial coverage
if (partialCoverage.length > 0) {
  recommendations.push({
    priority: 'MEDIUM',
    action: `Complete coverage for ${partialCoverage.length} partially covered requirements`,
    requirements: partialCoverage.map((r) => r.id),
  });
}

if (endpointCoverageGaps.length > 0) {
  recommendations.push({
    priority: 'HIGH',
    action: `Add API tests for ${endpointCoverageGaps.length} uncovered endpoint(s)`,
    requirements: endpointCoverageGaps.map((r) => r.id || r.endpoint || 'unknown'),
  });
}

if (authCoverageGaps.length > 0) {
  recommendations.push({
    priority: 'HIGH',
    action: `Add negative-path auth/authz tests for ${authCoverageGaps.length} requirement(s)`,
    requirements: authCoverageGaps.map((r) => r.id || 'unknown'),
  });
}

if (errorPathGaps.length > 0) {
  recommendations.push({
    priority: 'MEDIUM',
    action: `Add error/edge scenario tests for ${errorPathGaps.length} happy-path-only criterion/criteria`,
    requirements: errorPathGaps.map((r) => r.id || 'unknown'),
  });
}

if (uiJourneyGaps.length > 0) {
  recommendations.push({
    priority: 'HIGH',
    action: `Add E2E or component coverage for ${uiJourneyGaps.length} inferred UI journey(s)`,
    requirements: uiJourneyGaps.map((r) => r.id || r.route || r.journey || 'unknown'),
  });
}

if (uiStateGaps.length > 0) {
  recommendations.push({
    priority: 'MEDIUM',
    action: `Add loading/empty/error/permission state coverage for ${uiStateGaps.length} UI journey(s)`,
    requirements: uiStateGaps.map((r) => r.id || r.route || r.journey || 'unknown'),
  });
}

// Quality issues
recommendations.push({
  priority: 'LOW',
  action: 'Run /bmad:tea:test-review to assess test quality',
  requirements: [],
});

if (oracleResolutionMode === 'synthetic_source') {
  recommendations.push({
    priority: 'MEDIUM',
    action: 'Promote inferred journeys into formal acceptance criteria when the team confirms they reflect intended behavior',
    requirements: traceabilityMatrix.map((r) => r.id),
  });
}
```

---

### 4. Calculate Coverage Statistics

```javascript
const totalRequirements = traceabilityMatrix.length;
const coveredRequirements = traceabilityMatrix.filter((r) => r.coverage === 'FULL' || r.coverage === 'PARTIAL').length;
const fullyCovered = traceabilityMatrix.filter((r) => r.coverage === 'FULL').length;

const safePct = (covered, total) => (total > 0 ? Math.round((covered / total) * 100) : 100);
const coveragePercentage = safePct(fullyCovered, totalRequirements);

// Priority-specific coverage
const p0Total = traceabilityMatrix.filter((r) => r.priority === 'P0').length;
const p0Covered = traceabilityMatrix.filter((r) => r.priority === 'P0' && r.coverage === 'FULL').length;
const p1Total = traceabilityMatrix.filter((r) => r.priority === 'P1').length;
const p1Covered = traceabilityMatrix.filter((r) => r.priority === 'P1' && r.coverage === 'FULL').length;
const p2Total = traceabilityMatrix.filter((r) => r.priority === 'P2').length;
const p2Covered = traceabilityMatrix.filter((r) => r.priority === 'P2' && r.coverage === 'FULL').length;
const p3Total = traceabilityMatrix.filter((r) => r.priority === 'P3').length;
const p3Covered = traceabilityMatrix.filter((r) => r.priority === 'P3' && r.coverage === 'FULL').length;

const p0CoveragePercentage = safePct(p0Covered, p0Total);
const p1CoveragePercentage = safePct(p1Covered, p1Total);
const p2CoveragePercentage = safePct(p2Covered, p2Total);
const p3CoveragePercentage = safePct(p3Covered, p3Total);
```

---

### 4b. Build Deduplicated Test Inventory and Trace Metadata

Persist the unique discovered tests in Phase 1 so Step 5 does not need to reconstruct counts from per-requirement mappings.

```javascript
const coverageEligibleStatuses = new Set(['FULL', 'PARTIAL', 'UNIT-ONLY', 'INTEGRATION-ONLY']);
const byLevel = {
  e2e: { tests: 0, criteria_covered: 0 },
  api: { tests: 0, criteria_covered: 0 },
  component: { tests: 0, criteria_covered: 0 },
  unit: { tests: 0, criteria_covered: 0 },
  other: { tests: 0, criteria_covered: 0 }, // captures tests with unrecognized or empty level
};

const normalizeTestStatus = (test) => {
  const explicitStatus = String(test.status || '')
    .trim()
    .toLowerCase();
  if (['skipped', 'pending', 'fixme'].includes(explicitStatus)) return explicitStatus;
  if (test.fixme === true) return 'fixme';
  if (test.pending === true) return 'pending';
  if (test.skipped === true) return 'skipped';
  return 'active';
};

const uniqueTests = new Map();
(traceabilityMatrix || []).forEach((req) => {
  (req.tests || []).forEach((test, index) => {
    // Do NOT use the per-requirement `index` as a fallback — the same test can appear
    // at different indices across requirements, producing spurious duplicate entries.
    // Use only stable, test-intrinsic fields; omit line when unavailable.
    const stableId =
      test.id ||
      [test.file, test.title || test.name, test.line].filter((value) => value !== undefined && value !== null && value !== '').join(':') ||
      null; // unresolvable — skip rather than manufacture a key

    if (stableId === null || uniqueTests.has(stableId)) return;
    const status = normalizeTestStatus(test);
    uniqueTests.set(stableId, {
      id: stableId,
      file: test.file || '',
      line: test.line ?? null,
      title: test.title || test.name || stableId,
      level: String(test.level || '')
        .trim()
        .toLowerCase(),
      status: status,
      skipped: status === 'skipped',
      fixme: status === 'fixme',
      pending: status === 'pending',
      blocker_reason: test.skip_reason || test.blocker_reason || test.fixme_reason || test.pending_reason || '',
    });
  });
});

[...uniqueTests.values()].forEach((test) => {
  const bucket = byLevel[test.level] ? test.level : 'other';
  if (bucket === 'other' && test.level) {
    console.warn(`[trace] unknown test level "${test.level}" for test "${test.id}" — counted in "other"`);
  }
  byLevel[bucket].tests += 1;
});

(traceabilityMatrix || []).forEach((req) => {
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
const deduplicatedTestInventory = {
  summary: {
    files: [...new Set(deduplicatedTests.map((test) => test.file).filter(Boolean))].length,
    cases: deduplicatedTests.length,
    skipped_cases: deduplicatedTests.filter((test) => test.skipped).length,
    fixme_cases: deduplicatedTests.filter((test) => test.fixme).length,
    pending_cases: deduplicatedTests.filter((test) => test.pending).length,
    by_level: byLevel,
  },
  tests: deduplicatedTests,
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

const extractedTargetId = runtime.getTraceTargetId?.() || null;
const extractedTargetLabel = runtime.getTraceTargetLabel?.() || null;
const traceTarget = {
  type: '{gate_type}',
  id: extractedTargetId, // story_id / epic_num / release_version / hotfix identifier from Step 1
  label: extractedTargetLabel || null,
};
```

---

### 5. Generate Complete Coverage Matrix

**Compile all Phase 1 outputs:**

```javascript
const coverageMatrix = {
  phase: 'PHASE_1_COMPLETE',
  generated_at: new Date().toISOString(),
  trace_target: traceTarget,
  collection_mode: '{collection_mode}',
  allow_gate: '{allow_gate}',
  coverage_basis: resolvedCoverageBasis,
  summary_confidence: resolvedOracleConfidence,
  oracle: {
    resolution_mode: oracleResolutionMode,
    confidence: resolvedOracleConfidence,
    sources: oracleSources,
    external_pointer_status: externalPointerStatus,
    synthetic: oracleResolutionMode === 'synthetic_source',
  },

  requirements: traceabilityMatrix, // Full matrix from Step 3

  coverage_statistics: {
    total_requirements: totalRequirements,
    fully_covered: fullyCovered,
    partially_covered: partialCoverage.length,
    uncovered: uncoveredRequirements.length,
    overall_coverage_percentage: coveragePercentage,

    priority_breakdown: {
      P0: { total: p0Total, covered: p0Covered, percentage: p0CoveragePercentage },
      P1: { total: p1Total, covered: p1Covered, percentage: p1CoveragePercentage },
      P2: { total: p2Total, covered: p2Covered, percentage: p2CoveragePercentage },
      P3: { total: p3Total, covered: p3Covered, percentage: p3CoveragePercentage },
    },
  },

  gap_analysis: {
    critical_gaps: criticalGaps,
    high_gaps: highGaps,
    medium_gaps: mediumGaps,
    low_gaps: lowGaps,
    partial_coverage_items: partialCoverage,
    unit_only_items: unitOnlyCoverage,
  },

  coverage_heuristics: {
    endpoint_gaps: endpointCoverageGaps,
    auth_negative_path_gaps: authCoverageGaps,
    happy_path_only_gaps: errorPathGaps,
    ui_journey_gaps: uiJourneyGaps,
    ui_state_gaps: uiStateGaps,
    counts: heuristicGapCounts,
  },

  test_inventory: deduplicatedTestInventory,
  blockers: deduplicatedTestInventory.blockers,
  recommendations: recommendations,
};
```

---

### 6. Output Coverage Matrix to Temp File

**Write to temp file for Phase 2:**

```javascript
const outputPath = '{tempOutputFile}';
fs.writeFileSync(outputPath, JSON.stringify(coverageMatrix, null, 2), 'utf8');

console.log(`✅ Phase 1 Complete: Coverage matrix saved to ${outputPath}`);
```

**Record the resolved path in the progress document** so Step 5 can read the exact same file rather than re-evaluating the timestamp expression:

After writing the temp file, update the YAML frontmatter in `{outputFile}` to include:

```yaml
tempCoverageMatrixPath: '<resolved outputPath>'
```

Step 5 reads `tempCoverageMatrixPath` from the frontmatter first; falls back to reconstructing `{tempOutputFile}` only when the key is absent.

---

### 7. Display Phase 1 Summary

```
✅ Phase 1 Complete: Coverage Matrix Generated

📊 Coverage Statistics:
- Total Requirements: {totalRequirements}
- Fully Covered: {fullyCovered} ({coveragePercentage}%)
- Partially Covered: {partialCoverage.length}
- Uncovered: {uncoveredRequirements.length}

🎯 Priority Coverage:
- P0: {p0Covered}/{p0Total} ({p0CoveragePercentage}%)
- P1: {p1Covered}/{p1Total} ({p1CoveragePercentage}%)
- P2: {p2Covered}/{p2Total} ({p2CoveragePercentage}%)
- P3: {p3Covered}/{p3Total} ({p3CoveragePercentage}%)

⚠️ Gaps Identified:
- Critical (P0): {criticalGaps.length}
- High (P1): {highGaps.length}
- Medium (P2): {mediumGaps.length}
- Low (P3): {lowGaps.length}

🔍 Coverage Heuristics:
- Endpoints without tests: {endpointCoverageGaps.length}
- Auth negative-path gaps: {authCoverageGaps.length}
- Happy-path-only criteria: {errorPathGaps.length}

📝 Recommendations: {recommendations.length}

🔄 Phase 2: Gate decision (next step)
```

### Orchestration Notes for This Step

When `resolvedMode` is `agent-team` or `subagent`, parallelize only dependency-safe sections:

- Worker A: gap classification (section 1)
- Worker B: heuristics gap extraction (section 2)
- Worker C: coverage statistics (section 4)

Section 3 (recommendation synthesis) depends on outputs from sections 1 and 2, so run it only after Workers A and B complete.

Section 5 remains the deterministic merge point after sections 1-4 are finished.

If `resolvedMode` is `sequential`, execute sections 1→7 in order.

---

## EXIT CONDITION

**PHASE 1 COMPLETE when:**

- ✅ Gap analysis complete
- ✅ Recommendations generated
- ✅ Coverage statistics calculated
- ✅ Coverage matrix saved to temp file
- ✅ Summary displayed

**Proceed to Phase 2 (Step 5: Gate Decision)**

---

### 8. Save Progress

**Save this step's accumulated work to `{outputFile}`.**

- **If `{outputFile}` does not exist** (first save), create it using the workflow template (if available) with YAML frontmatter:

  ```yaml
  ---
  stepsCompleted: ['step-04-analyze-gaps']
  lastStep: 'step-04-analyze-gaps'
  lastSaved: '{date}'
  ---
  ```

  Then write this step's output below the frontmatter.

- **If `{outputFile}` already exists**, update:
  - Add `'step-04-analyze-gaps'` to `stepsCompleted` array (only if not already present)
  - Set `lastStep: 'step-04-analyze-gaps'`
  - Set `lastSaved: '{date}'`
  - Append this step's output to the appropriate section of the document.

Load next step: `{nextStepFile}`

---

## 🚨 PHASE 1 SUCCESS METRICS

### ✅ SUCCESS:

- Coverage matrix complete and accurate
- All gaps identified and prioritized
- Recommendations actionable
- Temp file output valid JSON

### ❌ FAILURE:

- Coverage matrix incomplete
- Gap analysis missing
- Invalid JSON output

**Master Rule:** Phase 1 MUST output complete coverage matrix to temp file before Phase 2 can proceed.
