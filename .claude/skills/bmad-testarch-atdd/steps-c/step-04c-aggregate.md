---
name: 'step-04c-aggregate'
description: 'Aggregate subagent outputs and complete ATDD test infrastructure'
outputFile: '{test_artifacts}/atdd-checklist-{story_key}.md'
nextStepFile: '{skill-root}/steps-c/step-05-validate-and-complete.md'
---

# Step 4C: Aggregate ATDD Test Generation Results

## STEP GOAL

Read outputs from parallel subagents (API + E2E red-phase test generation), aggregate results, verify TDD red phase compliance, and create supporting infrastructure.

---

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`
- ✅ Read subagent outputs from temp files
- ✅ Verify all tests are marked with test.skip() (TDD red phase)
- ✅ Generate shared fixtures based on fixture needs
- ✅ Write all generated test files to disk
- ❌ Do NOT remove test.skip() (that's done after feature implementation)
- ❌ Do NOT run tests yet (that's step 5 - verify scaffolds and optional RED activation)

---

## EXECUTION PROTOCOLS:

- 🎯 Follow the MANDATORY SEQUENCE exactly
- 💾 Record outputs before proceeding
- 📖 Load the next step only when instructed

## CONTEXT BOUNDARIES:

- Available context: config, subagent outputs from temp files
- Focus: aggregation and TDD validation
- Limits: do not execute future steps
- Dependencies: Step 4A and 4B subagent outputs

---

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

### 1. Read Subagent Outputs

**Read API test subagent output:**

```javascript
const apiTestsPath = '/tmp/tea-atdd-api-tests-{{timestamp}}.json';
const apiTestsOutput = JSON.parse(fs.readFileSync(apiTestsPath, 'utf8'));
```

**Read E2E test subagent output:**

```javascript
const e2eTestsPath = '/tmp/tea-atdd-e2e-tests-{{timestamp}}.json';
const e2eTestsOutput = JSON.parse(fs.readFileSync(e2eTestsPath, 'utf8'));
```

**Verify both subagents succeeded:**

- Check `apiTestsOutput.success === true`
- Check `e2eTestsOutput.success === true`
- If either failed, report error and stop (don't proceed)

---

### 2. Verify TDD Red Phase Compliance

**CRITICAL TDD Validation:**

**Check API tests:**

```javascript
apiTestsOutput.tests.forEach((test) => {
  // Verify test.skip() is present
  if (!test.content.includes('test.skip(')) {
    throw new Error(`ATDD ERROR: ${test.file} missing test.skip() - tests MUST be skipped in red phase!`);
  }

  // Verify not placeholder assertions
  if (test.content.includes('expect(true).toBe(true)')) {
    throw new Error(`ATDD ERROR: ${test.file} has placeholder assertions - must assert EXPECTED behavior!`);
  }

  // Verify expected_to_fail flag
  if (!test.expected_to_fail) {
    throw new Error(`ATDD ERROR: ${test.file} not marked as expected_to_fail!`);
  }
});
```

**Check E2E tests:**

```javascript
e2eTestsOutput.tests.forEach((test) => {
  // Same validation as API tests
  if (!test.content.includes('test.skip(')) {
    throw new Error(`ATDD ERROR: ${test.file} missing test.skip() - tests MUST be skipped in red phase!`);
  }

  if (test.content.includes('expect(true).toBe(true)')) {
    throw new Error(`ATDD ERROR: ${test.file} has placeholder assertions!`);
  }

  if (!test.expected_to_fail) {
    throw new Error(`ATDD ERROR: ${test.file} not marked as expected_to_fail!`);
  }
});
```

**If validation passes:**

```
✅ TDD Red Phase Validation: PASS
- All tests use test.skip()
- All tests assert expected behavior (not placeholders)
- All tests marked as expected_to_fail
```

---

### 3. Write All Test Files to Disk

**Write API test files:**

```javascript
apiTestsOutput.tests.forEach((test) => {
  fs.writeFileSync(test.file, test.content, 'utf8');
  console.log(`✅ Created (RED): ${test.file}`);
});
```

**Write E2E test files:**

```javascript
e2eTestsOutput.tests.forEach((test) => {
  fs.writeFileSync(test.file, test.content, 'utf8');
  console.log(`✅ Created (RED): ${test.file}`);
});
```

---

### 4. Aggregate Fixture Needs

**Collect all fixture needs from both subagents:**

```javascript
const allFixtureNeeds = [...apiTestsOutput.fixture_needs, ...e2eTestsOutput.fixture_needs];

// Remove duplicates
const uniqueFixtures = [...new Set(allFixtureNeeds)];
```

---

### 5. Generate Fixture Infrastructure

**Create fixtures needed by ATDD tests:**
(Similar to automate workflow, but may be simpler for ATDD since feature not implemented)

**Minimal fixtures for TDD red phase:**

```typescript
// tests/fixtures/test-data.ts
export const testUserData = {
  email: 'test@example.com',
  password: 'SecurePass123!',
};
```

Note: More complete fixtures will be needed when moving to green phase.

---

### 6. Generate ATDD Checklist

**Create ATDD checklist document:**

```markdown
# ATDD Checklist: [Story Name]

## TDD Red Phase (Current)

✅ Red-phase test scaffolds generated

- API Tests: {api_test_count} tests (all skipped)
- E2E Tests: {e2e_test_count} tests (all skipped)

## Acceptance Criteria Coverage

{list all acceptance criteria with test coverage}

## Next Steps (Task-by-Task Activation)

During implementation of each task:

1. Remove `test.skip()` from the current test file or scenario
2. Run tests: `npm test`
3. Verify the activated test fails first, then passes after implementation (green phase)
4. If any activated tests still fail unexpectedly:
   - Either fix implementation (feature bug)
   - Or fix test (test bug)
5. Commit passing tests

## Implementation Guidance

Feature endpoints to implement:
{list endpoints from API tests}

UI components to implement:
{list UI flows from E2E tests}
```

**Save checklist:**

```javascript
fs.writeFileSync(`{test_artifacts}/atdd-checklist-{story_key}.md`, checklistContent, 'utf8');
```

**If `{story_file}` exists and is writable, attempt to link artifacts back into the story:**

- Add or update a `### ATDD Artifacts` subsection under `## Dev Notes`
- Record:
  - `Checklist: {test_artifacts}/atdd-checklist-{story_key}.md`
  - `API tests: {api_test_file_path}` when present
  - `E2E tests: {e2e_test_file_path}` when present
  - `Component tests: {component_test_file_path}` when present
- Preserve all other story content verbatim
- The checklist template already contains the manual handoff instructions if story linking is not possible
- If the story file cannot be updated safely, continue without failing the workflow and keep the checklist's manual handoff instructions intact

---

### 7. Calculate Summary Statistics

**Aggregate test counts:**

```javascript
const resolvedMode = subagentContext?.execution?.resolvedMode; // Provided by Step 4's orchestration context
const subagentExecutionLabel =
  resolvedMode === 'sequential'
    ? 'SEQUENTIAL (API → E2E)'
    : resolvedMode === 'agent-team'
      ? 'AGENT-TEAM (API + E2E)'
      : resolvedMode === 'subagent'
        ? 'SUBAGENT (API + E2E)'
        : 'PARALLEL (API + E2E)';
const performanceGainLabel =
  resolvedMode === 'sequential'
    ? 'baseline (no parallel speedup)'
    : resolvedMode === 'agent-team' || resolvedMode === 'subagent'
      ? '~50% faster than sequential'
      : 'mode-dependent';

const summary = {
  tdd_phase: 'RED',
  total_tests: apiTestsOutput.test_count + e2eTestsOutput.test_count,
  api_tests: apiTestsOutput.test_count,
  e2e_tests: e2eTestsOutput.test_count,
  all_tests_skipped: true,
  expected_to_fail: true,
  fixtures_created: uniqueFixtures.length,
  acceptance_criteria_covered: [
    ...apiTestsOutput.tests.flatMap((t) => t.acceptance_criteria_covered),
    ...e2eTestsOutput.tests.flatMap((t) => t.acceptance_criteria_covered),
  ],
  knowledge_fragments_used: [...apiTestsOutput.knowledge_fragments_used, ...e2eTestsOutput.knowledge_fragments_used],
  subagent_execution: subagentExecutionLabel,
  performance_gain: performanceGainLabel,
};
```

**Store summary for Step 5:**

```javascript
fs.writeFileSync('/tmp/tea-atdd-summary-{{timestamp}}.json', JSON.stringify(summary, null, 2), 'utf8');
```

---

## OUTPUT SUMMARY

Display to user:

```
✅ ATDD Test Generation Complete (TDD RED PHASE)

🔴 TDD Red Phase: Test Scaffolds Generated

📊 Summary:
- Total Tests: {total_tests} (all with test.skip())
  - API Tests: {api_tests} (RED)
  - E2E Tests: {e2e_tests} (RED)
- Fixtures Created: {fixtures_created}
- Activated tests will FAIL until feature is implemented

✅ Acceptance Criteria Coverage:
{list all covered criteria}

🚀 Performance: {performance_gain}

📂 Generated Files:
- tests/api/[feature].spec.ts (with test.skip())
- tests/e2e/[feature].spec.ts (with test.skip())
- tests/fixtures/test-data.ts
- {test_artifacts}/atdd-checklist-{story_key}.md

📝 Next Steps:
1. Link ATDD artifacts into the story file if available
2. Implement the feature
3. Remove test.skip() from the tests for the current task
4. Run activated tests → verify they FAIL before implementation, then PASS after implementation
5. Commit passing tests

✅ Ready for validation (Step 5 - verify red-phase scaffolds and handoff metadata)
```

---

## EXIT CONDITION

Proceed to Step 5 when:

- ✅ All test files written to disk (API + E2E)
- ✅ All tests verified to have test.skip()
- ✅ All fixtures created
- ✅ ATDD checklist generated
- ✅ Summary statistics calculated and saved
- ✅ Output displayed to user

---

### 8. Save Progress

**Save this step's accumulated work to `{outputFile}`.**

- **If `{outputFile}` does not exist** (first save), create it with YAML frontmatter:

  ```yaml
  ---
  stepsCompleted: ['step-04c-aggregate']
  lastStep: 'step-04c-aggregate'
  lastSaved: '{date}'
  storyId: '{story_id}'
  storyKey: '{story_key}'
  storyFile: '{story_file}'
  atddChecklistPath: '{outputFile}'
  generatedTestFiles: []
  ---
  ```

  Then write this step's output below the frontmatter.

- **If `{outputFile}` already exists**, update:
  - Add `'step-04c-aggregate'` to `stepsCompleted` array (only if not already present)
  - Set `lastStep: 'step-04c-aggregate'`
  - Set `lastSaved: '{date}'`
  - Set `storyId` to `{story_id}`
  - Set `storyKey` to `{story_key}`
  - Set `storyFile` to `{story_file}`
  - Set `atddChecklistPath` to `{outputFile}`
  - Set `generatedTestFiles` deterministically to the list of present test paths in this order: API, E2E, Component (omit blanks / N/A values)
  - Append this step's output to the appropriate section.

Load next step: `{nextStepFile}`

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS:

### ✅ SUCCESS:

- Both subagents succeeded
- All tests have test.skip() (TDD red phase compliant)
- All tests assert expected behavior (not placeholders)
- All test files written to disk
- ATDD checklist generated
- Story metadata and handoff paths captured in checklist frontmatter

### ❌ SYSTEM FAILURE:

- One or both subagents failed
- Tests missing test.skip() (would break CI)
- Tests have placeholder assertions
- Test files not written to disk
- ATDD checklist missing

**Master Rule:** TDD RED PHASE requires ALL tests to use test.skip() and assert expected behavior.
