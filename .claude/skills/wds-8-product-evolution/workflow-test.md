---
name: acceptance-test
description: Test the implementation against the specification
borrows_from: Phase 5 [T] (acceptance testing)
---

# Acceptance Test

**Goal:** Validate the implementation against the specification's acceptance criteria before deploying.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Load Test Context

Gather everything needed for testing:

1. Read specification from [D] Design Solution
2. Read scenario from [S] Scope Improvement
3. Review implementation diff from [I] Implement
4. Extract acceptance criteria into a test checklist

### Step 2: Prepare Test Environment

Ensure the implementation is running and testable:

1. Confirm branch is checked out: `evolution/[scenario-name]`
2. Start local development server if needed
3. Navigate to the affected page/view
4. Note the URL and any required test data

### Step 3: Execute Tests

For each acceptance criterion:

| # | Criterion | Steps | Expected | Actual | Pass? |
|---|-----------|-------|----------|--------|-------|
| 1 | [From spec] | [How to test] | [Expected result] | [What happened] | Y/N |
| 2 | ... | ... | ... | ... | ... |

Also test:
- **Responsive**: Check all breakpoints defined in spec
- **Edge cases**: Empty states, long content, error states
- **Regression**: Verify nothing else broke on the page
- **Cross-browser**: If specified in project requirements

### Step 4: Document Results

Create test report at `{output_folder}/evolution/test-reports/`:

```markdown
# Test Report: [Scenario Name]

## Summary
[X/Y criteria passed]

## Results
[Test table from Step 3]

## Issues Found
[List any failures with severity and description]

## Recommendation
[Pass / Pass with notes / Fail — needs rework]
```

### Step 5: Handle Failures

If tests fail:

- **Minor issues** → Fix in the same branch, retest
- **Design issues** → Route back to [D] Design Solution
- **Scope creep** → Log as separate improvement target for next cycle

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action
3. Return to activity menu
