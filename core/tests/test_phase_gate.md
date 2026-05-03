# Test Documentation: PhaseGate Quality Gates

## Overview

This document provides test cases for the `PhaseGate` quality gate system. PhaseGates enforce quality checkpoints at different development phases (PLAN, IMPLEMENT, VERIFY) and support custom gate definitions with must-have requirements and context budgets.

---

## Test Case Table

| ID | Description | Input | Expected Output |
|----|-------------|-------|-----------------|
| PG-001 | Evaluate PLAN_GATE with all must-haves satisfied | PLAN_GATE, context with required plan artifacts present | Result(passed=True, blocked=False, details={}) |
| PG-002 | Evaluate PLAN_GATE with missing must-haves | PLAN_GATE, context missing required plan artifacts | Result(passed=False, blocked=True, missing_must_haves=[...]) |
| PG-003 | Evaluate IMPLEMENT_GATE with code complete | IMPLEMENT_GATE, context with all required code files | Result(passed=True, blocked=False) |
| PG-004 | Evaluate IMPLEMENT_GATE with incomplete code | IMPLEMENT_GATE, context missing required implementation | Result(passed=False, blocked=True, missing items listed) |
| PG-005 | Evaluate VERIFY_GATE with tests passing | VERIFY_GATE, context with passing test results | Result(passed=True, blocked=False) |
| PG-006 | Evaluate VERIFY_GATE with failing tests | VERIFY_GATE, context with failing test results | Result(passed=False, blocked=True, failing_tests=[...]) |
| PG-007 | Evaluate PLAN_GATE within context budget | PLAN_GATE, context with plan_size < context_budget | Result(passed=True, blocked=False) |
| PG-008 | Evaluate PLAN_GATE exceeding context budget | PLAN_GATE, context with plan_size > context_budget | Result(passed=False, blocked=True, reason="context_budget_exceeded") |
| PG-009 | Custom gate with single must_have | CustomGate(must_haves=["readme.md"]), context with readme.md | Result(passed=True) |
| PG-010 | Custom gate with multiple must_haves | CustomGate(must_haves=["readme.md", "tests/", "license"]), all present | Result(passed=True) |
| PG-011 | Custom gate with missing must_have | CustomGate(must_haves=["missing_file.txt"]), not present | Result(passed=False, blocked=True) |
| PG-012 | Custom gate with context_budget | CustomGate(context_budget=5000), context_size=4500 | Result(passed=True) |
| PG-013 | Custom gate exceeding context_budget | CustomGate(context_budget=5000), context_size=5500 | Result(passed=False, blocked=True) |
| PG-014 | PLAN_GATE with must_haves and context_budget both satisfied | PLAN_GATE with must_haves=["goals.md"], context_size=1000, budget=5000 | Result(passed=True) |
| PG-015 | PLAN_GATE with must_haves satisfied but context_budget exceeded | PLAN_GATE with must_haves=["goals.md"], context_size=6000, budget=5000 | Result(passed=False, blocked=True) |
| PG-016 | IMPLEMENT_GATE with must_haves satisfied but context_budget exceeded | IMPLEMENT_GATE with must_haves=["main.py"], context_size=50000, budget=10000 | Result(passed=False, blocked=True) |
| PG-017 | Gate evaluation with empty context | Any gate, empty context | Result(passed=False, blocked=True) |
| PG-018 | Gate evaluation with partial must_haves | CustomGate(must_haves=["a", "b", "c"]), context has ["a", "c"] | Result(passed=False, missing_must_haves=["b"]) |
| PG-019 | Multiple consecutive gates | Sequence: PLAN_GATE (pass) -> IMPLEMENT_GATE (pass) -> VERIFY_GATE (pass) | All pass |
| PG-020 | Gate blocked status propagation | VERIFY_GATE fails, blocked=True | Blocked status propagates to downstream gates |

---

## Test Execution Steps

### PG-001 & PG-002: PLAN_GATE Must-Haves
1. Initialize PLAN_GATE
2. For PG-001: Create context with all required plan artifacts
3. Call gate.evaluate(context)
4. Verify passed=True, blocked=False
5. For PG-002: Create context missing required artifacts
6. Verify passed=False, blocked=True, missing_must_haves populated

### PG-003 & PG-004: IMPLEMENT_GATE Must-Haves
1. Initialize IMPLEMENT_GATE
2. For PG-003: Provide context with required implementation files
3. Verify passed=True
4. For PG-004: Provide incomplete implementation context
5. Verify passed=False with missing items listed

### PG-005 & PG-006: VERIFY_GATE Must-Haves
1. Initialize VERIFY_GATE
2. For PG-005: Provide context with passing test results
3. Verify passed=True
4. For PG-006: Provide context with failing tests
5. Verify passed=False with failing_tests list

### PG-007 & PG-008: Context Budget - PLAN_GATE
1. Initialize PLAN_GATE with context_budget value
2. For PG-007: Create context with size < budget
3. Verify passed=True
4. For PG-008: Create context with size > budget
5. Verify passed=False with context_budget_exceeded reason

### PG-009 & PG-010 & PG-011: Custom Gate Must-Haves
1. Create CustomGate with specified must_haves list
2. For PG-009: Provide context satisfying single must_have
3. For PG-010: Provide context satisfying all multiple must_haves
4. For PG-011: Provide context missing must_have
5. Verify results match expectations

### PG-012 & PG-013: Custom Gate Context Budget
1. Create CustomGate with context_budget
2. For PG-012: Context size within budget
3. For PG-013: Context size exceeds budget
4. Verify results match expectations

### PG-014 & PG-015: Combined Must-Haves and Context Budget
1. Initialize PLAN_GATE with both must_haves and context_budget
2. For PG-014: Context satisfies both
3. For PG-015: Context satisfies must_haves but exceeds budget
4. Verify only budget failure in PG-015

### PG-016: Combined Failure - IMPLEMENT_GATE
1. Initialize IMPLEMENT_GATE with must_haves and context_budget
2. Provide context satisfying must_haves but exceeding budget
3. Verify passed=False due to context_budget_exceeded

### PG-017: Empty Context
1. Create any gate
2. Provide empty context {}
3. Verify passed=False, blocked=True

### PG-018: Partial Must-Haves
1. Create CustomGate with must_haves=["a", "b", "c"]
2. Provide context with ["a", "c"]
3. Verify passed=False
4. Verify missing_must_haves=["b"]

### PG-019: Consecutive Gates
1. Evaluate PLAN_GATE, verify pass
2. Evaluate IMPLEMENT_GATE with passing context, verify pass
3. Evaluate VERIFY_GATE with passing context, verify pass

### PG-020: Blocked Status Propagation
1. Set up VERIFY_GATE with failing context
2. Verify blocked=True in result
3. Verify downstream gates recognize blocked status

---

## Pass/Fail Criteria

- **All tests pass** if:
  - Each gate correctly evaluates must_haves requirements
  - Each gate correctly enforces context_budget limits
  - Custom gates function identically to predefined gates
  - Missing must_haves are accurately reported in results
  - Context budget violations are clearly identified
  - Blocked status propagates correctly to downstream gates
  - Empty context is handled gracefully with clear failure reason

- **Test fails** if:
  - Gate passes when must_haves are missing
  - Gate passes when context_budget is exceeded
  - Missing must_haves list is incomplete or incorrect
  - Budget exceeded reason is missing or unclear
  - Custom gate behavior differs from predefined gates
  - Blocked status fails to propagate
  - Empty context causes exception instead of graceful failure
