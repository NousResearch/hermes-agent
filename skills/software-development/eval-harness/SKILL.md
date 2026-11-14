---
name: eval-harness
description: 'Formal evaluation framework implementing eval-driven development (EDD). Use when setting up evals, defining pass/fail criteria, measuring agent reliability with pass@k metrics, or creating regression test suites.'
version: 1.0.0
author: Hermes Agent (adapted from ECC)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [evaluation, testing, quality, metrics, regression, EDD]
    related_skills: [verification-loop, test-driven-development, agent-self-evaluation]
---

# Eval Harness

A formal evaluation framework for Hermes Agent sessions, implementing eval-driven development (EDD) principles.

## When to Activate

- Setting up eval-driven development (EDD) for AI-assisted workflows
- Defining pass/fail criteria for task completion
- Measuring agent reliability with pass@k metrics
- Creating regression test suites for prompt or agent changes
- Benchmarking agent performance across model versions

## Philosophy

Eval-Driven Development treats evals as the "unit tests of AI development":

- Define expected behavior BEFORE implementation
- Run evals continuously during development
- Track regressions with each change
- Use pass@k metrics for reliability measurement

## Eval Types

### Capability Evals

Test if the agent can do something it couldn't before:

```markdown
[CAPABILITY EVAL: feature-name]
Task: Description of what the agent should accomplish
Success Criteria:

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3
      Expected Output: Description of expected result
```

### Regression Evals

Ensure changes don't break existing functionality:

```markdown
[REGRESSION EVAL: feature-name]
Baseline: SHA or checkpoint name
Tests:

- existing-test-1: PASS/FAIL
- existing-test-2: PASS/FAIL
- existing-test-3: PASS/FAIL
  Result: X/Y passed (previously Y/Y)
```

## Grader Types

### 1. Code-Based Grader

Deterministic checks using code:

```bash
# Check if file contains expected pattern
grep -q "export function handleAuth" src/auth.ts && echo "PASS" || echo "FAIL"

# Check if tests pass
npm test -- --testPathPattern="auth" && echo "PASS" || echo "FAIL"

# Check if build succeeds
npm run build && echo "PASS" || echo "FAIL"
```

### 2. Model-Based Grader

Use the agent to evaluate open-ended outputs:

```markdown
[MODEL GRADER PROMPT]
Evaluate the following code change:

1. Does it solve the stated problem?
2. Is it well-structured?
3. Are edge cases handled?
4. Is error handling appropriate?

Score: 1-5 (1=poor, 5=excellent)
Reasoning: [explanation]
```

### 3. Human Grader

Flag for manual review:

```markdown
[HUMAN REVIEW REQUIRED]
Change: Description of what changed
Reason: Why human review is needed
Risk Level: LOW/MEDIUM/HIGH
```

## Metrics

### pass@k

"At least one success in k attempts"

- pass@1: First attempt success rate
- pass@3: Success within 3 attempts
- Typical target: pass@3 > 90%

### pass^k

"All k trials succeed"

- Higher bar for reliability
- pass^3: 3 consecutive successes
- Use for critical paths

## Eval Workflow

### 1. Define (Before Coding)

```markdown
## EVAL DEFINITION: feature-xyz

### Capability Evals

1. Can create new user account
2. Can validate email format
3. Can hash password securely

### Regression Evals

1. Existing login still works
2. Session management unchanged
3. Logout flow intact

### Success Metrics

- pass@3 > 90% for capability evals
- pass^3 = 100% for regression evals
```

### 2. Implement

Write code to pass the defined evals.

### 3. Evaluate

Run capability and regression evals, record PASS/FAIL.

### 4. Report

```markdown
# EVAL REPORT: feature-xyz

Capability Evals:
create-user: PASS (pass@1)
validate-email: PASS (pass@2)
hash-password: PASS (pass@1)
Overall: 3/3 passed

Regression Evals:
login-flow: PASS
session-mgmt: PASS
logout-flow: PASS
Overall: 3/3 passed

Metrics:
pass@1: 67% (2/3)
pass@3: 100% (3/3)

Status: READY FOR REVIEW
```

## Eval Storage

Store evals in project:

```
evals/
  feature-xyz.md      # Eval definition
  feature-xyz.log     # Eval run history
  baseline.json       # Regression baselines
```

## Hermes Integration

- Use `write_file` to create eval definitions
- Use `terminal` for code-based graders
- Use `delegate_task` to run evals in parallel
- Use `read_file` to check eval results
- Combine with `verification-loop` for comprehensive quality gates
- Combine with `agent-self-evaluation` for self-assessment

## Best Practices

1. **Define evals BEFORE coding** — Forces clear thinking about success criteria
2. **Run evals frequently** — Catch regressions early
3. **Keep evals versioned** — Store alongside code in git
4. **Start with code-based graders** — Deterministic before model-based
5. **Use pass@3 as the default reliability target**
