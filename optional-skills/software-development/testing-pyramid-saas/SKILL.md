---
name: testing-pyramid-saas
description: "Testing pyramid, coverage gates, and mutation testing."
version: 1.0.0
author: Rafael Zendron (rafaumeu)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [testing, coverage, mutation-testing, e2e, integration, unit, saas]
    related_skills: [test-driven-development, requesting-code-review]
    requires_toolsets: [terminal]
---

# Testing Pyramid for SaaS

A structured testing strategy for SaaS applications. Defines the ratio of test types, coverage requirements, and mutation testing thresholds for production software.

**What it does:** Enforces a testing pyramid with coverage gates and mutation testing.
**What it doesn't do:** Replace TDD. The pyramid defines strategy; TDD defines the mechanism.

## When to Use

- Setting up testing for a new SaaS project
- Establishing CI quality gates
- When coverage is below target and needs systematic improvement
- Before launching features that handle money or user data

## Prerequisites

- A test framework configured (Vitest, pytest, etc.)
- CI pipeline configured (GitHub Actions, etc.)
- Coverage tool configured (v8 recommended for JS/TS, coverage.py for Python)

## Procedure

### Step 1: Define your pyramid ratio

## The Testing Pyramid

```
           /\
          /E2\       5% — critical flows (payments, auth)
         /----\
        /Integ\      15% — payment+DB, core+state
       /--------\
      /Contract \    10% — API response shape, webhook payloads
     /------------\
    /Property-based\  5% — input sanitization, parsers, rate limits
   /----------------\
  /   Unit + Mutation \  65% — pure logic, proven by mutation testing
 /--------------------\
```

## Rules by Test Type

### Unit (mandatory, 65%)

- TDD RED > GREEN > REFACTOR for every function
- 100% coverage per file
- Mock ONLY external dependencies (DB, network, filesystem)
- Each branch = 1 test
- No testing implementation details — test behavior and results

### Integration (mandatory, 15%)

- Required for: billing, payments, any feature involving DB transactions
- Use real DB or a mock that faithfully simulates the target (Postgres, etc.)
- Test transactions: success, failure, rollback
- Never mock the DB in integration tests — use a test database

### E2E (mandatory, 5%)

- Required for: money-handling flows, authentication flows
- Framework: Playwright, Cypress, or equivalent
- Scenarios: login > core flow > completion, pricing > checkout > webhook
- Clean DB state between tests
- Run in CI, not pre-push (too slow for local gates)

### Contract (mandatory, 10%)

- Shared Zod schema (or equivalent) between API and frontend
- API validates output shape; frontend validates input shape
- Contracts live in a shared location (e.g., `src/lib/contracts/`)
- Breakage = blocked deploy

### Property-based (required for external input, 5%)

- Framework: fast-check (JS/TS), hypothesis (Python)
- Apply to: input sanitization, rate limit logic, parsers, formatters
- Generate random inputs to find edge cases humans miss

### Mutation (mandatory after unit)

- Framework: Stryker
- Score >= 80% for general code, >= 90% for money-handling code
- Equivalent mutants: exclude via line ranges in config with documented reason
- Surviving mutants = false-positive tests

### Regression (mandatory)

- Every bug fix = new test tagged with the bug context
- Never delete regression tests
- Tag with `@regression` for traceability

## Coverage Requirements

### Coverage Provider

Use the native V8 provider (not Istanbul). Istanbul inflates coverage by over-counting async routes, error paths, and conditional branches. V8's source-map quirks are manageable; Istanbul hiding untested code is worse.

### Targets

For SaaS handling money or personal data:
- Statements: 100%
- Branches: 100%
- Functions: 100%
- Lines: 100%

**Achievable with V8** after:
1. Dead code removal (unreachable branches after schema parse)
2. Targeted tests for error paths
3. Honest exclusions for V8 false negatives (documented with reason)

### Coverage Strategy (order matters)

1. **Remove dead code first** — methods never called, modules never imported, type-only exports
2. **Exclude type-only files** — files with 0% statements but no executable code
3. **Write targeted tests** — only for uncovered branches on live code paths
4. **Document honest exclusions** — every `coverage.exclude` entry needs a comment explaining why

**Anti-pattern:** Adding source files to `coverage.exclude` to boost numbers. The answer is always "write more tests", never "exclude more files".

### Realistic Starting Thresholds

For existing codebases not yet at 100%:
- Statements: 94%
- Branches: 88%
- Functions: 95%
- Lines: 94%

Increase 2-3% per sprint until reaching 100%.

## CI Test Pipeline per PR

```
Unit (coverage, 100% threshold)
  > Integration
    > Mutation (>= 80%)
      > E2E
        > Security audit
```

Each stage blocks the next. No bypassing.

## Pre-Implementation Checklist (blocking)

- [ ] Read the spec completely
- [ ] Read existing code that will be modified
- [ ] Map ALL branches (if/else/switch/ternary)
- [ ] Identify required mocks (DB, API, payment)
- [ ] Write test FIRST (RED) — watched it fail for the right reason
- [ ] Implement minimal code (GREEN)
- [ ] Refactor if needed
- [ ] All existing tests pass, coverage = 100%, validate passes

## Key Rules

1. 100% coverage is mandatory for SaaS with money/personal data. Not aspirational — rule.
2. Each bug fix = regression test. Never delete.
3. Pre-implementation checklist is blocking. TDD is mechanism, not guideline.
4. Never merge with failing tests. "Pre-existing" failures get fixed or removed immediately.
5. V8 coverage provider over Istanbul. Always.

## Pitfalls

- **Adding files to coverage.exclude to boost numbers.** The answer is always "write more tests", never "exclude more files".
- **Testing implementation details.** Test behavior and outputs, not which internal methods are called.
- **100% statement coverage with 0% branch coverage.** Each branch (if/else/switch/ternary) needs its own test.
- **Mocking the database in integration tests.** Use a real test database. Mocked DB hides real query failures.
- **Running E2E tests pre-push.** Too slow. Run in CI only. Pre-push = unit + coverage.
- **Ignoring mutation testing survivors.** Surviving mutants = tests that pass without catching real bugs.
- **Deleting regression tests.** Every regression test exists because a real bug shipped. Never remove.

## Verification

```bash
# Run coverage with thresholds
npx vitest run --coverage

# Check mutation score (if configured)
npx stryker run

# Verify no secrets in test files
search_files("password|secret|token", path="tests/", target="content")
```
