# /project:tdd — Test-Driven Fix Workflow

You are given a spec, bug report, finding ID, or fix prompt. Your job is to write the FAILING TEST FIRST, then implement.

## Phase 1 — Read the Spec
1. Read the provided spec, finding, or fix prompt completely.
2. Identify the INVARIANTS — what must be true after the fix.
3. Identify the FAILURE MODE — what is currently broken.

## Phase 2 — Write Failing Tests
4. Create test file(s) in the appropriate test directory (see CLAUDE.md → Commands for your test setup).
   - Naming: `{target-file}.{spec-id}.test.ts` (or project convention)
5. Write tests that assert the CORRECT behavior (per the spec).
6. Include at minimum:
   - **Happy path test:** The fix works as specified.
   - **Error path test:** Error/catch/undefined states produce RESTRICTIVE defaults.
   - **Boundary test:** Edge cases at pipeline boundaries (empty input, null fields, type mismatches).
7. Run the tests.
8. **STOP.** Tests must FAIL. If they pass, either:
   - The bug doesn't exist (possible false positive — report to the operator), OR
   - The test is wrong (fix the test, not the code).
9. Report: "Tests written. X tests failing. Ready to implement." Then WAIT for the operator to say "implement."

## Phase 3 — Implement (only after operator confirms)
10. Implement the fix per the spec.
11. Run tests again.
12. All tests must PASS.
13. Run your type/build checker.
14. Run `/grep-verify` on the key claims from the spec.

## Phase 4 — Verify
15. Report: which tests pass, build status, grep evidence for key claims.
16. If any test still fails, diagnose and fix. Do not skip failing tests.

## Rules
- DO NOT write implementation code before tests exist and fail.
- DO NOT mark a test as "skipped" or "todo" — every test must execute.
- DO NOT import from or edit known dead code paths (see CLAUDE.md → DO NOT Edit).
- Error path tests MUST assert that failure states produce the restrictive/safe outcome.

## When NOT to use this command
- Single-line defensive hardening (just fix it and `/grep-verify`).
- Comment-only changes or documentation updates.
- Use TDD when the fix touches 2+ files or crosses a pipeline boundary.
