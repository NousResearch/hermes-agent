3-LEG VERIFICATION LOOP for:

$ARGUMENTS

Run all three independent verification checks against the specified fix, file, or recent changes. Each leg catches a different class of error. Do NOT skip any leg.

═══════════════════════════════════════════
LEG 1 — STRUCTURE (type/build check)
═══════════════════════════════════════════

Run your project's type checker or build:
- TypeScript: npx tsc --noEmit
- Other: {your build/type command from CLAUDE.md → Commands}

If errors:
- List every error with file:line
- Classify each: type mismatch | missing field | broken import | wrong interface | other
- Fix each error
- Re-run until 0 errors

Report: "Build: {N} errors found → {N} fixed → CLEAN" or "Build: CLEAN (0 errors)"

═══════════════════════════════════════════
LEG 2 — FACTS (grep verification)
═══════════════════════════════════════════

Identify the key claims about what the fix does. For each claim, grep to verify.

Common claims to verify:
- "This function is called from N places" → grep -rn '{function}(' {source_dir}/ --include='*.ts'
- "This field is always populated" → check every write site
- "This file is on the LIVE path" → trace import chain to entry point
- "No catch block returns the success state" → grep -rn 'catch' {file} -A 10 | grep {success_pattern}
- "This value is never null" → grep for all assignment sites, check each

For each claim:
[CONFIRMED] {claim} — Evidence: {file:line}
[REFUTED] {claim} — Actual: {what grep found} at {file:line}
[INCONCLUSIVE] {claim} — Need: {what additional check would resolve}

If ANY claim is REFUTED: STOP. The fix has a problem. Report it.

Report: "Facts: {N} claims verified, {N} refuted, {N} inconclusive"

═══════════════════════════════════════════
LEG 3 — BEHAVIOR (test suite)
═══════════════════════════════════════════

Run tests relevant to the changed files.

Step 1 — Find relevant tests:
grep -rn '{changed_file_name}' tests/ --include='*.ts' -l
find tests/ -name '*{module_name}*' -type f

Step 2 — Run them:
{your test command} --testPathPattern='{test_pattern}'

Step 3 — If no relevant tests exist:
Report: "No existing tests cover this change."
Recommend: "Run /project:tdd to create tests from the fix spec."

Step 4 — If tests exist and run:
- All pass → Report: "Tests: {N}/{N} passing"
- Some fail → List each failure. Diagnose: is the test wrong or is the fix wrong?
- Test errors → Fix the test infrastructure issue first

Report: "Tests: {N} passed, {N} failed, {N} errored" or "Tests: no coverage for this change"

═══════════════════════════════════════════
COMBINED VERDICT
═══════════════════════════════════════════

| Leg | Status | Details |
|-----|--------|---------|
| 1 — Structure | ✅ PASS / ❌ FAIL | Build: {result} |
| 2 — Facts | ✅ PASS / ❌ FAIL | {N} confirmed, {N} refuted |
| 3 — Behavior | ✅ PASS / ❌ FAIL / ⚠️ NO COVERAGE | {test results} |

OVERALL: ✅ VERIFIED / ❌ FAILED / ⚠️ PARTIAL (no test coverage)

If FAILED: list every failure and recommend next steps.
If PARTIAL: the fix compiles and fact-checks pass, but behavioral verification is missing. Recommend /project:tdd.
If VERIFIED: all three legs pass.

RULES:
- Do NOT skip Leg 2 just because Leg 1 passed. Build checks catch type errors, not logic errors.
- Do NOT skip Leg 3 just because Leg 2 passed. Grep confirms code state, not runtime behavior.
- Do NOT report VERIFIED if any leg is FAIL.
- PARTIAL is acceptable for quick fixes but NOT for boundary-crossing or multi-file changes.
