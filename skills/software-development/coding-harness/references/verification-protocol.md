# Verification Protocol

The defining rule of this harness: **an increment is only done when a signal the agent did not
author confirms it.** This is the agent-facing version of the agentic-harness-engineering
`adb ask` step, where a separate verifier cross-references the agent's internal trace against
the *external* test output to catch "the agent believed it succeeded when the verifier
disagreed" (`evolve.py` `DEFAULT_DEBUG_QUERY_K1`).

## What counts as proof, by task type

| Task type | Acceptable proof | Not proof |
|-----------|------------------|-----------|
| Bug fix | A test that failed before and passes now | "I can see the bug is fixed" |
| New feature | New tests pass + existing suite still green | The code reads correctly |
| Refactor | Full existing suite green, behavior unchanged | "Pure refactor, can't break" |
| Build/config | `build` / `compile` exits 0 on a clean tree | It built last time |
| Performance | A measurement before vs. after | "This should be faster" |
| CLI/UI behavior | Actually running it and observing output | The handler looks right |
| Lint/types | Tool reports clean or no-new-findings vs. baseline | Eyeballing the diff |

## The belief-vs-reality cross-check

After every VERIFY, compare three things:
1. What you **predicted** would happen (from `add-increment`).
2. What you **believe** happened (your read of the change).
3. What the **external signal actually reported**.

When (3) disagrees with (1) or (2), that gap is the highest-value information in the loop:
- Predicted pass, got fail → your model of the code is wrong. Root-cause before continuing.
- Predicted fail/regression, got pass → either the risk wasn't real or the check didn't
  exercise it. Confirm the check actually covers the at-risk path before trusting the pass.
- "Looks done" but the suite is red → classic self-report failure. The suite wins, always.

Record the actual command and its outcome in the verification note. Future-you (post-compaction)
trusts the recorded external result, not a vague "passed".

## Baseline awareness

Before declaring a regression, know the baseline. If a test was already failing before your
change, your increment didn't break it — capture the pre-change suite state once at SCOPE (or
before the first risky increment) so VERIFY can distinguish *new* failures from pre-existing
ones. Don't chase red that you didn't cause (but do note it).

## Independent verifier subagent (high-stakes increments)

For changes that are risky, security-relevant, or hard to test directly, do not rely on your
own verification — **no agent should verify its own work** (the core principle of
`requesting-code-review`). Spawn a fresh-context reviewer subagent and give it:

- The diff / changed files.
- The increment's stated goal and prediction.
- The done-definition from state.
- An instruction to run the build/tests itself and report pass/fail with evidence, and to
  look specifically for the at-risk regression you recorded.

Fresh context finds what the implementing context misses. Treat a reviewer "fail" as a real
VERIFY fail → ROOT-CAUSE.

## Final verification

The per-increment checks can each pass while the whole is broken (integration drift). FINAL
VERIFY runs the full suite + an end-to-end exercise of the feature. Only a green FINAL VERIFY
closes the task.
