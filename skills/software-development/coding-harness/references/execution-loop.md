# Execution Loop — full specification

The loop is a small state machine. You are always in exactly one phase and you only leave it
when its **exit criterion** is met. Adapted from the agentic-harness-engineering iteration loop
(`evolve.py`), which runs snapshot → evaluate → analyze → root-cause → attribute → evolve →
verify/commit each iteration. Here the same shape drives a single coding task.

```
        ┌─────────┐
        │  SCOPE  │  goal + done-definition
        └────┬────┘
             v
        ┌─────────┐
        │  PLAN   │  ordered, independently-verifiable increments
        └────┬────┘
             v
   ┌──> ┌───────────┐
   │    │ IMPLEMENT │  smallest change for next increment + prediction
   │    └────┬──────┘
   │         v
   │    ┌─────────┐
   │    │ VERIFY  │  run the EXTERNAL check
   │    └────┬────┘
   │         v
   │    ┌───────────┐      fail     ┌────────────┐
   │    │ ATTRIBUTE │ ───────────>  │ ROOT-CAUSE │ (systematic-debugging)
   │    └────┬──────┘               └─────┬──────┘
   │         │ keep                       │ fix-forward / revert
   │         v                            │
   │    increments remain? ──yes──────────┘
   │         │ no
   │         v
   │    ┌──────────────┐
   └─no─│ FINAL VERIFY │  full suite green ?
        └──────┬───────┘
               │ yes
               v
             DONE
```

## Phase contracts

### SCOPE
- **Input:** the user's request.
- **Do:** restate the goal in your own words. Enumerate constraints, unknowns, and the
  concrete signals that will prove success (which tests, what behavior, what command exits 0).
- **Exit:** `harness_state.py init "<done-definition>"` has been run. The done-definition must
  be falsifiable — "all auth tests pass and login still works" not "auth is better".

### PLAN
- **Input:** the scoped goal.
- **Do:** decompose into an ordered list of increments. Each increment must be independently
  verifiable (you can prove it works without the later ones) and small enough to revert
  cleanly. Use the `plan` skill for the decomposition discipline.
- **Exit:** the ordered increment list is captured (in the plan artifact and/or as the intended
  sequence you will feed to `add-increment`).

### IMPLEMENT
- **Input:** the next un-started increment.
- **Do:** make the *smallest* change that could satisfy it. For new behavior, write the test
  first (`test-driven-development`). Before/while editing, record the increment and its
  prediction with `add-increment ... --predict ... --risk ...`.
- **Exit:** code is written and the increment (with prediction) exists in state.

### VERIFY
- **Input:** the just-implemented increment.
- **Do:** run the external check appropriate to it (see `verification-protocol.md`). Capture
  the actual command and its result.
- **Exit:** `record-verification <id> <pass|fail|partial> --note "<command + outcome>"`.

### ATTRIBUTE
- **Input:** the verification result + the recorded prediction.
- **Do:** compare them. A surprise (predicted pass → got fail, or predicted regression →
  didn't happen) is signal — note it. Assign a verdict: **keep** (advance), **revert** (undo,
  log the lesson), **partial** (refine, re-verify).
- **Exit:** verdict recorded and the working tree is in a known-good state (failing increments
  are reverted or fixed before moving on).

### ROOT-CAUSE (only on fail)
- Switch to `systematic-debugging`. Find the actual cause before touching code again. Then
  return to IMPLEMENT (fix forward) or revert the increment. Never blind-patch a failing check.

### FINAL VERIFY
- When no increments remain, run the **full** relevant suite (not just the last increment's
  test) plus any end-to-end / run-the-app check. Drift between increments only shows up here.
- **Exit:** full suite green → DONE. Any failure re-enters ROOT-CAUSE.

## Termination rules

Stop the loop when **any** of:
- All increments are `keep` and FINAL VERIFY is green → success.
- An increment is fundamentally blocked (missing access, ambiguous spec) → stop and ask the
  user; record the blocker in state. Do not thrash.
- You have reverted the same increment twice without progress → stop, summarize what you tried
  (state log has it), and ask for direction. Repeated identical attempts are a loop, not work.

## Resuming after compaction / interruption

Run `status` first. It reprints the goal, the increment list with verdicts, and the log. Pick
up at the first increment without a `keep` verdict. Do **not** re-plan from scratch — that
discards verified progress.
