# Caduceus — planning-loop parity

**Claim:** Caduceus's deep-planning behaviour faithfully implements the
disciplined to-do loop that makes top-tier agentic coders feel reliable — and
this is *auditable*, not asserted. The reference contract is captured verbatim
in [`evidence/PLANNING_LOOP_CONTRACT.md`](evidence/PLANNING_LOOP_CONTRACT.md);
the rubric that measures it lives in [`eval/parity_eval.py`](eval/parity_eval.py).

We deliberately do **not** claim byte-identical behaviour to any specific agent:
Caduceus reuses Hermes's own `todo` tool and runs on whatever model the user has
configured. What we claim — and test — is **behavioural fidelity** of the loop.

## Rule-by-rule

| # | Reference rule | Caduceus mechanism | Status |
|---|---|---|---|
| 1 | Plan for 3+ step / non-trivial work | `STANDING_REMINDER` "RIGHT-SIZE THE PLAN" + core `todo` tool | ✅ |
| 2 | Multiple asks → capture as todos | same | ✅ |
| 3 | Capture new instructions immediately | "capture new instructions as todos immediately" | ✅ |
| 4 | Mark `in_progress` **before** starting | "ONE STEP AT A TIME" rhythm | ✅ |
| 5 | Exactly one `in_progress` | reminder + `todo` schema | ✅ |
| 6 | Mark `completed` immediately, no batching | "never collapse several completions into one write" | ✅ |
| 7 | Remove irrelevant todos | "drop todos that stop being relevant" | ✅ |
| 8 | On failure: keep `in_progress` + add a blocker todo | "COMPLETION HONESTY" clause | ✅ |
| 9 | Never mark done on failing/partial work | same | ✅ |
| 10 | Verify before done (lint/typecheck/build/tests) | "VERIFY" clause | ✅ |
| 11 | Failing-test-first when tests exist | "prefer writing a failing test first" | ✅ |
| 12 | Proactive but not surprising | "PROACTIVE, NOT SURPRISING" clause | ✅ |
| 13 | Right-size — skip ceremony on trivial work | "a simple 1-2 step ask — just do it" | ✅ |
| 14 | Specific, actionable items | "ordered, specific plan" | ✅ |
| 15 | Plan first, then execute | "your FIRST action is a `todo` plan" | ✅ |

**15/15 behavioural rules implemented.**

## Irreducible differences (not claimed identical)

1. **Model.** Behaviour ultimately depends on the configured model; the prompt
   gets the *planning* and *right-sizing* reliably, while *maintenance*
   granularity is model-dependent on short tasks.
2. **Tool schema.** Caduceus reuses Hermes's `todo` tool rather than cloning a
   specific tool's schema; usage is steered to the discipline above.
3. **Enforcement.** Like every leading agent, the discipline is prompt-driven,
   not a hard state machine. The eval *measures* adherence.

## Re-verify

```bash
python3 docs/caduceus/eval/parity_eval.py          # offline rubric self-test (no keys)
python3 docs/caduceus/eval/parity_eval.py --live   # run fixed prompts through the agent + score
python3 docs/caduceus/eval/ab_compare.py           # baseline Hermes vs Caduceus, same model
```

The offline rubric self-test ships a set of good and deliberately-broken
trajectories and confirms the evaluator passes the good ones and flags each
specific violation — so the measurement itself is trustworthy.
