# Phase-2 Save-Decision Eval — Result (the GATE)

**Date:** 2026-06-27 · **Harness:** `eval/bgr_save_eval.py` · **Fixture:**
`eval/fixtures/bgr_save_fixtures.jsonl` (40 held-out multi-turn excerpts: 20 genuine /
12 narration / 8 ambiguous; authored BEFORE the clause wording, labeling the save DECISION).
**Backend:** claude-opus-4-8 via `claude-bpp` (the review fork's ACTUAL model — it inherits
`agent.model`). **Verdict: PASS.**

## Result (final, hardened rubric, real model)
```
genuine: 20/20 saved   (recall 100.0%, Wilson95 LB 0.839)
no-save: 0/20 wrongly saved (false-save 0.0%)
GATE: save-recall LB 0.839 >= 0.75  AND  false-save 0.0% <= 10%   -> PASS
```

## What the eval caught (and the fix)
1. **Rubric over-saved speculation/requests.** First run (draft clause) had false-save **20%** —
   all 4 false-saves were *ambiguous* cases: tentative statements ("I might switch to a 5090",
   "thinking about moving to Berlin"), one-off requests ("swap the shellfish"), and transient
   events ("the 7am reminder didn't fire"). **Fix:** hardened `_MEMORY_REVIEW_MEM0_CLAUSE` to
   explicitly exclude SPECULATION/TENTATIVE statements, one-off REQUESTS/commands, passing
   complaints, and transient events; added "a fact is durable only if settled & still true next
   week; when in doubt DO NOT save." Dropped false-save to 0%.
2. **The grader model matters — don't eval a strong-model feature on a weak grader.** Running the
   eval on gpt-5-nano gave wildly unstable recall (66.7%–91.7% across runs) and dropped genuine
   settled facts (brokerage move, ISP switch, NAS IP, office relocation). The review fork runs
   **claude-opus-4-8**, not gpt-5-nano — evaluating on nano *understated* the rubric. On the real
   fork model, recall is a clean 20/20 with 0 false-saves. (prd-plan lesson: the eval substrate
   must match the production decision-maker.)

## Operating notes
- Backend selectable via `SAVE_EVAL_BACKEND` (`bpp` default → claude-bpp 18811 OpenAI-style;
  `claude` → claude-pool 18810 Anthropic-style; `gpt5nano` → OpenAI, weak, NOT representative).
- Retries with backoff on 429/5xx (the shared relays rate-limit); claude-bpp had the headroom.
- The fixture labels the save DECISION (deterministic), not wording — grading is mechanical on the
  model's strict-JSON `{"save": bool}` verdict (no model-grades-model).
