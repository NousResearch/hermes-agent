# NBQ improvement backlog

This is a living ranked list. Rank by expected value ÷ evaluation cost, highest first.

1. **#30 answerability weighting — PARKED AGAIN 2026-07-04 (iteration two): premise falsified for
   free.** The re-open condition (unanswerable 77% > 50%) was met, but a zero-cost retro probe of the
   objective corpus did NOT support #30's premise ("kept high-EVSI unanswerable questions cause
   objective failure"): highest-EVSI-unanswerable × fail r=+0.05 (within SE), any-unanswerable × fail
   degenerate (97% base rate) + wrong-direction. Unanswerability is near-universal in this corpus, so
   there is no answerable-question contrast to steer toward. **Re-open condition (new):** a
   higher-contrast answerability corpus exists (built via candidate 2 or 3 below) AND a non-self-rated
   mechanism is pre-registered (the batched strict-simulator answer/refuse probe is designed + parked
   in `next-best-questions/references/prereg-iteration-two.md` item B, unexercised). Full verdict:
   `next-best-questions/references/{evsi-validation-findings.md,design-decisions.md}` §Answerability.
2. **[NOW TOP for lap three] Build a real answerability corpus — candidate 2 (nbq→relentless
   integration) or candidate 3 (reach→investigate loop).** #30 is blocked on corpus, not mechanism:
   the retro probe showed the objective task set is ~universally unanswerable, so it can't test
   answerability steering. Candidate 3 is especially apt — routing reach questions to a (mocked)
   investigator that *resolves* some unanswerables into answerables is exactly the contrast #30 needs.
   Build one of these next; #30 rides on it.
3. **Retro answerability probe (candidate 1) — DONE 2026-07-04: PARK verdict delivered.** Built
   `evals/probe_answerability.py` (offline, deterministic, stdlib-only) + 5 unit tests; ran on the
   pinned objective corpus. Zero new model calls. Product: parked #30 for free and diagnosed *why*
   (near-universal unanswerability). Kept as a standing instrument. See candidate 1 below.
3. **#32 First-order candidate source — CLOSED 2026-07-04: NO ADOPT.** Built, off-by-default
   (`--firstorder`). Gate (n=34, all-deepseek): paired vs nbq Δ+0.049 but 6W/6L (broad-win guard
   fails), unanswerable 77%, a lens-payoff regression, and +16.8% wall (> 10% ceiling). Altitude has
   signal (mean beat plain nbq) but did not close the gap — zeroshot still won +0.274 (15W/1L,
   p=0.0005). Re-open only with a NEW hypothesis about *why* altitude-in-the-pipeline underperforms
   (candidate: the pipeline demotes the first-order candidates — inspect their per-stage P/Δ/stakes).
   Full verdict: `next-best-questions/references/design-decisions.md` §First-order candidate source.
4. **#33 Discrimination preflight — CLOSED 2026-07-04: ADOPTED (opt-in instrument).** `--strict-preflight`;
   live check fast 8/8, deepseek 8/8. No default flip (8 calls/model, off by default).
5. **Prompt distillation — unblocked by the #32 no-adopt (the value model is not the gap), lower
   priority than #30.** Re-scope the certified-prompt path against whatever answerability mechanism
   #30 lands.

## Agentic-workflow integration candidates

The RESEARCH step (step 2, run by the main loop each iteration) populates this section with ranked
candidates for where `next-best-questions` should hook into planner/executor loops:
`relentless-solve`'s clarify step, `investigator` routing, and `task-decomposer`.

Ranked by expected value ÷ evaluation cost (iteration-one sweep, 2026-07-04). Sources:
Clarifier→Planner→Implementor (EMNLP 2025 industry 163), QualityFlow (verifier-gated clarify
branch), DenoiseFlow (arXiv:2603.00532, sense/propagate/control semantic uncertainty), adaptive
planning-horizon (arXiv:2605.08477), Routine (arXiv:2507.14447).

1. **[HIGH EV/cost] Retro answerability↔objective-failure probe (inside `outcome_eval`, zero new
   model calls). DRIFT-CORRECTED 2026-07-04 (iteration two REVIEW).** The originally-planned join
   against `relentless`'s `journey.json` is NOT runnable: the only journey.json files present are the
   three relentless smoke fixtures, and none carry nbq assume-default annotations (that annotation is
   the *unbuilt* candidate 2 integration — so the join key does not exist yet). The genuinely cheapest
   falsifying test lives in the OBJECTIVE harness output instead: `~/.hermes/outcome_eval_32.json`
   already carries, per task/arm, `meta.q_values` (per-question EVSI, index-aligned), `qa[i].revealed`
   + `qa[i].answer` (answerability signal — "The spec doesn't say." = unanswerable), and `frac` /
   `per_test` (objective outcome). Hypothesis: tasks whose kept high-EVSI questions were *unanswerable*
   fail the objective outcome more often than tasks whose high-EVSI questions were answerable.
   Falsifying result (no correlation, or unanswerable↔success) → #30's "answerability causes failure"
   thesis is falsified for free → park #30, re-scope. Feeds #30. The journey.json join is re-parked
   behind candidate 2's (unbuilt) nbq→relentless integration.
2. **[HIGH] nbq as the Clarifier preflight into `relentless-solve`'s clarify step (EMNLP pattern).**
   Hypothesis: EVSI-ranked pre-plan clarification reduces replans / failed paths vs unstructured
   clarify. Mechanism: wire nbq's top-K bucket as the Clarifier's critical Q-A pairs feeding the
   planner. Cheapest test: A/B a small relentless task set with vs without the nbq preflight, primary
   metric = replan count / journey failed-path count (a workflow-integration change, not an
   elicitation change → two-arm, not the objective harness). Cost: needs the relentless harness.
3. **[MED-HIGH] Reach→investigate→evidence loop (ties reach lens #29 to `investigator` routing).**
   Hypothesis: routing reach questions to the investigator (execute the hop, return the observable
   as evidence, re-run nbq) resolves "unanswerable" questions that today inflate the unanswerable
   rate. Cheapest test: on access/systems tasks, measure unanswerable-rate drop when reach questions
   are answered by a (mocked) investigator returning the observable. Connects to #30 answerability.
4. **[MED] Mid-execution re-clarify trigger (QualityFlow / adaptive-horizon).**
   Hypothesis: clarification value peaks at replan boundaries where a plan just failed, beating
   up-front-only clarification. Mechanism: expose nbq as the "clarify" branch of a verifier choosing
   submit/clarify/revert/continue at replan nodes. Cheapest test: overlaps candidate 1 — measure
   whether residual high-value questions cluster on the tasks that hit a replan. Gate the retro
   analysis before any real integration.
5. **[MED] Uncertainty propagation into execution (DenoiseFlow).**
   Hypothesis: propagating each kept question's assume-default risk (already rendered as "~X% chance
   that's off in a way that matters") forward as an executor caution signal reduces silent
   wrong-output. Cheapest test: instrument the existing assume-default annotations; check whether
   they flag the tasks that later fail an objective check. Formula-frozen (report annotation only).

## Parked (with re-open conditions)

- **A1:** parked audit item; re-open condition: TBD on next review.
- **A2:** parked audit item; re-open condition: TBD on next review.
- **A10:** parked audit item; re-open condition: TBD on next review.
- **M2:** parked audit item; re-open condition: TBD on next review.
