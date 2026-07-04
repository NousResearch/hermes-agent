# NBQ improvement backlog

This is a living ranked list. Rank by expected value ÷ evaluation cost, highest first.

1. **#30 answerability weighting — NOW RE-OPENED, top of the queue (iteration two lead).** The
   iteration-one gate met its re-open condition: nbq-firstorder unanswerable rate 77% > 50%, and the
   #32 diagnosis pinned the residual P4 gap on *answerability*, not candidate altitude. Constraint
   (unchanged): the mechanism must NOT be self-rated (the old multiplier was inert at 0.95 in 15/16
   cells; a self-rated ceiling already failed once). Candidate mechanism to pre-register: answerability
   estimated by the strict-simulator signal / a reachable-vantage check (see integration candidate 3),
   not by the model's own self-report.
2. **Agentic-workflow integration (see below) — candidate 1 is zero-build** and directly probes
   whether residual high-value (assume-default) questions predict failures; run it before #30's build.
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

1. **[HIGH EV/cost] Residual-question ↔ failed-path correlation (uses EXISTING logs, zero build).**
   Hypothesis: skipped high-EVSI (assume-default) questions predict downstream execution failures.
   Mechanism: `relentless-solve`'s `journey.json` already records failed-paths-as-evidence; join
   nbq's per-task assume-default annotations against those failures. Cheapest falsifying test: retro
   analysis of existing journey logs — if no correlation, the "clarify earlier" thesis is falsified
   for free. Feeds candidates 2 and #30.
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
