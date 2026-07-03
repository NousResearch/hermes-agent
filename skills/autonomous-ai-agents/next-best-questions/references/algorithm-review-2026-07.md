# Is the EVSI ranker the right algorithm? — research review (2026-07)

Question asked (jim, 2026-07-02): is the skill's algorithm — greedy one-step EVSI ranking,
`value = √(U·EVSI)`, `EVSI = Σ_a P(a)·Δplan(a)·stakes(a)`, U/EVSI gate, MMR/BatchBALD dedup,
families/lenses, real-evidence loop — the best agentic approach for surfacing the key questions?
Scope: the skill's validated domain (agentic task clarification: ask / go-find-out / just-do-it).

**Verdict: yes on architecture; the frontier's one real edge is input *estimation*, which became
the #26/#27 gated experiments** (see `design-decisions.md`). The formula stays FROZEN.

## What the 2023–2026 literature validates (keep as-is)

1. **Greedy one-step beats lookahead in this domain.** "Whom to Query for What" (2026,
   arXiv:2602.14279) compares greedy EIG selection against multi-rollout lookahead head-to-head:
   gains are "at most marginal … small, inconsistent, and largely disappearing or reversing at
   higher budgets." Multi-step lookahead is NP-hard even at depth 2 (arXiv:1802.03654). UoT's +38%
   (NeurIPS 2024, arXiv:2402.03271) is vs *naive prompting*, not vs a strong greedy ranker; lookahead
   earns its keep only where single questions have low marginal IG and inter-question dependencies
   dominate (20-Questions endgames, partial observability, relationship-graph mapping). The skill's
   rejected "multi-step projected chains" decision is correct; the real-evidence loop is the right
   depth mechanism (matches sequential BOED practice).
2. **Utility weighting (Δplan·stakes) is right and underused.** Most published EIG work optimizes
   Shannon information; decision-theoretic weighting (Howard VOI, Rao & Daumé EVPI,
   arXiv:1805.04655) is the correct objective and few systems have it. The skill is *ahead* of much
   of the field here.
3. **The ask-vs-derive gate matches** "Modeling Future Conversation Turns" (ICLR 2025,
   arXiv:2410.13788): ask only when the answer changes the response. The derivable→U→0 mechanic is
   the ask-vs-find-out discriminator, and in-domain validation (per-answer ρ=0.64) confirms it.
4. **Same-target collapse + MMR is a lightweight BatchBALD** (arXiv:1906.08158) — the principled fix
   for redundant top-k; full joint-information scoring would be cost without evidence of need.
5. **Families/lenses (esp. premortem) have no direct literature analog and validated well** (top
   lens by realized_regret at both #25 eval tiers). GATE (arXiv:2310.11589) is the closest framing
   (generative elicitation), but nothing published hunts the stakes tail the way premortem does.
6. **Closed negatives stay closed.** Nothing in the literature re-opens #24 pairwise (null at n=12),
   #23 rank-relative, the graded judge, or the answerability multiplier.

## Where the frontier is ahead (→ #26/#27)

The skill elicits its two load-bearing numbers — P(a) and Δplan(a) — by *asking a model to state
them*. Three convergent lines say that is the weak link, and it maps exactly onto the one measured
weakness (within-task ranking, per-prompt ρ≈0.34–0.36):

- **BED-LLM** (arXiv:2508.21184): entropy/LLM-scored EIG proxies are unreliable; Monte-Carlo EIG
  from sampled answer rollouts is materially better (10–20% on multi-class tasks). → **#26
  sampled P(a)**.
- **OPEN / Bayesian preference elicitation** (arXiv:2403.05534): LLM proposes the space, a
  calibrated estimator owns the probabilities. → same.
- **Active Task Disambiguation** (ICLR 2025, arXiv:2502.04485) + **ClarifyGPT** (FSE 2024,
  arXiv:2310.10996): compute value in the *solution space* — sample viable solutions, ask what
  splits them — instead of judging questions abstractly. → **#27 solution-space Δplan**.

Both shipped as off-by-default selectors gated by the #24-pattern powered A/B (n=12,
`realized_regret`, paired-Δρ broad-win guard).

**Outcome (same day, 2026-07-02): both frontier critiques FAILED the gate on this skill's domain.**
#26 sampled P(a) = real-contrast null (P moved on 79% of pairs, ranking Δρ +0.010, keep stated);
#27 solution-space Δplan = decisively worse (deltas collapse to near-binary, ρ −0.047 vs +0.360,
keep absolute). Combined with #24 (pairwise null), the empirical picture is now three independent
literature-motivated input-estimation upgrades, three powered negative results — while `√(U·EVSI)`
re-validated as the best within-task formula in both new runs (+0.356/+0.360 on regret). The
skill's stated-P, absolute-judge, frozen-formula configuration is not just defensible; it has now
out-tested the frontier's specific alternatives on its own domain. Details in
`evsi-validation-findings.md` §§Sampled P(a) (#26) / Solution-space Δplan (#27).

**Instrument-robustness addendum (2026-07-03):** a deepseek-v4-pro adversarial audit of this
review's assertions named the same-class-instrument confound (weak model judging weak-model
improvements) as the central threat, so both verdicts were re-adjudicated under deepseek — #26
with deepseek elicit+judge (still a null, Δρ +0.058), #27 ALL-deepseek including solution sampling
(still decisively worse, Δρ −0.369, even though granularity partially recovered — the audit's
"model floor" rescue falsified on its own terms). Judge agreement on identical responses ρ 0.814;
the within-task ρ ceiling did not move under the deep judge (A9 falsified). One audit hit stands:
"U is inert for ranking" was stale — measured ρ(U-only vs full ordering) ≈ 0.35–0.50, so U does
ranking work. Full audit: `~/.hermes/assertion_critique_ds.md`; verdicts: findings §Deepseek
re-adjudication.

## Considered and not built (with reasons)

- **UoT/MCTS lookahead** — evidence says marginal-at-best for this domain; expensive; projected
  chains already rejected.
- **Cost-aware value-per-cost ranking** (CuriosiTree, arXiv:2506.09173) — the investigator wrapper's
  top-K + floor already approximates budget control; YAGNI.
- **Learned ranker** (STaR-GATE arXiv:2403.19154, DPO-on-EIG arXiv:2406.17453) — needs fine-tuning
  infra; wrong fit for a local-Ollama prompt-time skill.
- **Sampling-disagreement ambiguity gates** (ClarifyGPT-style consistency, INTENT-SIM NAACL-F 2025)
  — partially subsumed: #26's forced-choice sampling IS an empirical self-consistency signal over
  the answer distribution; a full intent-graph layer is unjustified while #26's verdict is pending.
- **Structure-aware / relational elicitation** (KG-traversal question asking arXiv:2601.17716,
  causal-order-first LeGIT arXiv:2503.01139, competency questions arXiv:2412.20942) — out of scope:
  the skill's ontology is a flat set of independent latents, and jim scoped this review to the
  current domain. If a relationships/systems-mapping domain ever becomes a target, this is the
  family to revisit (question value there is structure-dependent: asking about edge A→B changes the
  value of asking about B→C — one-step question-space EVSI is weakest exactly there).

## Known failure modes of a greedy one-step EVSI ranker (from the survey)

1. Miscalibrated stated P(a)/U — #26 targets this.
2. Question-space (not solution-space) scoring — #27 targets this.
3. Inter-question dependencies / low per-question IG — accepted; not this domain's regime.
4. Redundant top-k — already handled (target collapse + MMR).
5. Heterogeneous answer cost ignored — deferred (wrapper top-K approximates).
6. Over-asking without an ask-vs-answer gate — already handled (U/EVSI gate + derivability).
