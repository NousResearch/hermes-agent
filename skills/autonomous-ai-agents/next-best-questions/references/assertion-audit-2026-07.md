# Adversarial assertion audit — deepseek-v4-pro (2026-07-03)

> Auditor: deepseek-v4-pro:cloud, prompted to attack, not agree. Verdicts on the program's
> ten standing assertions + unstated assumptions. Outcomes of the falsifying tests that were
> actually run are in evsi-validation-findings.md §Deepseek re-adjudication (A3/A4/A9
> falsified-rescue / A8 confirmed-hit / M3 neutralized). Kept verbatim below.

# A1: Greedy one-step question selection is at least as good as multi-step lookahead

**VERDICT: PLAUSIBLE-BUT-UNDERTESTED**

**Strongest counter-argument:** The cited literature head-to-heads (arXiv:2602.14279) compare greedy vs. lookahead *for information gain*, not for utility-weighted EVSI with stakes. The program's own formula multiplies by Δplan·stakes, which creates exactly the regime where lookahead should dominate: when a low-EVSI question gates a high-stakes question by resolving which plan branch to pursue. The program never tested whether utility-weighting changes the greedy-vs-lookahead calculus, and the literature they cite didn't either.

**Cheapest falsifying test:** Construct 10 task-clarification scenarios where one question is a "gate" (low EVSI but determines which of two high-stakes follow-ups becomes relevant). Run greedy vs. two-step lookahead with the actual √(U·EVSI) formula. If lookahead wins on realized regret in ≥7/10, the architecture decision is wrong.

---

# A2: value = sqrt(U*EVSI) is the best within-task ranking formula

**VERDICT: SUSPECT**

**Strongest counter-argument:** The program admits P1c showed an all-fast judge once ranked U-only best, then dismisses this as "instrument-sensitive" while treating the +0.36 re-validations as definitive. But the re-validations use the *same judge architecture* that produced the anomaly. This is circular: the judge that prefers sqrt(U·EVSI) keeps confirming sqrt(U·EVSI). There's no independent ground truth — "realized regret" is just the same judge scoring a different prompt. The formula could be fitting judge bias, not actual decision value.

**Cheapest falsifying test:** Run a human-graded ranking experiment (n=8 tasks, 3 domain-expert humans per task) comparing sqrt(U·EVSI) against U-only, EVSI-only, and U·EVSI (no root). If human rankings correlate more strongly with a different formula, the frozen-formula claim collapses.

---

# A3: #26 verdict — sampled P(a) doesn't improve ranking despite moving probabilities

**VERDICT: PLAUSIBLE-BUT-UNDERTESTED**

**Strongest counter-argument:** The null result used an all-fast judge (same model that generated the answers). If the judge is insensitive to the *correctness* of P(a) estimates — i.e., it rates questions similarly whether they're based on calibrated or miscalibrated probabilities — then the experiment measures judge consistency, not ranking quality. The 79% probability movement could be real improvement that the judge simply can't detect because it's the same model that produced the miscalibrated stated-P in the first place.

**Cheapest falsifying test:** Run #26 again but with a *stronger model* as judge than as generator (e.g., Claude Opus judging Ollama-generated questions). If sampled-P now shows positive Δρ, the null was a judge-ceiling artifact, not a true negative.

---

# A4: #27 verdict — solution-space Δplan collapses to near-binary and is decisively worse

**VERDICT: SUSPECT**

**Strongest counter-argument:** The program's own alternative hypothesis is the most parsimonious explanation: the fast local model used for solution sampling *cannot generate diverse viable solutions*, so deltas collapse because the model can only imagine one or two solution approaches. The program tested a method that requires creative solution-space exploration using a model that likely lacks that capacity, then declared the method dead. This is like testing a telescope by pointing it at the ground.

**Cheapest falsifying test:** Re-run #27 with a strong frontier model (GPT-4o or Claude 3.5 Sonnet) doing the solution sampling, keeping the local model for everything else. If solution-space Δplan recovers granularity and beats stated Δplan, the "inherent" failure claim is wrong — it's a model-capability floor, not a method failure.

---

# A5: Premortem/vantage keyword gates are the right recall/precision mechanism

**VERDICT: PLAUSIBLE-BUT-UNDERTESTED**

**Strongest counter-argument:** The validation is a 13-prompt scored scan showing nouns re-open a "known false positive." That's a recall test on a tiny, hand-picked sample. There's no systematic precision measurement — how many irrelevant questions do these gates *admit* that a simpler mechanism (e.g., "always include one premortem-style question") would also admit? The gates add complexity (verb-only matching, framing detection) with no evidence they outperform a trivial baseline of always injecting one stakes-probe question.

**Cheapest falsifying test:** Run a precision comparison on 50 diverse task prompts: verb-only gates vs. a baseline that unconditionally appends "What's the worst-case failure mode here?" Measure false-positive rate (questions the judge deems irrelevant). If the baseline's false-positive rate is within 5% of the gated version, the gate machinery is dead weight.

---

# A6: Dropping question_relevance when bucket is empty is sound

**VERDICT: SUSPECT**

**Strongest counter-argument:** The program acknowledges the risk — an empty bucket could indicate a genuine tool failure rather than a well-specified prompt — and claims `calibration` catches it. But `calibration` checks whether the model can answer *content* questions, not whether it correctly identified that *no clarification is needed*. A model that hallucinates confidence in an underspecified prompt will pass calibration and then the empty-bucket path will silently suppress the relevance check that would have flagged the problem.

**Cheapest falsifying test:** Construct 10 deliberately underspecified prompts where clarification is objectively needed. Run the full pipeline with and without the empty-bucket relevance bypass. If the bypass path fails to flag ≥3 prompts that humans identify as needing clarification, the `calibration` guard is insufficient.

---

# A7: Single empty-content preflight is sufficient to catch unusable judge models

**VERDICT: SUSPECT**

**Strongest counter-argument:** The program itself names the failure mode — a model that answers but judges randomly passes preflight. This isn't a hypothetical edge case; it's the expected behavior of weak local models under the program's own scoring regime. The preflight only catches models that can't format output at all. A model that produces plausible-looking but noise-dominated rankings (exactly what would explain the persistent ρ≈0.34 ceiling) sails through.

**Cheapest falsifying test:** Run preflight on 5 candidate judge models, then run each through a *discrimination* preflight: present 10 question pairs where one is objectively better (constructed so the answer to Q1 changes the response, Q2 doesn't). A model that scores <7/10 correctly fails the real preflight. Compare which models pass the current empty-content preflight vs. this discrimination preflight.

---

# A8: U is inert for ranking but load-bearing for the gate

**VERDICT: SUPPORTED** (with a caveat)

**Strongest counter-argument:** If U is genuinely inert for ranking, then sqrt(U·EVSI) ≈ sqrt(EVSI) for live questions, which means the formula is effectively sqrt(EVSI). But EVSI already contains P(a) terms. The U multiplier is either redundant (if U is constant across live questions) or it's doing something (if U varies). The program can't have it both ways — either U varies and affects ranking, or it doesn't and the formula is mis-specified. The gate behavior (U→0 retires questions) proves U varies, so it *must* affect ranking when questions have different derivability.

**Cheapest falsifying test:** On the existing validation data, compute the rank correlation between U-only and the full sqrt(U·EVSI) ranking. If ρ > 0.9, U is genuinely inert for live-question ranking and the formula is overparameterized. If ρ < 0.7, A8 is false and U is doing ranking work.

---

# A9: After three powered negatives, input estimation is not the lever; residual is irreducible noise

**VERDICT: SUSPECT**

**Strongest counter-argument:** All three "powered negatives" share the same confound: the judge is the same class of model as the estimator. #24 tested pairwise vs. absolute (judge is the bottleneck), #26 tested sampled vs. stated P(a) (judge is the bottleneck), #27 tested solution-space vs. question-space (judge AND generator are the bottleneck). The program has tested "can a weak model detect improvements made by a weak model" and concluded improvements don't exist. The consistent ρ≈0.34–0.36 ceiling is exactly what you'd expect if the judge's noise floor is ~0.35.

**Cheapest falsifying test:** Run one experiment where the *judge* is a strong frontier model (not local Ollama) scoring questions generated by the local model. If ρ jumps above 0.5, the ceiling was judge noise, not irreducible task noise. The entire "estimation is not the lever" conclusion unravels.

---

# A10: LLM-judged "realized change/regret" is valid ground truth

**VERDICT: SUSPECT**

**Strongest counter-argument:** The program uses an LLM to judge how much a re-derived response differs from baseline, then multiplies by judged stakes. This is ground truth only if the judge correctly assesses counterfactual response quality — but the judge never sees actual task outcomes, user satisfaction, or objective success metrics. It's evaluating whether the *answer changed*, not whether the change was *correct or valuable*. A question that introduces a confident error scores high on "realized change" but is actively harmful. The entire validation edifice rests on a proxy that confuses change magnitude with value.

**Cheapest falsifying test:** For 20 task-clarification episodes, collect human judgments of whether the clarified response was *actually better* than the baseline (binary: improved/degraded). Compare human judgments to the LLM "realized regret" scores. If correlation is below 0.5, the ground-truth proxy is invalid and all ρ-based conclusions are uninterpretable.

---

# MISSING: Unstated assumptions most likely to bite

## M1: The judge model's ranking noise is uncorrelated with question characteristics

The program assumes judge error is random and washes out across questions. But if the judge systematically overrates certain question types (e.g., concrete factual questions over abstract strategic ones, or short questions over long ones), then the ranking formula is being optimized for judge preference, not task value. The consistent ρ≈0.35 could reflect the judge's systematic biases, not the formula's ceiling. No experiment controls for judge-model preference artifacts.

**Test:** Run the same question set through three different judge model families (e.g., Llama, Qwen, Gemma). If question rankings are unstable across judges (inter-judge ρ < 0.5), the "ground truth" is judge-specific and no formula optimization is meaningful.

## M2: Task clarification value is additive across questions

The entire EVSI framework assumes the value of asking Q1+Q2 equals value(Q1) + value(Q2). But in real task clarification, questions interact: asking Q1 can change what Q2 means, make Q2 unnecessary, or reveal that Q2 was the wrong question entirely. The program acknowledges this for inter-question dependencies (failure mode #3) but only in the context of lookahead. The deeper problem is that *even single-step EVSI estimates are contaminated* because the baseline plan against which Δplan is computed assumes no other questions will be asked — but in practice, multiple questions are asked and answered before the plan changes.

**Test:** Run episodes where questions are asked one-at-a-time with plan re-derivation between each (sequential), vs. the current batch-ask-then-re-derive. If sequential produces different question rankings or higher realized regret, the additivity assumption is violated and the entire one-step EVSI framework is built on sand.

## M3: The frozen formula generalizes beyond the validation task distribution

All validation was done on whatever task distribution the program naturally encounters. If that distribution is narrow (e.g., mostly code-generation tasks, mostly well-specified prompts, mostly single-turn clarifications), the formula may be overfit to that regime. The program has no holdout by task type, no stratification by task complexity, and no adversarial task sampling. A formula that works for "write a Python function" may fail for "design a system architecture" or "debug this distributed system failure" — domains where stakes structure, answer distributions, and derivability patterns differ qualitatively.

**Test:** Stratify the existing validation data by task category (code, writing, analysis, design, debugging). If ρ varies by >0.15 across categories, the frozen-formula claim is premature and category-specific formulas or adaptive selection are needed.