# Methodology — why this scores questions the way it does

This skill estimates the **value of information** of clarifying questions and keeps only the
ones worth resolving before work begins. The design follows the decision-theoretic and
information-theoretic literature; this note records the grounding so the scoring can be audited
and tuned.

## 1. The quantity we approximate: Expected Value of Sample Information (EVSI)

Howard's *Information Value Theory* (1966): the value of an observation is the expected payoff of
the best decision you can make **with** it minus the best decision **without** it. The load-bearing
consequence: **information has value only if it could change the optimal decision.** If the answer
wouldn't change what you do, it is worthless no matter how interesting.

Formally, for a decision `d`, latent `θ`, utility `U`, and a question `q` whose answer is `a`:

```
EVSI(q) = E_a[ max_d E_{θ|a} U(d,θ) ]  −  max_d E_θ U(d,θ)
```

Note the **outer expectation over answers, weighted by P(a)**. The information-theoretic analogue
(Lindley 1956) replaces utility with entropy reduction: value = mutual information `I(θ; a)`.

## 2. How we estimate it with LLMs (answer simulation)

We can't enumerate the true decision space, so we use the standard LLM approximation — **simulate
the plausible answers and measure how much each would change the recommended plan** (Rao & Daumé
2018; Uncertainty-of-Thoughts 2024; Mazzaccara et al. 2024 "LLM-as-EIG-estimator"):

1. **Baseline plan `plan*_0`** — the plan given only the problem as stated (stage 0).
2. **Project answers** with probabilities `P(a)` (stage 2).
3. **Per-answer plan-change** `Δplan(a)` and **stakes** `stakes(a)` vs the baseline (stage 3).
4. **EVSI estimate:**  `EVSI(q) = Σ_a P(a) · Δplan(a) · stakes(a)`  — probability-weighted expected
   "regret avoided." Δplan and stakes are coupled **per answer** inside the expectation (a large
   plan change under a 2 %-likely answer is correctly down-weighted).

## 3. The uncertainty gate U (reducible, not raw)

EVSI alone can be inflated if the model invents answer-spread for a question whose answer is
actually obvious from the prompt. We gate by **reducible (epistemic) uncertainty** — the BALD
distinction (Houlsby et al. 2011): we want uncertainty that *asking can reduce*, not irreducible
noise.

```
U = normalized_entropy(P(a)) · (1 − derivable_prob)
```

`derivable_prob` (stage 2) discounts uncertainty that's resolvable from the prompt alone. We
*measure* U from simulated answers rather than asking the model to self-rate ambiguity — CLAMBER
(Zhang et al. 2024) shows LLMs are unreliable self-judges of when to clarify.

## 4. The composite value and the gate

```
gate:   discard if  U ≈ 0  OR  EVSI ≈ 0          # necessary conditions (Howard)
value:  value(q) = √( U · EVSI )                  # geometric mean, [0,1]
```

Why this shape:
- The **product/gate** correctly encodes that EVSI = 0 if *any* necessary condition fails — no
  reducible uncertainty, or no probability-weighted plan-change×stakes.
- The **geometric mean** keeps `value` on an interpretable ~0–1 scale (so absolute thresholds like
  0.40 are meaningful) while preserving "any factor 0 ⇒ value 0."
- The earlier naive design (a geometric mean of three *independent global* scalars U·S·K) is
  rejected: it omits P(a) weighting and decouples plan-change from stakes. The fix — fold P(a) in
  and couple Δplan×stakes per answer inside EVSI — is the single most important correction from the
  literature.

## 5. Diversity: don't keep redundant questions (BatchBALD)

Scoring questions independently and taking top-k yields a **redundant** set — BatchBALD (Kirsch et
al. 2019) showed naive top-k batch acquisition picks near-duplicates and can underperform random.
Two questions that resolve the **same hidden latent** have joint value ≈ the max, not the sum
(submodularity / diminishing returns; Golovin & Krause 2011).

Mitigation here:
- Each question is labelled with the `target` latent it resolves; questions sharing a `target`
  (or with high text overlap) are collapsed to the **highest-value representative**.
- Remaining keepers are ordered by greedy **MMR** (Carbonell & Goldstein 1998):
  `value − λ · max similarity to already-kept`, λ ≈ 0.4.

## 6. Stopping / bucket-fill strategy

Three OR-combined stop conditions (active learning + sequential experimental design):

1. **Bucket target reached** — enough diverse high-value questions.
2. **Marginal-value floor** — the best *fresh* candidate falls below `refill_floor`: continuing
   buys little (the EVSI ≤ cost-of-asking rule; fixed thresholds are the standard practical
   heuristic for the otherwise-myopic optimal-stopping rule).
3. **Round cap** — bounded cost.

**Default sizes** are grounded in entropy/20-questions reasoning: resolving `N` roughly-equally
plausible interpretations needs ≈ `log₂(N)` well-chosen questions; typical underspecified problems
have a handful of independent ambiguous dimensions → ~3–5 questions. The upper cap (~7) is a
human cognitive-load limit, not a VOI limit. We **report transparently** when the bucket can't be
filled — a well-specified problem is itself a useful finding (no silent truncation).

| knob | default | rationale |
|---|---|---|
| `min_bucket_size` | 3 | floor of useful clarification (≈ log₂ of a few ambiguous dims) |
| `target_bucket_size` | 5 | typical sweet spot |
| `hard_cap` | 7 | UX cognitive-load ceiling |
| `discard_threshold` | 0.40 | below = not valuable (user-chosen) |
| `pre_answer_threshold` | 0.60 | above = resolve before continuing |
| `refill_floor` | 0.30 | stop refilling when best fresh candidate drops below this |
| `answers_per_question` | 5 | literature uses 4–8 simulated answers |
| `max_rounds` | 3 | cost cap (user-chosen) |
| `mmr_lambda` | 0.4 | relevance-vs-diversity balance (0.3–0.5 typical) |

## 7. Key sources

| Source | Takeaway | URL |
|---|---|---|
| Howard, *Information Value Theory* (1966) | VOI = best-with − best-without; **0 unless it changes the decision** | https://scispace.com/papers/information-value-theory-1hz7dq8m1k |
| Lindley, EIG (1956) | Experiment value = expected entropy reduction = `I(θ;a)` | https://eprints.qut.edu.au/75000/1/75000.pdf |
| BALD (Houlsby et al. 2011) | Rank by reducible uncertainty = models confident but **disagreeing** | https://arxiv.org/abs/1112.5745 |
| BatchBALD (Kirsch et al. 2019) | Independent top-k is redundant; use joint info (submodular) | https://arxiv.org/abs/1906.08158 |
| MMR (Carbonell & Goldstein 1998) | `relevance − λ·redundancy` reranking for diverse sets | https://www.researchgate.net/publication/2269571 |
| Adaptive submodularity (Golovin & Krause 2011) | Sequential info-gathering has diminishing returns; greedy near-optimal | https://www.jair.org/index.php/jair/article/download/10731/25633/19980 |
| Rao & Daumé (2018) | Clarifying-Q value = `Σ_a P(a|p,q)·U(p+a)` — simulate answers, weight by prob | https://arxiv.org/abs/1805.04655 |
| GATE (Li et al. 2023) | LM-driven active task elicitation beats user-written prompts | https://arxiv.org/abs/2310.11589 |
| STaR-GATE (Andukuri et al. 2024) | Self-improve question-asking; effective in ≤3 turns | https://arxiv.org/abs/2403.19154 |
| Uncertainty-of-Thoughts (Hu et al. 2024) | Simulate future answers, reward = info gain (20-questions style) | https://arxiv.org/abs/2402.03271 |
| Mazzaccara et al. (2024) | Train to maximize `EIG = H_prior − E_a H_post`; EIG=1 at 50/50 split | https://arxiv.org/abs/2406.17453 |
| CLAM (Kuhn et al. 2022) | Selective clarification: classify-if-ambiguous → ask | https://arxiv.org/abs/2212.07769 |
| CLAMBER (Zhang et al. 2024) | LLMs are **poor at self-judging** when to clarify — measure, don't ask | https://arxiv.org/abs/2405.12063 |
| Optimal Stopping for Seq. BED (2025) | Stop when terminal reward ≥ expected continuation value | https://arxiv.org/abs/2509.21734 |
