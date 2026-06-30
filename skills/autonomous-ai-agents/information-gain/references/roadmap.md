# Roadmap — validate EVSI, then build the iterative context-builder

Driven by the benchmark (`benchmark-findings.md`). Sequence is deliberate: **get the rating right
before wrapping a loop around it.** Done first (cleanup): removed `answerability` (inert);
benchmark + conclusions saved.

## Simplified plan (YAGNI/KISS — CURRENT, 2026-06)

The validation was being gold-plated. The skill is a **report-only ranker**; the only thing that
matters is whether answering its top-ranked questions yields better outcomes — an **end-to-end** test
that makes per-factor (stakes) validation unnecessary. Cut back to what changes a decision.

**Cut / defer (YAGNI):**
- **Pairwise stakes instrument + realized-stakes decomposition** — drop; the end-to-end test covers it.
- **Formula changes** (drop-U, switch to max-Δ, simplify √) — don't; keep the frozen formula, no churn
  without a decision-forcing reason (U is needed in-domain — settled).
- **NOT_FOUND revival + `ctx_version` state machine** — defer; start with answered-facts + a plain gap
  record, add revival only if a real run shows it's needed.
- **Rank-relative change in `voi.py` (#23)** — defer; the wrapper takes **top-K by rank** from the
  ranked list, sidestepping the absolute threshold without touching the skill.
- **More eval harnesses** — stop; we have enough.

**Keep / do (simple, evidence-backed):**
1. **Minimal iterate-context wrapper** (`scripts/iterate.py`): rank → top-K by rank → grounded `ask`
   answers → append tombstone facts → re-rank → stop on a relative floor or `max_rounds` → final
   response. Configurable `K`/`max_rounds` + `stop_reason`. No revival machinery.
2. **Validate end-to-end — falls out of (1):** run the wrapper answering **top-K vs bottom-K**,
   blind-judge the two final responses; top should win (and the curve shows the floor). One artifact.
3. **Only if (2) says the ranking is the weak link** → revisit elicitation/thresholds. Evidence-gated.

Keeps the original "validate before you wrap" intent — the simplest valid test *is* running the
minimal wrapper two ways. Sections below are retained as history/rationale.

## Revised plan (post-domain-learnings, 2026-06)

The domain investigation (`evsi-validation-findings.md` §Domain sensitivity + Agentic realized
calibration) changed the plan in five concrete ways:

1. **Target domain, not life.** Validate and tune on the **agentic bank** (`testbank.BANK`), analyzed
   **per regime** — *ask-the-user* / *go-find-out* / *just-do-it*. Life questions are a degenerate corner.
2. **EVSI is partly rehabilitated.** On the target domain value/EVSI predict the *clean* realized-change
   signal (+0.66/+0.70 vs life's null) — mostly between-regime. So the formula is **not** scrapped; it's
   validated where it matters, pending the stakes half.
3. **`U` stays.** It's alive in agentic (sd 0.26) and is the ask-vs-find-out discriminator. The life-only
   "drop U" is dead. Freeze vindicated.
4. **Rank-relative selection is now REQUIRED** (#23), not conditional — the absolute threshold mis-fires
   across regimes (61% discarded).
5. **Phase 2 seam is visible now.** The *go-find-out* regime (high `derivable_prob` → U-gate fires) is
   exactly the grounded-research trigger the wrapper consumes — the skill already routes ask-vs-research.

Still gated on a de-confounded, powered **#21** (realized-stakes judge + max-Δ competitor + per-regime
+ pooling). Formula stays **frozen** until then.

## Phase 1 — Validate (and if needed, fix) EVSI

**Question:** does a high `value` actually predict a question whose answer improves the response?
The benchmark showed internal `value` and external quality can diverge (usaw: value 0.69, relevance
0.20). The EVSI *structure* is sound; the suspect links are input estimates, threshold scale, and
unproven validity.

**Phase-1 results so far (2026-06) — see `evsi-validation-findings.md`.** P1a (calibration) + P1c
(ablations) ran (`evals/validate_evsi.py` + `analyze_evsi.py`), reproduced + adversarially verified.
Verdict: **Δ component directionally calibrated** (per-answer ρ=0.39, cluster p=0.005); **`U` inert**
(0/40 within-prompt reorderings, anti-predictive alone); **full EVSI not-yet-validated** — null vs the
clean realized-change signal (ρ=−0.009), and its +0.605 vs realized-EVSI is a **stakes-reuse confound**
(partial-ρ|stakes = −0.13); **max-Δ** is the best clean predictor but marginal (p=0.064). n=17 / 3
prompts → directional. **Consequence: gate the wrapper on a de-confounded #21.**
**Decision (2026-06): FREEZE the formula — no changes on n=17; #21 decides every formula question
de-confounded and properly powered.** Caveat carried forward: `U` is inert *for ranking* but
**load-bearing for the gate** (`is_gated_out`: `derivable_prob`→1 → `U`→0 retires answered questions
across rounds), so any later "drop U" must **keep the derivability gate**, only removing U from the
`value` number.

**1.1 Use-relevant validity study (the core test).** For N prompts, produce three responses and
judge them blind for relevance to the prompt:
- `baseline` — respond with no clarification.
- `top2` — answer the **top-2** ranked questions, fold the answers in (evidence loop), then respond.
- `low2` — answer 2 **low-ranked / random** questions, fold in, then respond.
Pass condition: `top2 > low2 ≥ baseline`. If answering top-ranked questions doesn't beat answering
low-ranked ones, the rating isn't earning its place and we recalibrate/re-elicit. This study also
*builds the answerer + response-generator that Phase 2 needs* — so it de-risks the wrapper directly.
**It also yields `diminishing_floor` from evidence:** answer questions across the whole `value`
spectrum (not just top-2 vs low-2) and plot **realized improvement vs question `value`** — the floor is
where improvement flattens to ~0. Set the cap from the curve, don't guess it.

**Hard requirements added by Phase-1 (to break the confound P1c exposed):**
- **Validate on the AGENTIC bank, per regime — not life questions.** The domain scan (`evsi-validation-
  findings.md` §Domain sensitivity) shows life prompts are a degenerate corner (homogeneous,
  non-derivable; U inert; everything survives). The target domain spans three regimes —
  **ask-the-user** (spec-heavy coding/planning), **go-find-out** (high `derivable_prob` → U-gate fires
  → route to research), **just-do-it** (low EVSI → default). Use `testbank.BANK`; **analyze per regime**
  (a single pooled number averages three different mechanisms into mush).
- **Measure realized *stakes*, blind.** P1a/P1c only measured realized *change*; any "realized EVSI"
  that reuses projected stakes is confounded (ρ=0.96 collinearity nullifies it). The blind judge must
  rate the **importance/consequence** of the differences between responses, not just whether they
  changed — so realized EVSI = realized-Δ × realized-stakes is computed without the predictor's inputs.
- **Register `max-Δ` as a named competitor** alongside `√(U·EVSI)`, EVSI-only, U-only on the blind
  realized-improvement axis (P1c made it the leading clean-signal predictor, p=0.064 — resolve it here).
  Note: U is **not** inert in the agentic domain (sd 0.26 vs 0.07) — it discriminates ask-vs-find-out —
  so the √-form question must be re-run here, not inherited from the life result.
- **Pool across many more than 3 prompts**, with a **prompt-cluster bootstrap CI** (per-prompt n=5–6
  needs near-perfect monotonicity to reach significance; pool the question-level unit instead).

**1.2 Post-hoc formula ablations — DONE (P1c).** Ran from stored components (no model calls). Result:
**`U` inert** (`√(U·EVSI)` ≡ EVSI-only, 0/40 reorderings); **max-Δ** best vs the clean realized-change
signal (+0.526) while EVSI/value are ≈0 there (the EVSI signal lives entirely in stakes-weighting,
which 1.1 must validate de-confounded). See `evsi-validation-findings.md`.

**1.3 Calibration → rank-relative — now REQUIRED (was "likely").** The domain scan settles it: the
life-tuned absolute cutoff (0.40) discards **61%** of agentic candidates (vs 11% of life), and for
different reasons per regime — an absolute threshold cannot serve a domain this heterogeneous. Switch
selection to rank/relative (top-K, or ≥ X% of the round's best). **`U` is NOT dropped** — the life-only
"inert" finding does not hold in the agentic domain (U sd 0.26; it discriminates ask-vs-find-out). The
`√`-form / `U` question is re-opened for #21 on agentic data, not closed.

**1.4 Comparative elicitation — now the PATH FORWARD for stakes (was conditional).** Building #21's
de-confounding step proved absolute post-hoc **realized-stakes** judging is too fragile: a catastrophe
anchor collapses (35/36 → 0.0), a graded anchor central-tendency-clusters (12/18 → 0.6). See
`evsi-validation-findings.md` §"realized-stakes instrument". Fix: measure stakes **pairwise** —
*"which of these two clarifications matters more for the outcome?"* — and rank via Bradley-Terry / Elo
instead of brittle 0–1 ratings. Likely applies to **eliciting** projected Δ/stakes too (models compare
better than they score). Until this lands, the **stakes-half of EVSI is unvalidated by instrument
limitation, not by a negative result**; the Δ-half stands (agentic per-answer ρ 0.64).

**Exit:** an EVSI we've shown predicts realized improvement (or a recalibrated one that does).

## Phase 2 — The iterative context-builder ("iterate context")

A wrapper/orchestrator around the (validated) info-gain primitive that builds up context to
convergence and then responds. Keeps info-gain report-only; the wrapper does the answering + looping
+ responding (primitive-vs-orchestrator discipline).

**Why one continuously-growing context (the core principle).** Every round, the LLM conditions on the
*entire* accumulated context — prompt + all prior questions + every tombstone (answered facts *and*
known gaps). As tombstones accrue, the model's implicit posterior over the problem sharpens, so each
new round's questions, answer-projections, research, and the final response are all conditioned on
everything established so far. The single growing context **is** the substrate on which the LLM's
Bayesian updating operates continuously — which is exactly why we **always append** (never fragment or
reset the context) and why the tombstones must be **clean, high-signal facts**. That is the whole point
of the `ask`-isolation: the noisy research reasoning stays out of the loop; only the distilled fact
enters the context the model continuously reasons over.

**Loop:**
```
tombstones = []          # each: {question, status: ANSWERED | NOT_FOUND, answer?, ctx_version}
ctx_version = 0
for round in range(max_rounds):
    evidence = facts(tombstones)                               # answered facts + known-gaps — ONE shared context
    ranked = infogain.run(prompt, evidence=evidence)           # rank, given everything known so far
    top = [q for q in ranked.bucket if eligible(q, tombstones, ctx_version)][:K]
    if not top or top[0].value < diminishing_floor: break      # diminishing returns -> stop
    for q in top:
        res = ask_research(q, prompt, evidence)                # `ask` skill: isolated ctx, returns a distilled fact
        if res.found:
            tombstones += [(q, ANSWERED, res.answer, ctx_version)]; ctx_version += 1   # context grew
        else:
            tombstones += [(q, NOT_FOUND, ctx_version)]        # record the gap; don't re-research at this version
final = ask_respond(prompt, evidence)                          # best response, using the enriched context
return { final, tombstones }
```

**Components:**
- **ANSWERER = the `ask` skill (decided) — used to stay cheap and NOT pollute context.** Each research
  call is `ask` / `model_utils.dispatch_single`: a full Hermes agent in its OWN isolated context with
  **ALL tools available** (file, web, terminal, …) — the research agent is unconstrained in *how* it
  finds the answer. The heavy research reasoning stays in that subprocess; **only the distilled answer**
  (or `NOT_FOUND: <brief reason>`) returns and gets tombstoned. So the main iterate-context holds just
  the prompt + clean tombstones — lean, never bloated with research transcripts. Ask for a *concise*
  answer to keep the returned fact tight.
- **RESPONDER** — final step: "given the prompt + all established facts + known gaps, produce the best
  response." (A `dispatch_single` or strong Ollama call over the enriched context.)
- **CONVERGENCE + cap diagnostics** — stop when the round's top `value` < `diminishing_floor` or no
  eligible questions remain (**natural convergence**), OR when `K` / `max_rounds` is hit (**artificial
  cap**). Always record a **`stop_reason`** and flag **whether an artificial cap bound the run before
  it naturally ran dry** — i.e. `max_rounds` hit while top value ≥ floor, or rounds where *more than K*
  high-value questions were available (K rate-limited us). This is what tells us if the caps are cutting
  off real value vs. the loop genuinely converging.

**Tombstone state machine (the refinement):**
- **ANSWERED** (`Q → A`): a discovered fact. Enters the context; the evidence mechanism makes the
  question derivable so it drops out of future rounds.
- **NOT_FOUND** (`Q → not discoverable at this context state`): *informative, not a failure* — a known
  gap that shapes both the final response and the next questions. Two rules:
  - **Don't re-research it at the same `ctx_version`** (no wasted budget).
  - **Revive it when the context grows.** Every ANSWERED tombstone bumps `ctx_version`; a NOT_FOUND
    question attempted at an older version becomes **eligible again**, because a newly-discovered fact
    may open a path to it ("if we discover another path, the question could be answered"). `eligible()`
    enforces both rules. *(Likely a small addition to info-gain: pass NOT_FOUND questions as a
    "known gaps" list so generation neither re-asks them nor treats them as resolved.)*

**Where it lives:** `scripts/iterate.py` in this skill (composes `infogain.run` + `dispatch_single`
research + responder); promote to its own `clarify-and-respond` skill if it grows. Info-gain stays
report-only.

**Cost reality (grounded changes this a lot).** The scoring is cheap Ollama (~14 calls/round, ~30s),
but each grounded answer is a **full agent-loop research call (~30–60s + web)**. K=2 × ~3 rounds = ~6
research calls ≈ 3–6 min/run, dominating cost. Implications to decide: cap rounds tightly, cache
researched facts, and make the diminishing-returns floor *aggressive* so we don't pay to research
low-value questions.

**Decided:** answerer = `ask` (isolated context, distilled answer), **all tools available** to the
research agent · NOT_FOUND recorded as a tombstone + revivable when context grows · context is the
single continuously-growing shared state · **K = 6** (a cap for now) · everything **configurable** ·
the loop **reports `stop_reason` and flags when an artificial cap (K / max_rounds) bound it before
natural convergence**.
**Determined by evidence, not guessed:** `diminishing_floor` — Phase 1 measures where answering stops
improving the response (the improvement-vs-`value` curve); the floor is set from that, not assumed.
**Still to set:** research model · `max_rounds` cap · final deliverable (response, or response +
tombstone trail).
