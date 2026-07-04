# next-best-questions — key learnings

A digest of everything this skill's research program established, for anyone picking it up cold.
`SKILL.md` is the operational doc (how to run it); this is *what we learned building and
validating it* — including the negative results, which are products here, not failures. Deep
detail lives in `references/` and `evals/README.md`; raw run data paths are noted per finding.

## 1. The algorithm, and why it's frozen

`value = √(U · EVSI)` with `EVSI = Σ_a P(a)·Δplan(a)·stakes(a)` and
`U = entropy(P) · (1 − derivable_prob)`; gate out when U≈0 or EVSI≈0; MMR + same-target collapse
for diversity; depth via an **evidence loop** (answer → fold back → resolved questions retire),
never via projected multi-step chains.

- **The architecture is where the 2024–26 literature converged** (greedy one-step beats lookahead
  head-to-head; utility weighting is ahead of most published systems; the derivability gate
  matches the ask-only-if-it-changes-the-response result). Survey with citations:
  `references/algorithm-review-2026-07.md`.
- **The formula out-tested the frontier's specific alternatives on its own domain.** Five
  elicitation experiments, powered and pre-registered, all closed below the adoption bar:
  - **#24 pairwise/Bradley-Terry elicitation** — worse on every realized target (Δρ −0.156).
  - **#26 sampled forced-choice P(a)** — a real-contrast null: moved P on 79% of pairs, didn't
    move the ranking (Δρ +0.010 fast / +0.058 deepseek). Calibration gains don't transfer to
    utility-weighted ranking.
  - **#27 solution-space Δplan** (Active Task Disambiguation style) — decisively worse
    (Δρ −0.343 / −0.369): deltas collapse to near-binary, and a strong model *recovers the
    granularity yet still loses* — the failure is the K-solutions framing, not model capability.
  - **#23 rank-relative selection** — every within-task selection policy ≈ size-matched random;
    the calibrated absolute floor wins by size adaptation.
  - **#28 behavior-Δ judge** — see §3; directionally right, below the broad-win bar.
- **Verdicts are instrument-robust.** Re-adjudicated under deepseek-v4-pro after an adversarial
  audit alleged the weak-judge confound: everything held; same-response judge agreement ρ 0.814;
  the within-task ρ≈0.3 ceiling did not lift under the deep judge — it lives in the task, not the
  instrument. (`references/assertion-audit-2026-07.md`, findings §Deepseek re-adjudication.)
- **U is NOT inert for ranking** (a retired early claim): ρ(U-only vs full ordering) ≈ 0.35–0.50.
  U reshapes the order and earns its place; it is also the gate (derivable→U→0 retires questions).

## 2. Derive-or-ask: a derivability claim is an experiment, not a fact

The Bayesian reframe (1.3.0): the projected answer distribution is the model's posterior. Peaked +
derivable ⇒ **not a valid question — evidence wearing a question mark**; spread + underivable ⇒ a
valid question whose weight is the expected impact of resolving genuine ignorance. So claims ≥0.6
are **tested by an actual derivation attempt**: success tombstones the answer into the working
evidence (later rounds re-plan against it); CANNOT_DERIVE falsifies the inflated claim and the
question re-enters ranking with honest uncertainty.

Learnings that made it work (all probed, all pinned in tests):
- **The derive prompt must be knowledge-inclusive** ("…or your own general knowledge"): 22% of
  candidates claim derivable ≥0.8, largely from parametric knowledge; a strict "from the prompt
  alone" wording fails them and floods buckets with questions the gate retires correctly today.
- **Hedges are failed derivations** ("the prompt does not specify…" is a non-answer wearing an
  answer's clothes — tombstoning it injects junk evidence).
- **Fabrication risk measured low** (0/12 on user-only/tool-only questions, both models) — the
  escape hatch works; visible report provenance is the remaining guard.
- **It pays objectively**: +0.067 → +0.183 Δpass end-to-end (5W/0L), question waste 82% → 44% —
  the single biggest end-to-end improvement any change produced.

## 3. The objective-outcome tier — the ground truth that bit

Everything before 2026-07-03 validated against an LLM-judged proxy (realized response-change ×
stakes). `evals/outcome_bank.py` + `evals/outcome_eval.py` added executable ground truth:
ambiguous tasks with hidden specs and hidden tests, a **strict simulated user** (answers only what
the spec resolves), paired arms, hidden-test pass rate. Verdicts (n=20 micro + 8 agentic script
tasks; `~/.hermes/outcome_full.json`, `outcome_28.json`):

- **The purpose is real (P3):** clarification objectively improves artifacts — +0.317 pass rate,
  10W/0L, p=0.002. First statistically significant objective result of the program.
- **The machinery is out-asked by a naive baseline (P4 failed, loudly):** one zero-shot call for
  "the K best questions" beat the full pipeline (+0.327 vs +0.105 combined). Root cause, found
  mechanically: **Δplan is judged as text-volume, not consequence** — "case-sensitive or not?"
  (a one-token fix that flips every hidden test) was gated at 0.21 while robustness boilerplate
  top-ranked. The realized proxy *shares that lens* (it measures response diff), which is exactly
  why six proxy datasets never saw it — a demonstrated instance of the audit's A10 warning.
- **#28 (behavior-Δ judge) fixed only part of it:** re-eliciting Δplan as behavior/outcome change
  tripled the agentic-tier benefit (+0.077→+0.219) but failed its pre-registered gate (paired
  6W/5L vs the required 2:1; unanswerable 65% vs ~60%). Zeroshot still led by +0.157. **The
  successor hypothesis: the residual gap is GENERATION altitude** — candidates skew away from
  first-order unknowns before any judge ranks them.
- **Prompt-vs-script is decided by neither:** the whole framework carried in one prompt
  (`prompt-evsi` arm) beat the script slightly and trailed zeroshot — the value model, not the
  orchestration vehicle, is the bottleneck.
- **The first question is the game (P5):** benefit saturates at K≈1 and no over-asking penalty
  appears through K=7 — the scarcity is asking the *right first question*, precisely where a
  volume-biased judge mis-spends the slot.
- **The proxy keeps qualified standing (P6):** the skill's own q_value predicts objective Δpass at
  ρ 0.432 — above the pre-registered keep-line, forever caveated by the volume-bias blindness.

## 4. Lenses: the gate proposes, the formula disposes

Families are **exposure only** — no family-level scoring; every question stands on its own merit,
so misfired lenses self-prune. That division of labor is repeatedly validated:

- **Premortem (#25)** hunts the `stakes` tail and is the top lens by realized regret (~1.6× other
  lenses on failure-surface tasks), with zero read-only pollution across two differently-
  confounded instruments — even when the gate misfires ("write a brief"), scoring prunes it.
- **Reach (#29, adopted)** hunts **answerability**: "does a *reachable* different point of view
  exist — container, SSH host, in-service execution, possibly *chained hops* — that turns an
  unknown into an observable?" Tier-1: kept exactly on access/systems tasks, zero elsewhere.
  Tier-2: realized regret 0.351 ≈ vantage's 0.362 — signal, not noise. The ranker surfaces the
  hop; the investigator executes it (each hop widens the trust surface — see ASPI,
  arXiv:2605.17324).
- **Gates read the RAW PROMPT as well as the framing** (1.2.1): framing is model-paraphrased, so
  hint verbs can vanish before a framing-only gate sees them. Hint lists stay verb-only — nouns
  re-open known false positives (gmail-triage's "email").

## 5. Methodology learnings (the meta-lessons)

- **Pre-register the gate, apply it mechanically.** Every experiment ships off-by-default behind a
  selector with an adopt rule written *before* the run. Five closed negative/partial; the defaults
  changed only when evidence cleared the bar (derive-or-ask, reach). Negative results are
  documented as products (`references/design-decisions.md`, `references/evsi-validation-findings.md`).
- **Test the instruments, not just the algorithm.** Real defects found in our own tooling, each
  fixed and pinned: a reasoning-channel judge silently nulling a whole run (→ model preflight,
  exit 2 before any rows); a vacuous CI criterion deciding acceptance at bucket=0; framing leaking
  its own JSON schema; a too-strict simulator refusing spec-resolvable compound questions;
  truncated solver replies with unclosed code fences reaching the interpreter raw.
- **Proxies inherit their maker's blindness.** The realized-change judge and the Δplan projection
  share the response-diff lens, so proxy validation can never catch a volume-bias defect. Any
  future elicitation change gates on the objective harness.
- **Judge/ablation verdicts are instrument-sensitive at the top.** Formula rankings among close
  ablations shuffle across judge models; only large separations transfer. Don't move a default on
  a within-noise ablation win.
- **A dumb baseline arm is mandatory.** The naive-zeroshot arm did more to sharpen this program in
  one run than three literature-motivated upgrades did — always include the cheapest competitor.
- **Test derivability-like self-reports by making the model perform them** (CLAMBER generalized:
  LLMs are unreliable self-judges — of ambiguity, of derivability, and of when to clarify).
- **A passing test does not prove its target branch ran.** Coverage/execution and assertion are
  different: dead code wired into a live path still needs a test that actually exercises that
  branch. Otherwise the test is misleading even while green.
- **`trace --count` without `--missing` is a tautology.** It reports every executed line while
  hiding every gap; pass `--missing` or a claimed "100% coverage" can be an illusion.
- **Archive honestly instead of theater-testing.** CLI/print drivers whose findings are already
  recorded are deliberately left untested, while their pure math helpers remain pinned. Do not add
  tests solely to inflate a coverage number.

## 6. Open, honestly

- **Generation altitude** (#28's successor, #32 — NO ADOPT, cost-aware gate 2026-07-04): injecting
  a naive "K best clarifying questions" call as round-1 candidates (`--firstorder`) moved the mean
  (+0.132 vs nbq +0.083 over baseline, n=34) — altitude has signal — but did **not** close the P4
  gap: `zeroshot` still won +0.274 (15W/1L, p=0.0005) at ~1/5 the wall and ~1/150 the tokens.
  Failed the adopt rule on the broad-win guard (6W/6L paired vs nbq), unanswerable (77%), a
  lens-payoff regression, AND the efficiency ceiling (+16.8% wall). Built, off-by-default. The
  remaining gap is **answerability**, not candidate altitude — first-order questions fish *more*.
  (`references/design-decisions.md` §First-order candidate source (#32).)
- **#30 answerability weighting** — **now re-opened** (its condition "post-#32 IF unanswerable > 50%"
  is met: 77%). Re-open only with a mechanism that isn't self-rated (the old multiplier was inert at
  0.95 in 15/16 cells; the self-rated ceiling already failed once).
- **Prompt distillation**: unblocked by the #32 no-adopt (the value model is not the gap), but
  now lower-priority than #30 — re-scope against whichever answerability mechanism lands.
- **Discrimination preflight (#33 — built, adopted as an opt-in instrument)**: `--strict-preflight`
  runs 8 forced-choice fixtures per model; `fast` 8/8, `deepseek` 8/8. Closes "a judge that answers
  but judges randomly still passes." Remaining never-tested: utility-weighted lookahead
  head-to-head; human ground truth behind the realized judge.
- **Every gate is now cost-aware** (#32 onward): arms report mean wall/tokens/calls, and a result
  win that busts the pre-registered efficiency budget is a no-adopt — see the `nbq-improve` loop (§7).

## 7. The standing improvement protocol

The documented self-improvement loop lives at
`skills/autonomous-ai-agents/nbq-improve/SKILL.md`: REVIEW → RESEARCH → PLAN → BUILD → EVALUATE →
JOURNAL → LOOP. Every future iteration is cost-aware: it measures Δtokens and Δwall alongside
Δresult, not result alone.

## Where everything lives

| doc | contents |
|---|---|
| `SKILL.md` | how to run it (flags, modes, lenses, derive-or-ask, evidence loop) |
| `references/methodology.md` | the EVSI math + sources |
| `references/design-decisions.md` | every decision with its evidence, incl. closed experiments |
| `references/evsi-validation-findings.md` | all validation runs and verdicts, proxy + objective |
| `references/algorithm-review-2026-07.md` | the literature survey and its outcome |
| `references/assertion-audit-2026-07.md` | the adversarial audit (deepseek-v4-pro) verbatim |
| `evals/README.md` | the harness suite, headline results, run recipes |
| `../nbq-improve/SKILL.md` | the standing, cost-aware improvement protocol |
