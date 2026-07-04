# NBQ pre-registration — iteration two (#30 answerability, conditional full lap)

Filled instance of `preregistration-template.md`. Authored 2026-07-04 by the main loop BEFORE any
build, per house rules (formula frozen; feature off-by-default behind a selector; gate + adopt rule
fixed before the run; OBJECTIVE harness gates any elicitation change). Two staged items: **(A)** a
zero-model-call retro probe that gates whether **(B)** is worth building.

Baseline suite: 206 tests (`python3 tests/run.py` → "Ran 206 tests … OK"). SKILL.md version 1.3.4.

---

## ITEM A — Cheap retro probe (gate for #30)

### Hypothesis
- Candidate / backlog ID: candidate 1 (drift-corrected), gates #30.
- Hypothesis: tasks whose **kept high-EVSI questions were unanswerable** fail the objective outcome
  (`frac < 1`) more often than tasks whose high-EVSI questions were answerable.
- Prior closed experiment affected: #32 (first-order) pinned the residual P4 gap on *answerability*
  (nbq-firstorder unanswerable 77% > 50%), not candidate altitude. This probe tests whether that
  unanswerability is *causally* linked to objective failure before we spend a build on #30.
- Why re-opened: #30's re-open condition (post-#32 unanswerable > 50%) is met (77%).

### Expected mechanism
- Causal mechanism: an unanswerable high-EVSI question consumes the elicitation budget without
  reducing the assumption the solver must guess → the solver proceeds on a wrong default → hidden
  test fails.
- Observable consequence if correct: within the `nbq` arm of `~/.hermes/outcome_eval_32.json`, tasks
  with more/among-top unanswerable kept questions have lower `frac`.
- Cheapest falsifying observation: **zero new model calls.** The join fields already exist per
  task/arm in `outcome_eval_32.json`:
  - per-question EVSI = `meta.q_values[i]` (index-aligned, sorted desc);
  - per-question answerability = `qa[i].revealed` (bool) reinforced by `qa[i].answer` (the strict
    simulator's "The spec doesn't say." ⇒ unanswerable);
  - objective outcome = `frac` (fraction of hidden tests passed) / `per_test`.
- codex-worker FIRST verifies these fields are present and index-aligned (they are, per REVIEW). No
  `outcome_eval.py` emit change is required; if a field were missing the fallback (add a minimal
  annotation-emit + re-run smallest-n) would apply — it does not.

### Smoke test
- Command / procedure: run the probe helper against a durable copy of the JSON
  (`~/.hermes/outcome_eval_32.json` → durable named copy under `~/.hermes/`), print the 2×2 table +
  effect + SE.
- Pass condition: helper runs offline, reproduces the same numbers on re-run (deterministic).
- Stop condition: any join misalignment (len(questions) ≠ len(qa) ≠ len(q_values)) → stop, report.

### Targeted tests
- Tests: a unit test on the probe helper's pure math (2×2 counts, point-biserial, SE) with a small
  hand-built fixture where the answer is known.
- Inert-by-default pin: N/A for item A (analysis-only helper, touches no elicitation path).
- Required assertions: correlation sign + magnitude match the fixture; "unanswerable" derivation
  from `revealed`/`answer` matches expectation on both a revealed and a "spec doesn't say" row.

### Gate (arms, n, primary metric)
- Control arm: N/A (retro analysis of the existing `nbq` arm, n=34 tasks).
- Experimental arm(s): N/A.
- Paired sample and n: 34 tasks, `nbq` arm.
- Primary metric: association between "top-EVSI kept question unanswerable" (2×2 present? × objective
  fail?) and objective failure — point-biserial r (or 2×2 with the broad-win-style guard read as
  "effect clears its SE").
- Secondary diagnostics: report both framings — (i) **any** top-K kept question unanswerable × fail,
  and (ii) **the single highest-EVSI** question unanswerable × fail — plus the raw
  unanswerable-count↔frac point-biserial. Note base rates (unanswerable is very common: 2–3 of 3 in
  the spot check), which caps achievable variance; report the marginal cell counts so a near-constant
  predictor is visible as such rather than mistaken for signal.

### Mechanical GATE rule (fixed before running the probe)
- **#30 build PROCEEDS** iff the primary association is in the hypothesized direction
  (unanswerable ⇒ more failure) AND the effect clears its own SE (|effect| > SE), on ≥1 of the two
  primary framings, AND the marginal cells are not degenerate (both levels of the predictor occur on
  ≥ ~15% of tasks — otherwise the predictor is near-constant and the probe is uninformative, treated
  as falsified-for-build-purposes).
- **#30 is PARKED** (premise falsified for free) if: no association, wrong-direction association
  (unanswerable ⇒ *success*), OR a degenerate predictor. Lap ends at JOURNAL with a negative-result
  product; backlog re-scoped (candidate 2 nbq→relentless integration rises so a real answerability
  corpus can be built; #30 waits on it).

### Efficiency budget
- Δtokens / Δwall / Δcalls per run: **zero** (analysis of existing JSON; no model calls).
- Ceiling: N/A for the probe.

### Rollback
- N/A (analysis-only; no selector, no default touched).

---

## ITEM B — #30 answerability weighting (built ONLY if A's gate says PROCEED)

### Hypothesis
- Candidate / backlog ID: #30 answerability weighting.
- Hypothesis: down-weighting EVSI by a **non-self-rated** answerability estimate raises objective
  pass rate by steering elicitation toward answerable high-value questions, shrinking the 77%
  unanswerable rate.
- Prior closed experiment affected: the original #30 self-rated multiplier (inert at 0.95 in 15/16
  cells — a self-rated ceiling that already failed once).
- Why its old verdict no longer applies: the mechanism is now the **strict-simulator answer/refuse
  behavior**, NOT the ranker's self-report. Different signal, different failure mode.

### Expected mechanism
- Causal mechanism: a single **batched strict-simulator answerability probe** asks the strict
  simulator to answer-or-refuse each ranked candidate from the *visible spec only*; refusal ("spec
  doesn't say") ⇒ answerability≈0, concrete answer ⇒ answerability≈1. EVSI is multiplied by this
  weight so answerable high-value questions rise and unanswerable ones sink. Formula stays frozen —
  answerability enters as an explicit selector-gated post-multiplier, absent-key = 1.0 (identity), so
  the default ranking is byte-identical.
- Observable consequence if correct: with `--answerability on`, candidate re-ranking demotes the
  "spec doesn't say" questions; objective `frac` rises; unanswerable rate drops.
- Cheapest falsifying observation: the objective gate (below).

### Smoke test
- Command / procedure: run one task with `--answerability on`; show re-ranked candidates vs default.
- Pass condition: at least one previously-top unanswerable candidate is demoted; default run
  (no flag) unchanged byte-for-byte.
- Stop condition: default ranking changes with the flag off ⇒ stop (inert-by-default pin violated).

### Targeted tests
- Tests: (1) mechanism unit test — a fixture where a high-EVSI unanswerable candidate is demoted
  below a lower-EVSI answerable one after the weight applies; (2) **inert-by-default pin** — the
  default config never calls the answerability path and never changes any score (absent-key = 1.0,
  scores byte-identical to pre-change); (3) CLI/env plumbing parity (`--answerability on|off`,
  `INFOGAIN_ANSWERABILITY`) mirroring the `firstorder`/`reach` pattern at `infogain.py:_resolve_families`.
- Inert-by-default pin: REQUIRED — default cfg produces identical scores and makes zero added calls.
- Required assertions: dry-run parity; absent-key identity; selector precedence CLI > env > FAMILIES.

### Gate (arms, n, primary metric)
- Control arm: `nbq`.
- Experimental arm: `nbq-answerability`.
- Paired sample and n: both banks n=34; K=3; all-deepseek; `--strict-preflight`.
- Primary metric: paired Δpass vs `nbq`.
- Secondary diagnostics: unanswerable-rate drop; lens-payoff non-regression (json-migrate `.bak`
  class); the zeroshot gap; per-arm mean wall/tokens/calls (cost columns on).

### Mechanical ADOPT rule (frozen before build)
- Adopt (flip default on) exactly when: **Δpass > 0 AND wins ≥ 2× losses** (broad-win guard) AND
  unanswerable rate materially down AND no lens-payoff regression AND within the efficiency ceiling.
- Otherwise: **no-adopt**, or **adopt-with-knob-off-by-default** when the ONLY failure is the
  efficiency ceiling with a real result win (pre-declared disposition). Borderline / directional-only
  = no-adopt (#28 precedent).

### Efficiency budget
- Expected Δcalls per run: +1 (one batched answerability probe over the candidate set).
- Expected Δtokens per run: one batched prompt over ≤ candidate-set questions.
- Expected Δwall per run: one added round-trip.
- **Per-dimension ceilings that EACH independently veto even a result win** (a bust on ANY one is a
  ceiling failure — cost is not a single scalar):
  - **wall:** mean added wall ≤ 10% of an nbq run;
  - **tokens:** mean added tokens ≤ 15% of an nbq run;
  - **calls:** ≤ +1 added model call per run (a second probe call — e.g. per-candidate instead of
    batched — is itself a veto, independent of wall/tokens).
- If any ceiling is exceeded: adopt-with-knob-off-by-default IF Δpass is a broad win; else no-adopt.
- Rationale for per-dimension ceilings: a wall-neutral but token-heavy change (or one that adds a
  hidden second call) must not pass on wall alone; each cost axis is gated separately.

### Rollback (selector + flag)
- Selector: `answerability` in `FAMILIES` (default off), resolved in `_resolve_families`.
- Rollback flag / value: `--answerability off` / `INFOGAIN_ANSWERABILITY=off`.
- Absent-key behavior: weight = 1.0 (identity) — ranking byte-identical to pre-change.

### Journal stubs for BOTH outcomes
(Filled verbatim into `evsi-validation-findings.md` at JOURNAL time with real numbers.)

> **[#30 answerability] — ADOPTED.** Hypothesis: … Gate: paired Δpass vs nbq at n=34. Δresult: …
> Δcost: Δtokens …, Δwall …, Δcalls +1. Rule passed because Δpass>0 with wins≥2×losses, unanswerable
> down, no lens regression, within ceiling. Default flipped on. Evidence: …

> **[#30 answerability] — NO ADOPT.** Hypothesis: … Gate: paired Δpass vs nbq at n=34. Δresult: …
> Δcost: … The rule failed because … Feature stays off / removed because … Re-open only if …
