# Skill Revision Validation and Regression Rules

**Status**: design only — no live skill, validator, or pipeline config is touched by this
document. It specifies the rule a proposed skill revision must pass before it replaces the
current live version, in terms of the artifacts already designed:

- `docs/design/hermes-skill-ops-synthesis.md` — source-paper synthesis (SkillOps, Skill-V).
- `docs/design/skill-contract-schema.md` + `skill-contract.schema.json` — the contract shape,
  including `outcome_criteria.non_regression`, `evidence_scope.min_sample_per_bucket`, the
  three-tier (MAJOR/MINOR/PATCH) version-bump policy, and `lineage`/`retirement_state`.
- `docs/design/skill-utility-metrics.md` — the event log (`skill_invocation`, `skill_outcome`,
  `skill_rework`), the four-state outcome model, stratification by `(task_category,
  difficulty_bucket)`, Beta-Binomial shrinkage (`α=5`), per-stratum lift, aggregate `U(s)`, and
  the `N_scored(s) < 20 ⇒ provisional` flag.

This document does not re-derive any of those; it assumes they exist and specifies the one thing
none of them pins down: **the pass/fail/review decision for a specific candidate revision**, with
concrete numbers.

---

## 0. Why a single document for all four questions

The four questions in the task body are not independent: "what counts as regression" (q2) is the
threshold the decision rule (q3) is built from, "replay" (q1) is what produces the numbers that
feed that threshold, and "rollback" (q4) is the same regression test run again after promotion
instead of before. One shared definition is used throughout so that pre-promotion and
post-promotion checks cannot silently diverge.

---

## 1. What "replay against past tasks" means concretely

Hermes has two structurally different kinds of revision, and "replay" means something different
for each. Both are grounded in the event log from `skill-utility-metrics.md` §1 — there is no
second logging system to build.

### 1.1 Rubric-only revision (`rubric_version` bump, `skill_version` unchanged)

This is the cheap, fully Skill-V-shaped case. `outcome_criteria` changed but the procedure that
produced past trajectories did not, so nothing needs to be re-executed — only *re-scored*.

**Source of trajectories**: the skill's own historical `skill_invocation` → `skill_outcome` →
(optional) `skill_rework` event chain, scoped to `skill_id` and the *current* `skill_version`
(not older procedure versions — their behavior is no longer representative; see §6 lineage
note in the utility-metrics doc). Each retained trajectory carries its stored evidence (the
`evidence_matcher` results already computed at resolution time, e.g. `exit_code_zero` logs,
`golden_trace_replay` captures, `llm_judge_rubric` transcripts) so re-scoring is a pure function
of stored data — no re-execution, no re-running the skill.

**Procedure**: for each retained trajectory, compute `ŷ_old = pass/fail under the live
outcome_criteria` and `ŷ_new = pass/fail under the candidate outcome_criteria`, then compare each
against the trajectory's actual resolved `outcome_status` (`success` counts as the ground-truth
positive; `failure`/`reworked` count as the ground-truth negative, per §1.4 of the utility-metrics
doc). This reproduces Skill-V's Evidence Gate inputs (`Disc`, `FNR`, `BAcc`) exactly, because
Hermes already stores the ground truth (`outcome_status`) separately from the rubric prediction.

### 1.2 Procedure revision (`skill_version` bump, with or without a `rubric_version` bump)

Here the thing that changed is the thing that *produces* the trajectory, so re-scoring old
trajectories under a new rubric proves nothing about whether the new procedure behaves
differently. Concrete replay for Hermes is **shadow execution**: re-run the candidate procedure,
in a sandboxed (non-live, not affecting the real task) context, against a sample of past task
*inputs* (not past outputs), and score the resulting new trajectories with the current (or
candidate, if also changed) `outcome_criteria` using the same deterministic `evidence_matcher`
kinds the contract already requires (`exit_code_zero`, `exact_match`, `regex_match`,
`file_hash_equals`, `schema_validates`, `golden_trace_replay`, `external_command_assert`,
`llm_judge_rubric` with pinned model/prompt/threshold). This is possible specifically *because*
the schema design in `skill-contract-schema.md` requires every outcome criterion to be
mechanically re-runnable — if a skill's criteria were all free-text judgment calls, shadow replay
would not be well-defined, which is itself an argument for why that schema constraint matters
here, not just for live scoring.

The comparison is then: new-procedure shadow outcomes (fresh, same inputs) vs. old-procedure
historical outcomes (the actual recorded outcomes for those same task inputs, since they already
ran live). Where the same task input is used for both, this is a **paired** comparison; where a
matched past input is unavailable (e.g. the input no longer exists or depends on now-stale
external state), the comparison falls back to **unpaired**, stratum-matched comparison against
the historical rate in the same `(task_category, difficulty_bucket)` stratum. Which mode applied
is logged per replayed item (`replay_mode: paired | unpaired`) because the statistical test in
§2 differs between the two.

### 1.3 Which tasks, how many, how selected

| Parameter | Value | Source / justification |
|---|---|---|
| Candidate pool | All trajectories for `skill_id` at the *current live* `skill_version`, within the trailing window already defined in `skill-utility-metrics.md` §4.5 (last 90 days or last 200 invocations, whichever is larger) | Reuses the existing window definition instead of inventing a second one; keeps replay evidence as fresh as the utility score it will be compared against. |
| Sampling method | **Stratified sampling by `(task_category, difficulty_bucket)` with a per-stratum floor** | Matches the stratification already defined in `skill-utility-metrics.md` §3 — the same confounding problem (a skill that's mostly used on easy tasks looks fine on average) applies to regression testing, not just to ranking. A revision that silently breaks the `complex` bucket while leaving `trivial` untouched must not be averaged away. |
| Per-bucket sample floor | `max(contract's evidence_scope.min_sample_per_bucket, 10)` | The contract schema already lets each skill author declare a bucket-specific minimum (examples in `skill-contract-schema.md` use 6, 8, 10). This design adds a **board-wide floor of 10**, because at `n<10` per arm a one-sided two-proportion test has too little power to distinguish a real 5-point regression from noise at any usable significance level (see §2.2) — a skill author declaring `min_sample_per_bucket: 6` is setting a *scoring* floor (per the contract doc), not a *regression-testing* floor; the two are allowed to differ, and the higher one governs for a pass/fail decision. |
| Total replay buffer cap | The skill's own declared `non_regression.replay_buffer_size` (examples: 20, 32, 40) | Ports the number straight from the contract rather than inventing a second cap, and bounds compute/LLM-judge cost per Skill-V's and SkillOps' shared "near-zero library-time cost" posture (synthesis §f) — `external_command_assert` and `llm_judge_rubric` checks are not free, and an unbounded replay set would make every revision review expensive in proportion to the skill's total lifetime traffic. |
| Allocation within the cap | Proportional to each bucket's share of recent traffic, **after** guaranteeing the per-bucket floor | Ensures every bucket the skill claims to serve (`applicability.scope.task_types`) gets tested at all, rather than a high-traffic `standard` bucket crowding out a rare-but-claimed `research` bucket entirely. |
| Buckets with zero history (e.g. newly added `task_types` in the candidate's `applicability.scope`) | Cannot be replayed — logged as `evidence_scope: none` for that bucket | Routed to human review (§3), not silently skipped — see §3.3. |

**Reproducibility**: the specific sampled trajectory IDs for a given revision review are recorded
(`replay_set_id` → list of `invocation_id`s) so a human reviewer or a later audit can re-derive
exactly what was tested, matching the lineage discipline (`evidence_case_ids`) already required
by `skill-contract-schema.md` §`lineage`.

---

## 2. The definition of "regression"

Regression is defined **per bucket**, then rolled up. A revision regresses on bucket `k` iff
**both** of the following hold simultaneously — effect size alone or significance alone is not
sufficient:

### 2.1 Practical-significance floor (effect size)

| Metric | Regression floor | Justification |
|---|---|---|
| Net success rate (`net_success/(net_success+net_failure)`, bucket `k`) or `lift_k(s)` | drop of **≥ 5 percentage points** | The worked example in `skill-utility-metrics.md` §5 demonstrates genuinely meaningful lift differences in the 4–13 point range (`lift_easy(A) = −0.044`, `lift_hard(B) = +0.133`). Setting the floor at 5pp sits just above the smallest *real* effect shown there, so the gate is sensitive to effects of the size that document already treats as decision-relevant, without firing on sub-percentage-point wobble. |
| Rework rate (`Rw(s)/N_scored(s)`, bucket `k`), tracked **in addition to** net success even though rework already counts as failure in §1.4 of the utility-metrics doc | rise of **≥ 3 percentage points** | Rework is explicitly the "looks fine at first, generates hidden work later" case (`skill-utility-metrics.md` §1.3) — the case a pure net-success check is slower to catch, because a reworked outcome initially resolves as `success` and only flips to `reworked` after the rework-detection window (up to 30 days). A tighter floor here catches a revision that is quietly shifting failures into "accepted but needs fixing" rather than outright rejection, which would otherwise dilute into the general net-success number and take longer to surface. |
| Hard constraints (`outcome_criteria.hard_constraints`) | **any single violation, n = 1** | Not a statistical threshold — hard constraints are Skill-V's `H(τ)=0` binary gate (Eq. 3), carried into the Hermes contract schema unchanged. A revision that violates even one hard constraint in replay fails outright regardless of sample size or significance; averaging a safety/correctness invariant against a rate is a category error for this class of check. |

### 2.2 Statistical-significance requirement

A drop meeting §2.1's floor must also be **statistically significant at α = 0.05** (one-sided —
the hypothesis under test is specifically "the candidate is worse," not "the candidate differs"):

- **Paired replay** (§1.1 rubric-only case; §1.2 procedure case where matched inputs exist):
  **McNemar's exact test** on the discordant pairs (old=pass/new=fail vs. old=fail/new=pass),
  one-sided, α = 0.05. Paired tests are used wherever possible because they have substantially
  more power than unpaired tests at the same `n` — the same task input is being judged twice, so
  within-task variance cancels.
- **Unpaired replay** (§1.2 procedure case, no matched input available): one-sided two-proportion
  test — Fisher's exact test when any cell count < 5, otherwise a two-proportion z-test — at
  α = 0.05, comparing the shadow-execution success rate to the historical stratum rate.
- **α = 0.05** is the conventional default significance level; it is not loosened here because a
  false "pass" (shipping a real regression) is asymmetric in cost with a false "fail" (an
  unnecessary human-review routing, §3.3) — the design errs toward the cheaper mistake.

**Why both floor and test, not either alone**: at the board-wide sample floor of `n = 10` per
bucket (§1.3), the standard error of a proportion difference is large enough (≈0.20 at `p≈0.7`)
that a raw 5-point drop is well within noise — the significance test exists precisely so the
floor doesn't fire on small-sample noise. Conversely, significance alone (without a floor) would
flag trivial half-a-point differences as "significant" once a skill accumulates enough replay
history, which is not an actionable regression — this is why `skill-utility-metrics.md` itself
warns that raw rates are unstable at low `n` (§4.1) and why both constraints are required jointly.

### 2.3 Bucket-level vs. revision-level regression

The revision **regresses** if §2.1+§2.2 both fire on **any one** bucket with sufficient sample
(§1.3's floor). This is deliberately not averaged across buckets the way `U(s)` aggregates lift
for *ranking* (`skill-utility-metrics.md` §4.3) — a revision that improves the common `standard`
bucket while quietly breaking the rare `research` bucket must not pass just because its
invocation-weighted average looks fine; non-regression is a per-claim guarantee ("every bucket
this skill claims to serve is still at least as good"), not an aggregate-utility comparison.

---

## 3. The decision rule

### 3.1 PASS — auto-promote

All of the following must hold:

1. Candidate validates against `skill-contract.schema.json`.
2. `applicability.protected_constraints` is byte-for-byte unchanged from the live version.
3. Zero hard-constraint violations across the entire replay set (§2.1).
4. No bucket with `n_k ≥` the §1.3 floor shows a regression per §2.1+§2.2 (neither net-success/lift
   nor rework-rate).
5. At least one bucket actually met the sample floor (a revision with *zero* adequately-sampled
   buckets has not been tested at all — see §3.3, it cannot pass by default).
6. The resulting version bump classifies as MINOR or PATCH under the existing policy in
   `skill-contract-schema.md` (`"Procedure change that passes non_regression... behavior changed
   but evidence shows it is not worse"` — this section *is* that evidence).

On PASS: the candidate replaces the live contract in place, `lineage.previous_version` is set to
the version being replaced (never deleted — see §4), `lineage.evidence_case_ids` records the
`replay_set_id` from §1.3, and the new version's utility counters start at zero per
`skill-utility-metrics.md` §6 (it is judged only on its own future evidence, not inherited score).

### 3.2 FAIL — reject, current version stays live

Either of:

1. Any hard-constraint violation anywhere in the replay set (§2.1, binary — no vote needed), or
2. A statistically-and-practically-significant regression (§2.1+§2.2) on any adequately-sampled
   bucket.

On FAIL: the current live version is untouched (matches Skill-V §3.3: "rejected ones leave the
current version untouched"). The rejection, the specific bucket(s) and metric(s) that triggered
it, and the replay evidence are logged against the candidate (not silently discarded) so a future
resubmission of a fixed candidate can be compared against this attempt's failure mode.

### 3.3 NEEDS HUMAN REVIEW — routed to a reviewer with the gate's output as a decision aid

Any of the following routes to a human instead of an automated pass/fail, because the synthesis
note (§f) is explicit that neither source paper has a human-in-the-loop step and recommends
exactly this combination — automated gate as decision aid, human for the cases the gate cannot
resolve on its own:

1. **`protected_constraints` changed.** Always MAJOR (per the existing bump table), and the
   contract schema already reserves `lineage.reviewed_by` for exactly this case. A revision that
   changes a safety/scope invariant is, by construction, not something past replay evidence can
   validate (the invariant itself is what's changing), so there is no automated substitute.
2. **A `hard_constraints` entry's `evidence_matcher` changed (added, removed, or redefined).**
   This redefines what "pass" means, not just what procedure is scored — same reasoning as (1):
   the measuring instrument is changing, so past measurements under the old instrument cannot
   certify the new one.
3. **`N_scored` for the replay set (summed across all buckets) is `< 20`.** This reuses the exact
   `provisional` threshold already defined in `skill-utility-metrics.md` §4.4. Below it, neither a
   PASS nor a FAIL verdict is trustworthy — the gate has not seen enough evidence either way, and
   auto-passing "because nothing failed yet" is indistinguishable from "we didn't look hard
   enough."
4. **A bucket in the candidate's claimed scope has zero replay history** (e.g. `applicability.
   scope.task_types` was widened to a task type with no prior invocations at all). There is
   nothing to replay against for that claim; shipping on an unevidenced scope expansion is exactly
   the rubber-stamping risk the synthesis note flags (§f).
5. **Borderline statistical result**: an observed drop that meets the §2.1 effect-size floor but
   whose one-sided p-value falls in `(0.05, 0.20)` — i.e., "suggestive, not conclusive" given the
   available sample. Rather than forcing a binary call on an underpowered test, this is surfaced
   to a human with the actual numbers (the same information the automated gate used) rather than
   auto-passing it as "not significant" or auto-failing it as "probably bad."
6. **A new or materially changed `llm_judge_rubric` criterion is part of the candidate**, on its
   *first* promotion only. Pinning `model`/`prompt_ref`/`threshold` (per `skill-contract-schema.md`
   §`outcome_criteria`) makes it re-runnable and auditable, but it is still the one criterion kind
   that is not a pure mechanical check — a one-time human spot-check on introduction (not on every
   subsequent revision of an already-reviewed criterion) catches a miscalibrated rubric before it
   becomes the automated ground truth for every future gate run on that criterion.

A human-review routing is not a terminal state: the reviewer either approves (equivalent to PASS,
with `lineage.reviewed_by` recorded per the schema), rejects (equivalent to FAIL, same logging),
or sends the candidate back for rework with specific feedback — mirroring the kanban
`kanban_request_changes` pattern already used elsewhere in this project, so review is a cycle, not
a one-shot gate.

---

## 4. Rollback behavior after a revision regresses live

Passing the pre-promotion gate (§3.1) does not guarantee production behavior matches the replay
sample — the replay sample is necessarily historical and finite. Live monitoring is the backstop.

### 4.1 What triggers a rollback

Once the newly-promoted version has accumulated `N_scored(new_version) ≥ 20` (the same provisional
threshold as §3.3(3) — a live version is not trusted to rule on itself before that point either),
every new resolved outcome re-evaluates the **same regression test as §2** (effect-size floor +
significance test), comparing the new version's trailing-window rate per bucket against the
*immediately-prior* version's trailing-window rate in the same bucket (sourced from the historical
event log, not a frozen snapshot — the prior version's own numbers don't move once it's retired,
since its invocation counter stops incrementing). Check cadence is event-driven but rate-limited
to at most once per day per skill to avoid alarm-flapping on every single new resolution (a
config detail, not a design gap — the trigger condition itself is exact).

### 4.2 What happens on trigger: automatic, not human-gated

Unlike the pre-promotion ambiguous cases (§3.3), a live-regression trigger causes an **automatic
rollback**, not a human-review queue:

- The "active" pointer for `skill_id` reverts to `lineage.previous_version` immediately.
- This is possible with zero downtime specifically because `retirement_state` is archive-first
  (`skill-contract-schema.md` §`retirement_state` — `active → candidate_for_retirement →
  archived`, never deleted): the previous version's contract and procedure were never removed, so
  "rollback" is a pointer swap, not a restore-from-backup.
- A human is still notified (the triggering evidence — bucket, metric, effect size, p-value — is
  attached to the notification) but the rollback is not blocked on them acting.

**Why automatic here but human-gated pre-promotion**: by the time a live-regression trigger fires,
the revision already passed the pre-promotion gate once, on real replay evidence (§3.1). A live
trigger firing anyway is high-confidence evidence that the pre-promotion sample wasn't
representative of production reality — not an ambiguous edge case needing judgment, but the same
kind of signal the gate was already designed to act on, just arriving late. Queuing it for human
review would let a confirmed-live regression keep accumulating real task failures while waiting on
a person; reverting first and explaining second is the conservative default given that cost
asymmetry.

### 4.3 Recording the rollback

The rolled-back version's `lineage` gains `rolled_back_at`, `rollback_reason` (the specific
metric/bucket/numbers that triggered it), and `rollback_evidence_ids` (the `invocation_id`s that
constituted the live regression). Its `retirement_state.status` becomes `archived` with an
explicit `rollback: true` flag, distinguishing "archived because it regressed in production" from
an ordinary deliberate archive — this distinction matters for anyone later reviewing a skill's
history, and for the repeated-rollback guard in §4.4. A rolled-back version is not given a
fast-path back to live; resubmitting it (with a fix) goes through the full §3 gate again like any
other candidate.

### 4.4 Repeated-rollback guard

If the same `skill_id` accumulates **≥ 2 rollbacks within a rolling 30-day window**, every future
candidate revision for that `skill_id` is routed to mandatory human review (§3.3-style) regardless
of what the automated gate says, until a human explicitly clears the flag. This bounds a skill
that keeps getting promoted, regressing, and rolling back from cycling indefinitely on pure
automation — the same reasoning as Hermes's own kanban unblock-loop escalation (repeated
block/unblock on the same card escalates to triage rather than looping forever), applied to the
skill-revision lifecycle instead of the task lifecycle.

---

## 5. Threshold summary table

| Parameter | Value | Where justified |
|---|---|---|
| Per-bucket replay sample floor | `max(contract's min_sample_per_bucket, 10)` | §1.3 |
| Total replay buffer cap | Skill's declared `non_regression.replay_buffer_size` (20–40 in examples) | §1.3 |
| Minimum total `N_scored` before any pass/fail verdict is trusted | 20 | §3.3(3), reused in §4.1 |
| Net-success / lift regression floor | ≥ 5 percentage points | §2.1 |
| Rework-rate regression floor | ≥ 3 percentage points | §2.1 |
| Hard-constraint violation tolerance | 0 (any single violation fails) | §2.1 |
| Statistical significance level | α = 0.05, one-sided | §2.2 |
| Ambiguous-zone p-value band (routes to human) | (0.05, 0.20) | §3.3(5) |
| Live-rollback check start threshold | `N_scored(new_version) ≥ 20` | §4.1 |
| Repeated-rollback escalation threshold | ≥ 2 rollbacks / 30-day rolling window | §4.4 |

---

## 6. Explicit non-goals

Consistent with the scope boundaries already set in the three input documents, this design does
**not** specify:

- The exact sandboxing mechanism for shadow execution (§1.2) — containerization, replay harness,
  cost controls on `external_command_assert`/`llm_judge_rubric` calls during shadow runs. This is
  an implementation detail of "how replay actually executes," not the validation rule itself.
- Multiple-comparisons correction across buckets (e.g. Bonferroni) when a revision touches many
  buckets at once — flagged here as a known refinement (testing N buckets at α=0.05 each inflates
  the family-wise false-positive rate), left as a tuning decision for whoever implements this,
  since the right correction depends on how many buckets skills typically span in practice, which
  isn't known yet.
- The notification/UI mechanism for routing a candidate to human review (§3.3) or notifying a
  human of an automatic rollback (§4.2) — this assumes some such mechanism exists (e.g. a kanban
  task, per this project's own conventions) but doesn't design it.
- Any change to retirement policy (distinct from rollback) — `skill-contract-schema.md` already
  scopes `retirement_state` as a separate design item; §4's `archived`+`rollback:true` state reuses
  that field but does not redefine the broader retirement policy.
