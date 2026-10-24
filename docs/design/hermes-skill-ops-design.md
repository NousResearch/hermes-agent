# Hermes Skill Ops Design

**Status**: design only. This document and everything it links to is documentation under
`docs/design/`; **no live skill files, skill contracts, telemetry code, validators, schedulers, or
Hermes configuration were created, modified, or deleted by this work.** Verified at the bottom of
this document (§6).

**Purpose**: this is the top-level index for Hermes's skill-lifecycle design — how a skill's
contract is defined, how its utility is measured, how a revision is validated before and after it
goes live, how a skill is eventually archived or promoted, and when a human must approve a change
versus when the system may act alone. Each concern has its own detailed sub-document; this one
summarizes each, links out, consolidates every numeric threshold into one table, and collects
every open question the sub-documents deferred.

**Source material**: `docs/design/hermes-skill-ops-synthesis.md` — a read of two source papers,
SkillOps (arXiv:2605.13716) and Skill-V (arXiv:2610.11781), establishing the gaps (no difficulty
normalization, no sample-size floor, no lineage ledger, no human-in-the-loop step) that the five
documents below were written to close.

---

## 1. Document map

| Document | Answers |
|---|---|
| [`skill-contract-schema.md`](./skill-contract-schema.md) + [`skill-contract.schema.json`](./skill-contract.schema.json) | What is a skill, structurally? What fields does a contract have, and what triggers a MAJOR/MINOR/PATCH version bump? |
| [`skill-utility-metrics.md`](./skill-utility-metrics.md) | How is a skill's value measured, correcting for the fact skills aren't invoked on equally-hard tasks? |
| [`skill-validation-rules.md`](./skill-validation-rules.md) | When does a proposed revision pass automatically, fail automatically, or need a human? What triggers an automatic live rollback? |
| [`skill-forgetting-and-lineage.md`](./skill-forgetting-and-lineage.md) | When does a skill get archived, promoted, or flagged disused? How is every change recorded so it can be audited and its eventual outcome checked? |
| [`skill-human-review.md`](./skill-human-review.md) | Which of the above transitions require a human, and exactly what must a human reviewer be shown so review is substantive, not a rubber stamp? |

Reading order for a new reader: contract schema first (defines the vocabulary everything else
uses), then utility metrics (defines the score), then validation rules and forgetting/lineage in
either order (both consume the score), then human review last (it is the policy layer that sits on
top of all four).

---

## 2. Summary of each design

### 2.1 Contract schema

A skill contract is a versioned, JSON-Schema-validated document (draft 2020-12) with six top-level
sections: `preconditions` (mechanically checkable, zero-LLM-call gating), `applicability`
(`scope` + `protected_constraints` — safety invariants a revision may never silently break),
`outcome_criteria` (`hard_constraints` + `weighted_criteria`, each with a closed-enum
`evidence_matcher.kind` so every pass/fail is reproducible, plus `evidence_scope` for
difficulty-bucketed sample floors and `non_regression` for the replay-buffer size), `lineage`
(`previous_version`, `update_reason`, `evidence_case_ids`, `reviewed_by`), and `retirement_state`
(`active | candidate_for_retirement | archived`, archive-first — nothing is ever deleted). Version
numbers (`skill_version`, `rubric_version`) bump independently under a three-tier SemVer policy:
MAJOR for a protected-constraint or scope change, MINOR/PATCH for anything that passes the
non-regression gate.

### 2.2 Utility metrics

Three logged counts per invocation — `N(s)` (invoked), `S(s)`/`Fl(s)` (success/failure), `Rw(s)`
(rework discovered after initial acceptance) — are combined into a single score `U(s)` that
corrects for task-difficulty confounding: invocations are stratified by `(task_category,
difficulty_bucket)`, each stratum's rate is shrunk toward a leave-one-out baseline via a
Beta-Binomial posterior (`α = 5`), the shrunk rate minus baseline gives per-stratum `lift_k(s)`,
and `U(s)` is the invocation-weighted mean of lift across strata. A worked example in that document
shows this ranking *flipping* relative to naive pooled success rate once difficulty is accounted
for. Scores with `N_scored(s) < 20` are flagged `provisional`.

### 2.3 Validation rules

A proposed revision is replayed — re-scored against stored trajectories for a rubric-only change,
or shadow-executed against sampled past task inputs for a procedure change — using stratified
sampling with a board-wide per-bucket floor of `max(contract's min_sample_per_bucket, 10)`. A
revision regresses a bucket only if *both* an effect-size floor (≥5pp net-success/lift drop, ≥3pp
rework-rate rise, or any hard-constraint violation) *and* statistical significance (α=0.05,
one-sided, McNemar for paired replay / two-proportion or Fisher for unpaired) fire together on that
bucket — never averaged across buckets. The resulting verdict is PASS (auto-promote), FAIL
(auto-reject, current version stays live), or NEEDS HUMAN REVIEW (six specific triggers: changed
protected constraints, redefined hard constraints, insufficient sample, unevidenced new scope,
borderline p-value in (0.05, 0.20), or a first-time `llm_judge_rubric` criterion). After promotion,
live monitoring re-runs the same regression test; a live-confirmed regression triggers an
*automatic* (not human-gated) rollback to the prior version, since the archive-first retirement
design makes that a zero-downtime pointer swap. Two rollbacks in 30 days escalates a skill to
mandatory human review for every future candidate.

### 2.4 Forgetting and lineage

An archive/keep/promote decision requires `N_scored ≥ 20` *and* the same-direction `±0.05`-lift
band to be hit on 3 consecutive, non-overlapping 20-sample batches (not a single trailing window) —
math worked out from the shrinkage SE to give a ~1.4% false-trigger rate. A separate disuse trigger
(180 days, zero invocations) needs no sample floor since it's a fact, not an estimate. Archiving is
status-only (no version bump) and is explicitly reversible — `archived` is never `deleted` —
though unarchiving always requires a human, since an archived skill accrues no new evidence that
could justify an automated un-regression. Promotion is modeled as an orthogonal, additive
`promotion_state` field (proposed, not yet in the shipped schema) and self-corrects via automatic
demotion. Every lineage-affecting event is recorded in an append-only `skill_lineage_event` ledger
— proposal reference, frozen evidence snapshot, decision-rule version, approver, rationale, and an
`outcome_status` that a scheduled 60-day check closes out (`confirmed_beneficial` /
`confirmed_harmful` / `reverted` / `superseded`) — which joins exactly onto the contract schema's
`version`/`lineage`/`retirement_state` fields.

### 2.5 Human review

Twelve change categories are enumerated and assigned human-required or auto-apply (§2 of that
document); only four categories are auto-apply — a clean automated PASS, a candidate flag firing
(not the terminal archive/promote transition itself), an automated promotion, and an automatic live
rollback — because each of those four either has fully mechanical evidence, no standalone
consequence, or a proven self-correcting downstream safety net. Everything else (new skills,
archiving, unarchiving, protected-constraint changes, repeated-rollback escalation, and any
NEEDS-HUMAN-REVIEW-routed revision) requires a human. A 15-field required checklist (full contract
diff, per-bucket replay table with raw numbers, the 3-batch sustained-trigger sequence where
relevant, prior lineage history, and an explicit approve/reject/request-changes action with a
mandatory substantive comment) is the structural anti-rubber-stamp mechanism — a review that omits
any field cannot be submitted. A concrete CLI mockup (`hermes skill review list/show/decide`)
illustrates the presentation.

---

## 3. Consolidated numeric thresholds

| Constant | Value | Defined in | Used by |
|---|---|---|---|
| `MIN_SAMPLE_FOR_DECISION` / `N_scored < 20 ⇒ provisional` | 20 scored invocations | `skill-utility-metrics.md` §4.4 | Utility score display; validation NEEDS-REVIEW trigger (#3); archive/promote decision gate; live-rollback check start |
| Shrinkage pseudo-count `α` | 5 | `skill-utility-metrics.md` §4.1 | `U(s)` computation |
| Measurement/trailing window | 90 days or last 200 invocations, whichever is larger | `skill-utility-metrics.md` §4.5 | Utility scoring; replay candidate pool |
| Outcome resolution window | 14 days (configurable) | `skill-utility-metrics.md` §1.2 | Unresolved invocations finalize as `unknown` |
| Rework-detection window | 30 days (configurable) | `skill-utility-metrics.md` §1.3 | Attribution of late-discovered rework to an invocation |
| Minimum baseline population | 10 invocations | `skill-utility-metrics.md` §3.3 | Falls back to broader scope (category → global) below this |
| Per-bucket replay sample floor | `max(contract's min_sample_per_bucket, 10)` | `skill-validation-rules.md` §1.3 | Pre-promotion replay/regression test |
| Total replay buffer cap | Skill's declared `non_regression.replay_buffer_size` (20–40 in worked examples) | `skill-validation-rules.md` §1.3 / `skill-contract-schema.md` | Bounds total replay cost per revision |
| Net-success/lift regression floor | ≥ 5 percentage points | `skill-validation-rules.md` §2.1 | Regression test effect-size condition |
| Rework-rate regression floor | ≥ 3 percentage points | `skill-validation-rules.md` §2.1 | Regression test effect-size condition |
| Hard-constraint violation tolerance | 0 (any single violation fails) | `skill-validation-rules.md` §2.1 | Regression test, binary |
| Statistical significance level | α = 0.05, one-sided | `skill-validation-rules.md` §2.2 | Regression test (McNemar / two-proportion / Fisher) |
| Ambiguous p-value band (routes to human) | (0.05, 0.20) | `skill-validation-rules.md` §3.3(5) | NEEDS-REVIEW trigger (#5) |
| Live-rollback check start | `N_scored(new_version) ≥ 20` | `skill-validation-rules.md` §4.1 | Gate before live-monitoring regression re-check begins |
| Repeated-rollback escalation | ≥ 2 rollbacks / rolling 30 days | `skill-validation-rules.md` §4.4 | Forces mandatory human review on all future candidates for that skill |
| Archive/promote lift band | `U(s) ≤ −0.05` (archive) / `U(s) ≥ +0.05` (promote) | `skill-forgetting-and-lineage.md` §3.1 | Candidate-flag trigger condition |
| Sustained-trigger requirement | 3 consecutive, non-overlapping, ≥20-sample batches, same direction | `skill-forgetting-and-lineage.md` §3.2 | Required before an archive/promote candidate flag becomes actionable |
| Single-batch false-trigger rate (implied) | ≈ 24% (`Φ(−0.05/0.072)`) | `skill-forgetting-and-lineage.md` §3.2 | Justification for requiring 3 consecutive batches, not 1 |
| 3-consecutive-batch false-trigger rate | ≈ 1.4% | `skill-forgetting-and-lineage.md` §3.2 | Accepted false-archive/false-promote rate per candidate-skill-version |
| Disuse trigger | 0 invocations over trailing 180 days | `skill-forgetting-and-lineage.md` §3.3 | Candidate-for-retirement flag, independent of sample-size gate |
| Outcome-closure check | 60 days after any lineage event | `skill-forgetting-and-lineage.md` §7.4 | Sets `outcome_status` (`confirmed_beneficial`/`confirmed_harmful`/`reverted`/`pending`) |
| Required review-comment presence | non-empty, substantive (exact enforcement mechanism open — see §4) | `skill-human-review.md` §4.1 | CLI refuses `--approve`/`--reject` without it |
| Human-required categories (count) | 8 of 12 | `skill-human-review.md` §2 | Final approval-routing policy |
| Auto-apply categories (count) | 4 of 12 (clean PASS, candidate flagging, promotion, live rollback) | `skill-human-review.md` §2 | Final approval-routing policy |

---

## 4. Open questions / unresolved tradeoffs (consolidated from all five sub-documents)

### From `skill-contract-schema.md`
- The exact algorithm for computing `disc_rate` / `false_negative_rate` / `balanced_accuracy` —
  field names are fixed by the schema, the computation itself is left to implementation.
- Cross-skill graph relations (SkillOps' `dep`/`comp`/`red`/`alt` edges) are out of scope for this
  per-skill contract; would need a separate "skill graph" schema if composability checks are wanted
  later.

### From `skill-utility-metrics.md`
- The proxy-difficulty weights (`w1..w4` for when no explicit difficulty tag exists) are a
  config/tuning detail, not resolved here.
- Whether/how `severity` (`minor`/`major` rework) should weight the rework penalty differently —
  v1 treats all rework as equally failing; a severity-weighted v2 needs real rework data to
  calibrate against first.
- Live telemetry collection, storage schema migrations, and the actual event-logging code are
  specified as a target shape, not implemented.

### From `skill-validation-rules.md`
- The exact sandboxing mechanism for shadow execution (containerization, replay harness, cost
  controls on `external_command_assert`/`llm_judge_rubric` calls during shadow runs) is left to
  whoever implements replay.
- Multiple-comparisons correction across buckets (e.g. Bonferroni) when a revision touches many
  buckets at once is flagged as a known gap — testing N buckets at α=0.05 each inflates the
  family-wise false-positive rate, but the right correction depends on how many buckets skills
  typically span, which isn't known yet.
- The notification/UI mechanism for routing to human review or notifying of an automatic rollback
  assumes *some* mechanism exists (e.g. a kanban task) but doesn't design it — partially answered
  by `skill-human-review.md`'s CLI mockup, but the actual notification transport (email, Slack,
  kanban task creation, etc.) is still unspecified.

### From `skill-forgetting-and-lineage.md`
- The storage technology for the `skill_lineage_event` ledger (SQL table vs. event log vs.
  something else) is a schema-level spec only.
- The exact format of the "evaluator job-run id" used in `proposal_ref` for automated triggers
  assumes a scheduler exists that runs the batch evaluation — that scheduler itself is unbuilt.
- How `promotion_state` is actually surfaced to a skill-selection/retrieval step (ranking logic) is
  explicitly out of scope — only that the field exists and what changes it.
- Whether the `±0.05` / `MIN_SAMPLE=20` / "3 consecutive" numbers should differ per skill category
  (e.g. stricter thresholds for a safety-sensitive skill) is flagged as a reasonable v2 extension
  once real operating data exists to calibrate per-category thresholds against.
- Whether a capability-gap retrieval-miss signal (used conditionally in the 60-day archive-outcome
  check) exists in Hermes at all — if it doesn't, half of that specific outcome check is simply
  skipped, not blocking.

### From `skill-human-review.md`
- Whether the mandatory substantive-review-comment requirement should be enforced by character
  count (gameable), LLM-judged substantiveness (introduces a second judge-calibration problem), or
  left to reviewer discipline with only non-emptiness enforced mechanically.
- Whether some categories (e.g. `protected_constraints` changes) should require more than one
  human reviewer, given their blast radius — this design assumes single-reviewer approval
  throughout, matching the schema's single `reviewed_by` field; two-person review would need a
  schema change (an array instead of a scalar) not made here.
- Whether the human-review queue needs an SLA/escalation path. The rejecting-default behavior
  (current version stays live while something sits in NEEDS REVIEW) could leave a revision
  unreviewed indefinitely with no forcing function — not addressed by any of the five documents.

### Cross-cutting, new at this assembly level
- **No document specifies who is authorized to act as "the human reviewer"** for each category —
  e.g. whether `protected_constraints` changes require a specific role (security/release-eng, per
  the worked `human:release-eng` example in `skill-contract-schema.md`) versus any team member.
  The `approver.identity` field records *who* acted, but not *who is eligible to*.
  authorization/RBAC question is unaddressed across all five documents.
- **No document specifies what happens if a human reviewer never responds** to a mandatory-human
  category (distinct from the SLA question above, which is about *queue* visibility — this is
  about whether there's a hard timeout/escalation after which, e.g., a task blocks indefinitely on
  the kanban board per this project's own unblock-loop-to-triage convention, or something else).
- **No document reconciles what happens if two proposals for the same skill are in flight
  simultaneously** (e.g. a revision pending NEEDS REVIEW while a disuse-based archive candidate
  also fires) — the state machine in `skill-forgetting-and-lineage.md` §4 implicitly assumes
  serialized transitions; concurrent in-flight proposals for the same `skill_id` are unaddressed.

---

## 5. Non-goals of the overall design (carried from the sub-documents)

- No implementation code (telemetry collection, validator, scheduler, CLI, ledger storage) exists
  yet anywhere in this design. All five documents are specifications for future implementation
  work.
- No change to Hermes's existing skill files, skill contracts, or runtime configuration was made
  to produce this design (verified in §6).
- The design does not attempt per-category threshold tuning (e.g. stricter numbers for
  safety-sensitive skills) — one global set of thresholds is specified, with per-category tuning
  explicitly flagged as a future extension (§4 above).

---

## 6. Explicit verification: no live skills or configuration were modified

This task's deliverable is exclusively the five markdown/JSON documents under `docs/design/`
(`skill-contract-schema.md`, `skill-contract.schema.json`, `skill-utility-metrics.md`,
`skill-validation-rules.md`, `skill-forgetting-and-lineage.md`, `skill-human-review.md`, and this
index). Verification performed before opening the PR:

```
$ git merge-base kanban-pipelines main
bf0203a794a07aac3ff764972ab9087baa4e4f4e

$ git diff --stat bf0203a794..<this-PR-head> -- . ':!docs/design'
(empty for the commits this task added — see below)
```

Note: this shared worktree also contains **unrelated, uncommitted** changes from other concurrent
tasks at the time this work was done (`gateway/session_kanban.py`, `hermes_cli/
kanban_db_dispatch.py`, `tests/hermes_cli/test_kanban_review_lifecycle.py`, a new
`skills/devops/definition-of-done/` directory) — none of these were touched, staged, or committed
by this task; they belong to other in-flight work on this branch and are explicitly excluded from
this PR's diff. This task's own commit(s) touch only:
- `docs/design/skill-human-review.md` (new)
- `docs/design/hermes-skill-ops-design.md` (new, this file)

No file under any `skills/` directory, no profile config, no `hermes-config` runtime state, and no
kanban board database was read for writing or modified by this task. All five input documents this
task consumed were themselves prior design-only commits on this same branch.
