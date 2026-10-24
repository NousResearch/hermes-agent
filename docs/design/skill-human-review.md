# Human Review Process for Skill Changes

**Status**: design only — no live skill, config, CLI, or UI code is touched by this document.

**Inputs** (this document adds nothing to their numbers; it only decides who approves what and how
the decision is presented):
- `docs/design/skill-contract-schema.md` / `skill-contract.schema.json` — contract shape,
  `lineage.reviewed_by`, `retirement_state`, MAJOR/MINOR/PATCH bump policy.
- `docs/design/skill-utility-metrics.md` — `U(s)`, `N_scored(s)`, `provisional` flag,
  stratification by `(task_category, difficulty_bucket)`.
- `docs/design/skill-validation-rules.md` — the PASS / FAIL / NEEDS HUMAN REVIEW decision rule
  (§3), the six conditions that route to a human (§3.3), and the automatic-rollback rule (§4).
- `docs/design/skill-forgetting-and-lineage.md` — the archive/promote state machine (§4), its
  recommendation that archiving default to human approval and promotion default to automated
  approval (§6), and the `skill_lineage_event` ledger fields (§7) a review decision must populate.

This document closes the one gap all four of those explicitly deferred: **the final policy of
which change categories require a human, and what a human reviewer must be shown** so the review
is a real check, not a rubber stamp.

---

## 1. Design principle

A rubber-stamp review happens when the reviewer sees a conclusion ("approve this revision?") but
not the evidence that produced it, so the only rational move is to trust the system that already
decided. To avoid that, this design's rule is: **a human is only inserted into the loop at points
where the automated gates in the other four documents *cannot* render a verdict on their own
evidence** — never as a redundant rubber-stamp over a verdict the gate already reached with full
confidence, and never skipped at a point where the gate explicitly said it couldn't decide. Every
category below cites the specific place in a prior document that already established *why* that
category needs a human (so this document is reconciling existing commitments, not inventing new
ones), except where a genuinely new category is introduced (marked explicitly).

A second principle, specific to avoiding rubber-stamping: **the review prompt must contain
everything needed to independently re-derive the verdict**, not a trust-me summary. §3's
checklist is the review surface where that is enforced.

---

## 2. Category table: human-required vs. auto-apply

| # | Change category | Human required? | Why (source) |
|---|---|---|---|
| 1 | **New skill creation** (first `created` lineage event) | **Human** | Not addressed by name in any prior doc, but follows directly from their shared logic: a brand-new contract has no replay evidence at all (`N_scored = 0`), so the validation-rules gate (`skill-validation-rules.md` §3.3(3): `N_scored < 20` → NEEDS REVIEW) cannot even be evaluated — a new skill is permanently below that floor until someone approves its first live exposure. |
| 2 | **Revision scored PASS by the automated gate** (MINOR/PATCH bump, no hard-constraint violation, no regressing bucket, `N_scored ≥ 20`) | **Auto-apply** | `skill-validation-rules.md` §3.1 — this is precisely the case the gate was built to resolve unassisted; the evidence (replay set, regression test, p-values) is fully mechanical and reproducible. Routing it to a human anyway is the rubber-stamp failure mode this design exists to avoid: a reviewer with no reason to doubt a mechanical pass will approve it unread. |
| 3 | **Revision scored FAIL** | **Auto-reject, human notified (not blocked)** | `skill-validation-rules.md` §3.2 — rejection leaves the current version live; nothing changed that requires a human's permission. The author/proposer is notified of the failure with the same evidence a human reviewer would get (§3 below), so they can fix and resubmit, but the task is not held open waiting on anyone. |
| 4 | **Revision scored NEEDS HUMAN REVIEW** (any of the six triggers in `skill-validation-rules.md` §3.3: protected_constraints changed, a hard_constraint's evidence_matcher changed, `N_scored < 20`, unevidenced new scope, borderline p-value in (0.05, 0.20), first-time `llm_judge_rubric` criterion) | **Human** | By definition — `skill-validation-rules.md` §3.3 lists these as exactly the cases the automated gate cannot resolve on its own evidence. |
| 5 | **`active → candidate_for_retirement` / `candidate_for_promotion` (trigger firing)** | **Auto-apply** (flagging only, not the terminal transition) | `skill-forgetting-and-lineage.md` §4/§6(1) — flagging costs nothing and is reversible; requiring a human to approve a candidate *flag* (as opposed to the resulting archive/promotion) would make the trigger pointless, since the trigger's entire job is to surface candidates for review in the first place. |
| 6 | **`candidate_for_retirement → archived`** | **Human** (default; this document adopts the forgetting-doc's recommendation as the final policy) | `skill-forgetting-and-lineage.md` §4/§6(2): no automatic self-correction if archiving was wrong — an archived skill simply stops being offered, and nothing downstream flags "actually this was fine." |
| 7 | **`candidate_for_promotion → promoted`** | **Auto-apply** (final policy, adopting the forgetting-doc's recommendation) | `skill-forgetting-and-lineage.md` §4/§6(3): a bad promotion is self-correcting via the automatic demotion rule (§4.2 of that doc) once 3 consecutive batches drop back below `-0.05`. |
| 8 | **`archived → unarchived`** | **Human, always** (no automated path exists) | `skill-forgetting-and-lineage.md` §5 — there is no automated "un-regression" signal for a skill accruing zero invocations while archived; this is unconditional regardless of how the original archive was decided. |
| 9 | **Automatic live rollback** (`skill-validation-rules.md` §4.2) | **Auto-apply, human notified (not blocked)** | The rollback doc is explicit that this is intentionally *not* human-gated — by the time it fires, the revision already passed the pre-promotion gate once on real evidence, so a live trigger is high-confidence, and delaying it to wait on a human lets confirmed-live regressions keep accumulating real task failures. |
| 10 | **Any `protected_constraints` change, regardless of pass/fail status elsewhere** | **Human, always** | `skill-validation-rules.md` §3.3(1) and `skill-contract-schema.md`'s bump table (always MAJOR) — a changed safety invariant is, by construction, not something past replay evidence can validate, since the invariant itself is what changed. |
| 11 | **Repeated-rollback escalation** (≥2 rollbacks in 30 days) | **Human, mandatory for every subsequent candidate until cleared** | `skill-validation-rules.md` §4.4 — explicit escalation rule; this document adds no new numbers, only confirms the escalation routes through the same review surface (§3) as any other human-required category. |
| 12 | **Decision-rule or threshold changes** (editing any numeric constant in the consolidated table of the top-level design, e.g. `MIN_SAMPLE`, the `±0.05` band, the 0.05/0.20 p-value band) | **Human, always** (new category, not covered by the four input docs since they describe skill-level changes, not meta-level rule changes) | These numbers govern every other row in this table; an automated process changing its own gating thresholds with no human in the loop is a standing risk the other four documents never had to consider because they treat the thresholds as fixed. Any change to `decision_rule_version` (the version string `skill-forgetting-and-lineage.md` §7.2 requires on every automated ledger event) requires a human-approved lineage event of its own, logged the same way as a skill-level change. |

**Summary of the auto-apply set**: category 2 (clean PASS), category 5 (candidate flagging),
category 7 (promotion), category 9 (live rollback). Everything else requires a human. This is
deliberately a short auto-apply list — it is exactly the set of transitions each source document
already proved has either full mechanical evidence (2), no consequence of its own (5), or a
self-correcting downstream safety net (7, 9). Every other transition either has no automated
verdict to rubber-stamp (1, 4, 6, 8) or is a safety-relevant change the automated gates are
explicitly defined to never certify alone (10, 11, 12).

---

## 3. What must be presented to the human reviewer

A review prompt is **rejected as incomplete** (the proposal cannot be submitted for review) if any
required field below is missing — this is enforced the same way a schema validation failure is
enforced elsewhere in this design: structurally, not by reviewer discipline. The required-fields
checklist is organized by what question it lets the reviewer answer without re-deriving anything
by hand.

### 3.1 Required fields checklist

| # | Field | Answers the question | Populated from |
|---|---|---|---|
| 1 | `proposal_kind` + one-line description of the change category (which of §2's 12 rows this is) | "What kind of decision am I being asked to make?" | Routing logic; this is category 1–12, never free text |
| 2 | Full contract diff: **old version full text ↔ new version full text**, unified diff format, not just a changed-field summary | "What literally changed, structurally?" | `skill-contract.schema.json` documents (old vs. candidate) |
| 3 | Version bump classification (MAJOR/MINOR/PATCH) with the specific triggering rule cited from `skill-contract-schema.md`'s bump table | "Is the bump size consistent with the diff, or is this a MINOR label hiding a MAJOR-shaped change?" | Validation pipeline, cross-checked against field #2 |
| 4 | `protected_constraints`: explicit **unchanged / changed** flag, and if changed, the full before/after text of every constraint that differs | "Did a safety invariant move, and to what?" | `applicability.protected_constraints` diff |
| 5 | Replay/regression results table: one row per `(task_category, difficulty_bucket)` bucket touched, columns = `n_k`, net-success delta, rework-rate delta, hard-constraint violations (count), p-value, significance verdict | "Does the evidence actually support a pass, independent of the gate's own conclusion?" | `skill-validation-rules.md` §1–2 replay output, verbatim, not summarized |
| 6 | `N_scored` total and `provisional` flag | "Is this decision based on enough evidence to mean anything?" | `skill-utility-metrics.md` §4.4 |
| 7 | `replay_set_id` and the literal list of sampled `invocation_id`s (or a link to the stored list if long) | "Can I audit exactly what was tested, not just trust a summary count?" | `skill-validation-rules.md` §1.3 reproducibility requirement |
| 8 | Which specific NEEDS-REVIEW trigger(s) fired (if category 4), cited by the exact clause from `skill-validation-rules.md` §3.3(1)–(6) | "Why is a human being asked at all, specifically?" | Validation pipeline routing reason |
| 9 | For `llm_judge_rubric` criteria (new or changed): the full pinned `model` / `prompt_ref` / `threshold`, plus 3–5 example transcripts (trajectory input → rubric verdict) spanning both a clear-pass and a clear-fail case | "Is the automated judge itself trustworthy, since this is the one non-mechanical check kind?" | `skill-contract-schema.md` `evidence_matcher.kind = llm_judge_rubric` params + sampled transcripts from the replay set |
| 10 | `lineage.update_reason` (candidate's proposed rationale) and `proposal_ref` (originating task/PR/evaluator-job id) | "Who or what is asking for this, and what do they say the reason is?" | `skill-forgetting-and-lineage.md` §7.2 |
| 11 | For archive/promote decisions: the full 3-batch sequence (`batch_id`, `n`, `U_batch` for each of the 3 consecutive disjoint batches), not just the final aggregate | "Did this actually sustain across 3 independent windows, or is this one lucky/unlucky batch being over-interpreted?" | `skill-forgetting-and-lineage.md` §3.2, §7.2 `evidence_snapshot.batches` |
| 12 | For unarchive requests: the human-supplied reason text (mandatory, non-empty — there is no automated justification possible per `skill-forgetting-and-lineage.md` §5) | "What changed in the world that justifies bringing this back?" | Reviewer's own input, captured into the ledger |
| 13 | Prior lineage history for this `skill_id`: last 5 `skill_lineage_event` rows (type, outcome_status, approver) | "Does this skill have a track record of churn, repeated rollbacks, or contested decisions?" | `skill_lineage_event` ledger query |
| 14 | `decision_rule_version` that produced this routing/verdict | "Which version of the rules made this recommendation, in case the rules themselves are later found to be wrong?" | `skill-forgetting-and-lineage.md` §7.2 |
| 15 | Explicit reviewer action set: **Approve / Reject / Request changes (with required comment)** — no silent default, no pre-checked "approve" | "Am I being nudged toward a specific answer?" | UI/CLI contract, §4 below |

Fields 2, 5, 7, and 11 are the ones that most directly prevent rubber-stamping: they give the
reviewer the raw diff and raw evidence table, not a pre-computed verdict label alone. A reviewer
who only sees "PASS, approve?" has nothing to disagree with; a reviewer who sees the per-bucket
regression table can independently notice, e.g., that a borderline bucket was excluded from the
average for looking fine elsewhere.

### 3.2 What is deliberately *not* required

The checklist does not require the reviewer to re-run replay or regenerate the diff themselves —
all of §3.1's fields are pre-computed and shown, because the goal is a reviewer who can
**verify** a specific, inspectable claim quickly, not one forced to redo the pipeline's work by
hand (that would make human review a bottleneck disproportionate to its value, pushing toward
rubber-stamping via simple fatigue). The anti-rubber-stamp guarantee comes from showing raw
evidence, not from making the reviewer's job artificially harder.

---

## 4. Concrete CLI mockup

Hermes already exposes a CLI surface for kanban review (`kanban_request_review` /
`kanban_request_changes` in the kernel's own tool set) with a comparable intent: don't let a task
complete without a human-legible summary and explicit accept/reject signal. The mockup below
follows that same shape for skill-review, as a new `hermes skill review` subcommand family (naming
only — not a commitment to this exact CLI framework, this is deliberately a plain top-level
command set so the mockup is self-contained without assuming Hermes's existing CLI plumbing).

```
$ hermes skill review list
┌─────────────┬────────────────────────┬──────────────────────┬────────────┬──────────┐
│ REVIEW_ID   │ SKILL_ID               │ CATEGORY              │ ROUTED BY  │ AGE      │
├─────────────┼────────────────────────┼──────────────────────┼────────────┼──────────┤
│ rv-00418    │ pdf                    │ needs_human_review    │ gate:v1    │ 2h       │
│ rv-00417    │ github                 │ archive_candidate     │ gate:v1    │ 1d       │
│ rv-00415    │ new-skill/slack-notify │ new_skill             │ human:dm   │ 3d       │
└─────────────┴────────────────────────┴──────────────────────┴────────────┴──────────┘

$ hermes skill review show rv-00418
╔══════════════════════════════════════════════════════════════════════════════╗
║ Review rv-00418 — skill "pdf"  v1.2.0 → v1.3.0 (candidate)                    ║
║ Category: needs_human_review   Routed by: skill-validation-rules.md@v1        ║
║ Routing reason: borderline p-value 0.11 in bucket (pdf.fill-form, standard)   ║
╚══════════════════════════════════════════════════════════════════════════════╝

── Proposal ─────────────────────────────────────────────────────────────────
  proposal_ref     task:t_55a1  (kanban)
  proposal_kind    llm_updater_proposed
  update_reason    "Switch fields-filled-correctly to allow ±1 trailing space
                     after user reports of false fails on form exports from
                     LibreOffice (evidence case ec-pdf-0142)."
  decision_rule_version   skill-validation-rules.md@v1

── Version bump ─────────────────────────────────────────────────────────────
  rubric_version: 1.1.0 -> 1.2.0   classified MINOR
  trigger: weighted_criteria change, passed non_regression except as noted below
  protected_constraints: UNCHANGED (diff below confirms byte-identical)

── Contract diff ────────────────────────────────────────────────────────────
  --- v1.1.0 outcome_criteria.weighted_criteria[0]
  +++ v1.2.0 (candidate) outcome_criteria.weighted_criteria[0]
  @@ fields-filled-correctly @@
  -  "evidence_matcher": { "kind": "exact_match", ... }
  +  "evidence_matcher": { "kind": "regex_match", "pattern": "^{value}\\s{0,1}$" }
  [full contract diff: 1 hunk, view with --full-diff]

── Replay / regression evidence ─────────────────────────────────────────────
  replay_set_id: rs-9931   (47 trajectories, view ids with --show-replay-set)
  ┌──────────────────────────┬────┬────────────┬────────────┬──────┬─────────┬──────────┐
  │ bucket                   │ n  │ Δnet_succ  │ Δrework    │ HC   │ p-value │ verdict  │
  ├──────────────────────────┼────┼────────────┼────────────┼──────┼─────────┼──────────┤
  │ pdf.fill-form / standard │ 22 │ -4.1pp     │ +0.5pp     │ 0    │ 0.11    │ BORDERLINE│
  │ pdf.fill-form / complex  │ 15 │ +1.8pp     │ -0.2pp     │ 0    │ 0.41    │ pass     │
  │ pdf.extract-fields / std │ 10 │ -0.5pp     │ 0.0pp      │ 0    │ 0.88    │ pass     │
  └──────────────────────────┴────┴────────────┴────────────┴──────┴─────────┴──────────┘
  N_scored total: 47  (provisional: no, >= 20)

── Prior lineage (last 5 events for "pdf") ──────────────────────────────────
  2026-09-02  revised    v1.0.0->v1.1.0  approver=automated-gate-only  outcome=confirmed_beneficial
  2026-07-14  created     -> v1.0.0      approver=human:terra          outcome=confirmed_beneficial

── llm_judge_rubric criteria touched ────────────────────────────────────────
  (none in this candidate)

── Decision ──────────────────────────────────────────────────────────────────
  [a]pprove    [r]eject    [c]omment / request changes    [d]iff detail    [q]uit without deciding
  > _
```

```
$ hermes skill review decide rv-00418 --approve \
    --comment "Regex widening is reasonable for the reported LibreOffice case;
               the borderline bucket is low-traffic (n=22) and the effect is
               small (4.1pp) — accepting the gate's own borderline framing
               rather than overriding it either direction."
✓ rv-00418 approved by human:terra
✓ lineage_event le-10234 written: event_type=revised, approver={human, human:terra}
✓ contract "pdf" promoted to v1.3.0 (rubric_version), replay buffer reset per bump policy: no
  (MINOR bump keeps buffer)
```

Rejection and request-changes follow the same transcript shape, writing `outcome_status` and a
`rationale` into the ledger exactly as `--approve` does, matching `kanban_request_changes`'s
existing "closes the review run, requires concrete reason" behavior in this project's own kanban
tooling — the skill-review CLI deliberately mirrors that convention rather than inventing a
different one.

### 4.1 Mandatory fields enforced by the CLI, not just documented

`hermes skill review decide` refuses to accept `--approve` or `--reject` without a `--comment`
whose length exceeds a configured minimum (non-empty, substantive) — this is the CLI-level
enforcement of the "no rubber stamp" principle: a reviewer cannot click through to "approve" with
zero recorded rationale. The comment is stored verbatim as the ledger's `rationale` field
(`skill-forgetting-and-lineage.md` §7.2), which is itself copied into the contract's embedded
`lineage.update_reason` at commit time (§7.1 of that document) — so a lazy one-line comment is
visible forever in the committed contract, which is itself a soft incentive against writing one.

---

## 5. Open questions this document does not resolve

- Whether a minimum substantive-comment length (§4.1) should be enforced by character count,
  LLM-judged "is this a real rationale," or left to reviewer discipline with only non-empty
  enforced mechanically — character-count is gameable, LLM-judging a rationale adds another
  judge-calibration problem this design doesn't want to open.
- Whether multiple categories can require *more than one* human reviewer (e.g. `protected_
  constraints` changes might warrant two-person review given their blast radius) — this design
  assumes single-reviewer approval throughout, matching the schema's single `reviewed_by` field.
- Whether the review queue itself needs an SLA/escalation path (a NEEDS-REVIEW item sitting
  unreviewed for weeks leaves the rejecting-default of "current version stays live" indefinitely)
  — not addressed here; see also the top-level design's open-questions list.
