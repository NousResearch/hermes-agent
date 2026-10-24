# Skill Forgetting (Archive/Keep/Promote) and Lineage Tracking

**Status**: design only — no telemetry, scheduler, or skill-store code is written or run by this
document, and no live skill file or config is touched.

**Inputs**:
- `docs/design/hermes-skill-ops-synthesis.md` (the synthesis note) — §d (retirement) and §e
  (lineage) are the direct inputs; see also the Summary Table and Open Items 2/4.
- `docs/design/skill-utility-metrics.md` (the metrics doc) — supplies `U(s)`, `N(s)`,
  `N_scored(s)`, `coverage`, the `provisional` flag, and the `α=5` Beta-Binomial shrinkage this
  document builds its trigger on. This document does not redefine any of those; it only adds the
  decision rule that consumes them.
- `docs/design/skill-contract-schema.md` / `skill-contract.schema.json` — defines `version`
  (`skill_version`, `rubric_version`), `lineage` (`previous_version`, `update_reason`,
  `evidence_case_ids`, `reviewed_by`, `deprecates`), and `retirement_state`
  (`status ∈ {active, candidate_for_retirement, archived}`, `candidate_reason`, `min_sample_met`,
  `archived_at`). This document specifies *when* those fields change value and *what additional
  ledger* backs them.

**Hard constraint carried over from the task**: skills are **archived, never deleted**. Every rule
below treats `archived` as a terminal-but-reversible state (§5), not a deletion.

---

## 1. Why a separate minimum-sample gate is still needed

The metrics doc already does one layer of noise control: Beta-Binomial shrinkage (`α = 5`) pulls
a low-`n` stratum's score toward baseline so a skill with 2 invocations in a stratum cannot swing
its own lift far from zero (metrics doc §4.1). That is necessary but **not sufficient** for a
retirement/promotion *decision*, for two reasons:

1. Shrinkage controls how far a noisy estimate can move, but a noisy estimate near the threshold
   can still cross it by chance even when shrunk — at low `n` the *variance* of `util_k(s)` is
   still large even though its *mean* is pulled toward baseline.
2. A single evaluation window is one draw from a noisy process. Even a well-powered single window
   has a non-trivial false-positive rate at realistic effect sizes (quantified in §3 below). A
   decision with consequences (removing a skill from the active pool, or elevating it to
   preferred) should not fire on one unlucky — or lucky — window.

So this document adds two things on top of the metrics doc's shrinkage: (a) an explicit minimum
scored-sample size before a window's `U(s)` is eligible to count toward a decision at all, and
(b) a **sustained-trigger** requirement (multiple independent windows agreeing) before the
decision actually fires. Both numbers are derived below, not asserted.

---

## 2. Minimum sample size: `MIN_SAMPLE = 20` scored invocations per evaluation window

**Rule**: no archive, keep-confirmation, or promote decision may be made from a window with
`N_scored(s) < 20` in that window. A skill/version below this is **provisional** — it stays in its
current `retirement_state.status` (default `active`) with no verdict rendered. This directly
reuses the `provisional` flag already defined in the metrics doc (§4.4), so there is exactly one
place that number is defined and this document just adopts it rather than inventing a second
noise cutoff. If both documents land in Hermes, implementers should treat `N_scored(s) ≥ 20` as a
single named constant (e.g. `MIN_SAMPLE_FOR_DECISION`), consumed by both the reporting layer
(metrics doc) and the decision layer (this document).

**Justification (why 20, not 10 or 50)** — a variance argument, not a round number:

The shrunk per-stratum estimator (metrics doc §4.1) is a Beta-Binomial posterior mean with
pseudo-count `α = 5`:

```
util_k(s) = (net_success + α·b_k) / (n + α)
```

Its variance, worst-cased at `b_k = 0.5` (maximum-variance baseline), scales as:

```
Var(util_k) ≈ (n / (n+α))² · b_k(1−b_k) / n
```

At `n = 20`, `α = 5`: shrinkage factor `(20/25)² = 0.64`; raw-rate SE `= sqrt(0.25/20) ≈ 0.1118`;
shrunk SE `≈ 0.64 × 0.1118 ≈ 0.072`. That SE is used directly in the trigger-threshold
justification in §3 — `n = 20` is the smallest sample size at which the shrunk SE drops under
`0.08`, i.e. under the size of the archive/promote effect this design cares about detecting
(`0.05`, see §3). Below `n = 20` the SE exceeds the effect size itself, so a single window
literally cannot distinguish "skill is 0.05 below baseline" from "skill is at baseline" — any
decision there would be reading noise, which is exactly what the task asks this document to
prevent. `n = 20` is therefore the smallest sample size at which a single window's precision is
even in the right ballpark for the threshold chosen; §3's sustained-trigger requirement supplies
the rest of the confidence.

(`n = 50` would halve the SE again but at real Hermes usage volumes — most skills are invoked at
most a few times per day — would push the time-to-first-decision for a moderately-used skill out
by months with limited extra protection once the sustained-trigger rule in §3 is applied; `n = 10`
would leave the single-window SE at `~0.10`, larger than the 0.05 effect size, making even a
3-window sustained trigger's false-positive math in §3 too loose. 20 is the point where the
single-window math first becomes self-consistent with the threshold chosen.)

---

## 3. The archive / promote trigger: exact condition and numbers

### 3.1 The three bands

```
U(s) ≤ −0.05   →  below-baseline band   (feeds the ARCHIVE trigger)
−0.05 < U(s) < +0.05  →  keep band      (no retirement-state change)
U(s) ≥ +0.05   →  above-baseline band   (feeds the PROMOTE trigger)
```

`±0.05` is a lift of 5 percentage points relative to the stratum baseline — chosen as the
threshold because (worked in §2) it is the smallest effect size for which `n = 20`'s shrunk SE
(`≈ 0.072`) still gives a usable single-window signal-to-noise ratio, while remaining a
practically meaningful effect (a skill performing 5 points worse than "what anything would
achieve on tasks like this" is a real, actionable gap, not a rounding artifact).

### 3.2 Sustained trigger (hysteresis): 3 consecutive independent windows

A single window crossing `±0.05` at `n = 20` is **not** sufficient to change `retirement_state`.
The trigger requires the **same-direction** band to be hit in **3 consecutive, non-overlapping**
evaluation batches, each independently satisfying `MIN_SAMPLE = 20`.

**"Non-overlapping" is load-bearing**: the metrics doc's reporting window is a *trailing* window
(90 days or 200 invocations, re-evaluated continuously, metrics doc §4.5) — consecutive trailing
windows share most of their underlying invocations and are therefore statistically correlated,
not independent draws. For the purpose of *this* decision rule only, evaluation uses **disjoint
batches of invocations**: batch `k+1` begins at the first scored invocation after batch `k`'s
last one. This is a distinct accounting from the continuously-displayed trailing `U(s)` score —
the trailing score is what a human or dashboard sees at any moment; the disjoint-batch sequence is
what the archive/promote decision consumes. (Implementation note, not a live change: this means
the decision engine keeps its own cursor per `(skill_id, skill_version)` — "last invocation_id
included in the last closed batch" — separate from the trailing-window computation.)

**False-positive math**: under the null hypothesis that a skill genuinely performs at baseline
(`true lift = 0`), a single batch's chance of crossing `±0.05` by chance alone, using the shrunk-SE
normal approximation from §2 (`SE ≈ 0.072`):

```
P(single batch false-trigger) ≈ Φ(−0.05 / 0.072) ≈ Φ(−0.70) ≈ 0.242
```

That is too high to act on alone (~24% chance of a false archive/promote signal from one batch) —
which is exactly why §2 said `n=20` alone is not sufficient. Requiring 3 consecutive disjoint
batches to agree (treated as independent under the null, since they share no invocations):

```
P(3 consecutive false-triggers) ≈ 0.242³ ≈ 0.014  (~1.4%)
```

A ~1.4% false-archive / false-promote rate per candidate-skill-version is the number this design
accepts as "sustained enough." (Raising the requirement to 4 consecutive batches would drop this
to ~0.3% but roughly doubles the minimum time/volume to reach any decision for a moderately-used
skill; 3 is the point where the false-positive rate is already below the attrition/noise floor of
most comparable "flaky CI retry" conventions, so going further mainly trades decision speed for a
diminishing returns on confidence. If real operating data later shows 3-batch sustained triggers
still firing too often — i.e. the independence assumption behind `0.242³` breaks down, e.g.
because consecutive batches for a low-traffic skill span long enough wall-clock time that the
*underlying skill itself* drifted between batches — that is grounds to revisit this number, not a
flaw in the reasoning here.)

**A mixed streak resets the counter.** If batch `k` is below `−0.05` and batch `k+1` lands inside
the keep band or crosses to the opposite side, the consecutive counter resets to 0 — the trigger
requires 3 *consecutive* same-direction crossings, not 3-out-of-last-5 or any other leniency, to
keep the false-positive math in this section exact.

### 3.3 Disuse trigger (independent of the utility trigger — no sample-size gate needed)

A skill/version with **zero invocations for a trailing 180 days** is flagged
`candidate_for_retirement` with `candidate_reason = "disuse"` immediately, with no minimum-sample
requirement, because this trigger makes no claim about a noisy rate — "nobody has used this in six
months" is a fact, not an estimate, so there is nothing for a sample-size gate to protect against.
This exists because the utility-threshold trigger (§3.1–3.2) structurally **cannot** fire for a
genuinely dormant skill — if `N(s) = 0`, no batch ever closes, so a low-traffic-into-zero skill
would otherwise sit in `active`/`provisional` forever. 180 days is chosen to exceed the metrics
doc's 90-day default trailing window by 2x, so a skill does not get disuse-flagged merely for
being quiet for one reporting cycle during, e.g., a seasonal lull.

### 3.4 Minimum-sample gate applies to the utility trigger only

To be explicit about scope: `MIN_SAMPLE = 20` (§2) gates the *utility-band* trigger (§3.1–3.2)
only. The disuse trigger (§3.3) is sample-size-free by construction (§3.3's own reasoning). A
`retirement_state.candidate_reason` must therefore always say which trigger fired
(`"utility_sustained_low"` / `"utility_sustained_high"` for promote / `"disuse"`), since the two
have different evidentiary weight and different human-review expectations (§6).

---

## 4. State machine

```
                    ┌─────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
  ┌────────┐  3x U(s)≥+0.05  ┌─────────────────────┐       │ reverted within
  │ active │ ───────────────▶│ candidate_for_       │       │ cool-off window
  │        │                 │ promotion            │───────┘ (no change applied,
  │        │                 │ (new, see §4.2)      │         counter resets)
  │        │                 └──────────┬───────────┘
  │        │                            │ approved (human or
  │        │                            │ automated per companion
  │        │                            │ human-review design)
  │        │                            ▼
  │        │                 ┌──────────────────────┐
  │        │                 │ promoted (tier=       │
  │        │                 │ preferred, §4.2)      │
  │        │◀────────────────┤                       │
  │        │   demoted (3x   └──────────────────────┘
  │        │   U(s)≤-0.05
  │        │   while promoted)
  │        │
  │        │  3x U(s)≤-0.05  ┌──────────────────────┐   approved   ┌──────────┐
  │        │ OR 180d disuse  │ candidate_for_        │─────────────▶│ archived │
  │        │ ───────────────▶│ retirement            │               │          │
  │        │                 │ (schema status)       │               │          │
  │        │◀────────────────┤                       │◀──────────────┤          │
  └────────┘  rejected/      └──────────────────────┘  unarchived    └──────────┘
              reverted within                           (human only,
              cool-off window                            see §5)
```

Notes on transitions:

- `active → candidate_for_retirement` and `active → candidate_for_promotion` are **fully
  automated** — they are just the trigger conditions in §3 firing. Flagging a candidate costs
  nothing and is reversible, so it does not need a human in the loop to *propose* it.
- `candidate_for_retirement → archived` and `candidate_for_promotion → promoted` require
  **approval** (see §6 — this document defines the trigger and the lineage record; which specific
  categories of approval must be human vs. may be automated is the companion human-review design's
  job, `docs/design/hermes-skill-ops-design.md` / the human-review sub-document it assembles).
  This document's recommendation, stated here so the companion design has a starting position: an
  `archived` transition should default to **requiring human approval**, because an archived skill
  stops being offered to any task even though its record persists — that is a live behavior change
  with no automatic self-correction if wrong, unlike a promotion (§6 justifies this further). A
  `promoted` transition may default to **automated approval**, because promotion only affects
  *ranking/preference among already-active alternatives* — it is lower-stakes and self-correcting
  (§4.2's demotion rule below undoes a bad promotion automatically).
- **Cool-off / rejection**: a `candidate_for_*` state that is rejected (human says no) or whose
  underlying signal reverses before approval (e.g. a candidate-for-retirement skill's next
  disjoint batch comes back above `−0.05`) returns to `active` and the consecutive-batch counter
  resets to 0 — a fresh 3-in-a-row streak is required to re-trigger, so a skill cannot be flapped
  into archival by one favorable batch sandwiched between two unfavorable ones under a laxer
  "3-of-5" rule (§3.2 already rules this out, repeated here for the state machine's sake).

### 4.1 `candidate_for_retirement` → `archived`: no version bump required

Archiving is a **status** change, not a contract revision — the skill's `skill_version` and
`rubric_version` (procedure and rubric) do not change when a skill is archived; only
`retirement_state.status` changes. The lineage record for an archive event (§7) therefore carries
`to_version == from_version` (unchanged) and a separate `retirement_transition` field, distinct
from a `revised` lineage event where `to_version` strictly differs from `from_version`.

### 4.2 Promotion: a new, additive field (not yet in `skill-contract.schema.json`)

The existing schema's `retirement_state` only expresses the archive axis
(`active | candidate_for_retirement | archived`). Promotion is an **orthogonal** axis — a skill
can be simultaneously `active` and `preferred`. This document therefore proposes (as a forward
recommendation for whoever next revises `skill-contract.schema.json`, not as a change made by this
task) an additive, non-breaking sibling object:

```jsonc
"promotion_state": {
  "tier": "experimental" | "standard" | "preferred",   // default "standard"
  "candidate_reason": "utility_sustained_high | ...",
  "min_sample_met": true,
  "decided_at": "2026-10-23T00:00:00Z"
}
```

`tier = "preferred"` is a ranking/retrieval signal only (e.g. surfaced ahead of `red`/`alt`
equivalents in a skill-selection step) — it has **no effect on `preconditions`,
`outcome_criteria`, or whether the skill can still fail its own gates. A promoted skill that later
regresses (3 consecutive batches `≤ −0.05`, §3.2) is demoted back to `tier = "standard"`
automatically (no human approval required for demotion, matching the asymmetry in §4's approval
note: elevating requires approval, stepping back down from an elevation does not, since the
self-correction is itself the safety mechanism for having allowed automated promotion in the
first place).

---

## 5. Archive is reversible; `archived` is not `deleted`

An `archived` skill's full contract record (all prior versions, all lineage events) remains
queryable forever — nothing is removed from storage, matching the task's hard constraint. An
archived skill is simply excluded from the active invocation pool (preconditions/applicability
matching skips `archived` skills by construction — not a separate rule, just a consequence of
"not offered").

**Unarchiving** (`archived → active`) is **always a human-approved action**, regardless of how
the original archive was approved (human or automated) — this is an explicit asymmetry from
promotion/demotion's self-correcting design (§4.2): there is no automated "un-regression" signal
analogous to the demotion trigger, because an archived skill accrues zero new invocations (it is
not offered), so there is no fresh evidence that could automatically justify bringing it back. A
human must supply a reason (new context, a human manually wants to re-test it, a dependency
reappeared) and that reason is captured in the lineage record (§7) as a new `unarchived` event
with its own `update_reason` and `approver`.

---

## 6. Interaction with human review (scope boundary)

This document defines **what** triggers a candidate decision and **what evidence** justifies it.
It deliberately does **not** define the full policy of which categories of change require human
vs. automated approval in general — that is `t_2981cd73`'s scope (the companion human-review
design feeding the same top-level `docs/design/hermes-skill-ops-design.md`). What this document
*does* commit to, as inputs that design should treat as fixed rather than re-derive:

1. `active → candidate_for_*` is always automated (§4) — there is nothing to review yet, it is
   just a trigger firing; reviewing every trigger-fire would defeat the point of having an
   automated trigger at all.
2. `candidate_for_retirement → archived` should default to requiring human approval (recommended,
   not mandated by this document) because it is the one transition in this state machine with no
   automatic self-correction if the decision was wrong (§4's note) — archiving stops offering a
   skill; nothing downstream will notice and flag "actually this skill was fine," only a human
   audit or a complaint would.
3. `candidate_for_promotion → promoted` may default to automated approval because demotion is
   self-correcting (§4.2).
4. Whatever the human-review design settles on, the lineage record (§7) must capture
   `approver.approver_type ∈ {human, automated}` and `approver.identity` regardless of which path
   was used — the lineage schema does not presuppose the answer to "which transitions need a
   human," it just always records who (or what) actually approved.

---

## 7. Lineage record: exact fields

### 7.1 Two layers, not one

The skill contract (`skill-contract.schema.json`) already carries a `lineage` object
(`previous_version`, `update_reason`, `evidence_case_ids`, `reviewed_by`, `deprecates`) — but that
object is **denormalized**: it holds only the most recent change, embedded in the current contract
document, for convenience when a human or router reads a single skill file and wants to know "why
does this version exist." It is not, and should not be made into, a full history — a contract
document should not grow an ever-longer embedded array as a skill accumulates revisions over
years.

This document adds the second layer: an **append-only `skill_lineage_event` ledger**, one row per
skill change of any kind (`created`, `revised`, `archived`, `unarchived`, `promoted`, `demoted`),
keyed independently of any single contract document. The contract's embedded `lineage` object is
always generated **from** the most recent ledger row for that `skill_id` at commit time — it is a
read-optimized cache of the ledger's head, never edited independently of it. This is the same
relationship a database table has to its latest-row view: one source of truth (the ledger), one
convenience projection (the embedded field).

### 7.2 `skill_lineage_event` fields

| field | type | required | meaning |
|---|---|---|---|
| `lineage_event_id` | uuid | yes | primary key of the ledger row |
| `skill_id` | string | yes | matches `skill_contract.skill_id` |
| `event_type` | enum | yes | `created` \| `revised` \| `archived` \| `unarchived` \| `promoted` \| `demoted` |
| `from_version` | `{skill_version, rubric_version}` or `null` | yes | the contract's `version` object immediately before this event; `null` only for the first `created` event |
| `to_version` | `{skill_version, rubric_version}` | yes | the contract's `version` object immediately after this event. For `archived`/`unarchived`/`promoted`/`demoted` events, `to_version == from_version` (status-only change, §4.1) |
| `retirement_transition` | `{from, to}` over `active\|candidate_for_retirement\|archived` or `null` | required for `archived`/`unarchived` events, else `null` | mirrors the exact `retirement_state.status` change this event caused |
| `promotion_transition` | `{from, to}` over `experimental\|standard\|preferred` or `null` | required for `promoted`/`demoted` events, else `null` | mirrors the `promotion_state.tier` change (§4.2) |
| `proposal_ref` | string | yes | pointer to the **originating proposal** — a kanban task id, a PR URL, or (for automated triggers) the evaluator job-run id that produced the candidate flag. This is the field the task explicitly asked for: "link to its originating proposal." |
| `proposal_kind` | enum | yes | `human_authored` \| `llm_updater_proposed` \| `automated_evaluator_triggered` — distinguishes "a person wrote this skill/revision" from "an LLM proposed this revision from failure analysis" (Skill-V's updater pattern, synthesis §e) from "the archive/promote trigger in §3 fired automatically" |
| `evidence_refs` | array of strings | yes (may be empty only for the very first `created` event, matching the contract schema's own rule for `evidence_case_ids`) | identifiers of the specific evidence that justified this event — trajectory/evidence-case IDs for a revision, or the three (or more) `lineage_batch_id`s whose `U(s)` crossed the threshold for an archive/promote event |
| `evidence_snapshot` | object | yes for `revised`/`archived`/`promoted`/`demoted`; omitted for `created` | a **frozen copy** (not a live pointer) of the numbers that justified the decision at the time it was made: `{U_s, N_s, N_scored_s, coverage, per_stratum_lift: [...], batches: [{batch_id, n, U_batch}, ...], decision_rule_version}`. Frozen because later changes to telemetry, re-aggregation, or a bugfix in the utility computation must never retroactively alter the historical justification of a past decision — the ledger records what was believed and acted on *at the time*, auditable as such even if later shown to be wrong (that is what §7.4's `outcome` field is for) |
| `decision_rule_version` | string | yes for any event produced by an automated trigger | identifies which version of *this document*'s thresholds (`§3.1`'s `±0.05`, `§2`'s `MIN_SAMPLE=20`, `§3.2`'s "3 consecutive") was in effect — e.g. `"skill-forgetting-and-lineage.md@v1"`. Required so that if these numbers are later tuned, every historical decision remains attributable to the rule version that actually produced it, not silently reinterpreted under the current rule |
| `approver` | `{approver_type: human\|automated, identity: string, approved_at: timestamp}` | yes | who/what approved the event. `identity` for a human is e.g. `"human:release-eng"` (matching the format already used in `skill-contract-schema.md`'s worked example 2); for automated, `"automated-gate-only"` or the specific evaluator-job identifier |
| `rationale` | string | yes | free-text justification; this is the value that gets copied verbatim into the contract's embedded `lineage.update_reason` at commit time (§7.1) |
| `outcome_status` | enum | yes, defaults `pending` | `pending` \| `confirmed_beneficial` \| `confirmed_harmful` \| `reverted` \| `superseded` — see §7.4 |
| `outcome_evaluated_at` | timestamp or `null` | yes | when `outcome_status` was last set to a non-`pending` value |
| `outcome_evidence_ref` | string or `null` | yes | pointer to whatever evidence closed the loop (§7.4) |
| `created_at` | timestamp | yes | when this ledger row was written (immutable once written, except for the `outcome_*` fields which are the one sanctioned later update — see §7.3) |

### 7.3 Append-only, with one sanctioned mutation

Every field above except `outcome_status` / `outcome_evaluated_at` / `outcome_evidence_ref` is
immutable once written — this is what "append-only ledger" means and is what makes
`evidence_snapshot` trustworthy as a historical record (§7.2). The three `outcome_*` fields are
the **one** sanctioned later update, because the task explicitly requires linking each change to
"the eventual outcome of that change," and an outcome is by definition not known at the moment the
change is made — see §7.4 for exactly when and how that update happens. Implementations should
make this an explicit single-purpose `UPDATE ... SET outcome_status=..., outcome_evaluated_at=...,
outcome_evidence_ref=...` with the three original-moment fields never touched by that statement,
not an in-place rewrite of the whole row.

### 7.4 Closing the loop: how `outcome_status` gets set

A scheduled outcome-check runs **60 days** after any ledger row's `created_at` (chosen to exceed
one evaluation batch's typical accrual time at moderate skill-usage volume, so there's a real
chance of new evidence existing, without waiting so long the review becomes useless):

- **`created` / `revised` events**: outcome is `confirmed_beneficial` if the new version's `U(s)`
  (first non-provisional disjoint batch after the change) is `≥ 0`, i.e. the revision did not make
  things worse relative to baseline; `confirmed_harmful` if that first batch comes back `< −0.05`
  (the archive band itself) within the 60 days, which strongly suggests the revision regressed;
  otherwise `pending` is left in place and the check retries at the next 60-day mark (a skill with
  low traffic may simply not have accrued a fresh batch yet — that is not a verdict, it's absence
  of data, and should not be forced into a `confirmed_*` state).
- **`archived` events**: outcome is `confirmed_beneficial` if the skill was not `unarchived` within
  the 60 days and no task during that window recorded an explicit "no applicable skill found" gap
  matching this skill's former `applicability.scope.task_types` (a capability-gap signal, if
  Hermes's retrieval layer logs misses — if it doesn't, this half of the check is skipped and the
  verdict rests on the unarchive signal alone); `reverted` if a human unarchived it within the
  window (the clearest possible signal the archive was premature, and the reason this document
  requires human approval for unarchiving — §5 — so that signal is always attributable to a
  specific accountable decision, not an automated flip-flop).
- **`promoted` / `demoted` events**: outcome is `confirmed_beneficial` if `tier` is unchanged 60
  days later; `reverted` if a `demoted` event followed within the window (§4.2's automatic
  demotion firing is itself the outcome signal for a bad promotion).
- **`superseded`**: not produced by the scheduled check — set immediately, at the moment a *later*
  lineage event for the same `skill_id` lands, on whichever prior event's `outcome_status` was
  still `pending` and whose `to_version` is superseded by the new event's `from_version`. This
  keeps the ledger from accumulating indefinitely-`pending` rows for versions nobody cares about
  anymore once a newer version has already been judged.

---

## 8. Linking the lineage record to the skill contract's `version` field

The join is exact and bidirectional:

- `skill_lineage_event.to_version` (both `skill_version` and `rubric_version`) **must equal**
  the `version` object of the contract document that this event produced. For a `created` or
  `revised` event that is a brand-new contract document; for `archived`/`unarchived`/`promoted`/
  `demoted` events it is the contract document's version **unchanged** (§4.1).
- The contract's own embedded `lineage.previous_version` **must equal**
  `skill_lineage_event.from_version.skill_version` for the ledger row whose `to_version` matches
  the contract's current `version.skill_version` — i.e. walking from a contract document to its
  producing ledger row and reading that row's `from_version` must reproduce exactly what the
  contract's own embedded field already claims. This is a consistency invariant a validator can
  check mechanically: **for every contract document, there must exist exactly one ledger row with
  matching `to_version`, and that row's `from_version.skill_version` must equal the contract's
  `lineage.previous_version`.** A mismatch means either the ledger or the contract was hand-edited
  out of band, which should never happen (§6/§7.1 — contracts are generated from ledger events,
  never edited independently).
- `skill_lineage_event.evidence_refs` and `.rationale` are the values the contract's embedded
  `lineage.evidence_case_ids` and `lineage.update_reason` are copied from at commit time (§7.1) —
  one source, two readers (the full ledger, and the convenience embed), never two independent
  authors of the same fact.
- `skill_lineage_event.approver.identity` is the value the contract's embedded `lineage.reviewed_by`
  is copied from — e.g. a ledger approver of `{approver_type: "human", identity: "human:release-eng"}`
  produces a contract `lineage.reviewed_by = "human:release-eng"`, matching the exact string format
  already used in `skill-contract-schema.md`'s worked Example 2.
- `skill_lineage_event.retirement_transition.to` is the value written into the contract's
  `retirement_state.status`; `min_sample_met` on the contract is set to `true` only when the
  triggering ledger row's `evidence_snapshot.N_scored_s ≥ 20` (the utility trigger, §2) — for a
  pure disuse trigger (§3.3), `min_sample_met` is `false` (there was no sample at all, by
  definition of disuse), which is consistent with the schema's own description of that field
  ("whether `evidence_scope.min_sample_per_bucket` was satisfied before this status was set") —
  disuse-triggered archival is explicitly flagged as not sample-justified, because it isn't making
  a claim about a rate at all (§3.3).

---

## 9. Worked example, using the metrics doc's own numbers

The metrics doc's worked example (its §5) gives `U(A) ≈ −0.049`, `U(B) ≈ +0.080`, each from a
single 50-invocation snapshot. Applying this document's rules to that same data:

- **Skill A**, `U(A) = −0.049`: `N_scored = 50 ≥ 20` ✓ (not provisional), but `−0.049` is *inside*
  the keep band (`> −0.05`, just barely — §3.1) despite looking bad relative to its own naive raw
  rate. **No candidate flag fires.** This is a feature, not an edge-case gap: A is slightly
  underperforming its stratified baseline, but not by enough to clear the noise-adjusted
  threshold from a single batch, and three independent batches all landing at exactly this
  boundary would be a coincidence worth a human look rather than an automatic archive — the
  sustained-trigger rule (§3.2) is exactly what prevents a borderline single number like this from
  being over-interpreted.
- **Skill B**, `U(B) = +0.080`: `N_scored = 50 ≥ 20` ✓, and `0.080 ≥ +0.05` — this **single batch**
  crosses into the promote band. It still does **not** promote B by itself (§3.2 requires 3
  consecutive disjoint batches at `+0.05` or above); it sets/continues B's promotion-candidate
  streak counter to 1 of 3. If B's next two disjoint 20+-sample batches also land `≥ +0.05`, B
  transitions to `candidate_for_promotion` and a `skill_lineage_event` row is written with
  `event_type = "promoted"` (pending the approval path from §6), `proposal_kind =
  "automated_evaluator_triggered"`, `evidence_refs` pointing at the three batch ids, and
  `evidence_snapshot` embedding each batch's `U_batch` and the aggregate — reproducing the exact
  math a human reviewer could re-derive from `docs/design/skill-utility-metrics.md` §4 without
  having to trust the ledger blindly.

---

## 10. Explicit non-goals / open items

This document deliberately does **not** specify:

- The storage technology for the `skill_lineage_event` ledger (SQL table, event log, etc.) — a
  schema-level spec, not an implementation choice.
- The exact format/wire schema of `evaluator job-run id` referenced in `proposal_ref` for
  automated triggers — assumed to exist as part of whatever scheduler runs the §3 batch
  evaluation, itself a future implementation task.
- How `promotion_state` (§4.2) is actually surfaced to a skill-selection/retrieval step — only
  that the field exists and what changes it; consuming it to re-rank candidates is a retrieval
  design question, out of scope here.
- Whether the `±0.05` / `MIN_SAMPLE=20` / "3 consecutive" numbers should differ per skill category
  (e.g. a safety-sensitive skill might warrant a stricter, more sample-hungry threshold before
  archiving). This document picks one global default, justified in §2–3, and flags per-category
  tuning as a reasonable v2 extension once real data exists to calibrate against — consistent with
  the metrics doc's own §7 treatment of its open items.
- Whether a capability-gap retrieval-miss signal (referenced conditionally in §7.4's archive
  outcome check) exists in Hermes today — if it doesn't, that half of the outcome check is simply
  skipped, as noted inline, not blocking on it.
- The final allocation of which transitions require human vs. automated approval — §6 states this
  document's inputs/recommendations to that decision but defers the full policy and the
  review-presentation design to the companion human-review design document.
