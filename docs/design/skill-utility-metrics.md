# Skill Utility Metrics and Difficulty-Confounding Control

**Status**: design only — no telemetry collection, logging code, or config changes are made by
this document. It specifies what to measure, how to log it, and how to compute a fair utility
score once that logging exists.

**Input**: `docs/design/hermes-skill-ops-synthesis.md` ("the synthesis note"). That note's §(b)
and Open Item 1 establish that **neither SkillOps nor Skill-V normalizes outcome metrics for task
difficulty** — SkillOps pools raw success/failure across all calls regardless of difficulty;
Skill-V scopes evidence to applicable tasks but still pools difficulty levels within that scope.
This is flagged there as novel work for Hermes. This document is that work.

---

## 1. What counts as "utility": the three raw counts

A skill's utility is built from three logged counts, each defined precisely enough to compute
without ambiguity. All three are counted **per invocation**, not per task — if a skill is invoked
twice in one task episode, that is two invocation records.

### 1.1 Invocation count — `N(s)`

**Definition**: the number of times skill `s` was selected and executed by an agent, regardless
of outcome. An invocation is logged at the moment the skill's procedure begins executing (not at
retrieval/consideration time — a skill that is surfaced by search but never actually run does not
count).

**Logged fields per invocation** (`skill_invocation` event):

| field | type | meaning |
|---|---|---|
| `invocation_id` | uuid | primary key |
| `skill_id` | string | which skill, including skill version (see §4, lineage note) |
| `skill_version` | string | the `ν`-style version tag active at invocation time |
| `task_id` | string | the kanban task this invocation happened under |
| `episode_id` | string | worker run / session id, for multi-invocation tasks |
| `task_category` | string | declared task category (see §3.1) |
| `difficulty_bucket` | enum | `trivial` \| `standard` \| `complex` \| `research` (see §3.2) |
| `invoked_at` | timestamp | UTC, when execution began |
| `invoking_profile` | string | which assignee/profile invoked it |

`N(s) = count(skill_invocation WHERE skill_id = s)` over the measurement window (see §5.1 for
window definition).

### 1.2 Outcome success/failure count — `S(s)`, `Fl(s)`

**Definition**: each invocation resolves, after the fact, to exactly one of four outcome states.
Resolution happens when the *consuming* step (the task's completion, the next pipeline stage, or
a reviewer) judges the artifact the skill produced:

- **`success`**: the artifact/action the skill produced was accepted as-is by the consuming step
  (task completed, review approved, next stage proceeded) with no correction needed.
- **`failure`**: the artifact/action was rejected outright, discarded, or the task/step failed
  because of it (e.g. `kanban_request_changes` citing this skill's output as the reason, or the
  task later got blocked/escalated attributing the skill's output as the cause).
- **`reworked`**: the artifact was initially accepted but had to be corrected afterward (see §1.3
  — this is its own state, not folded silently into success or failure at logging time).
- **`unknown`**: no resolution signal arrived within the resolution window (e.g. the task never
  reached a reviewable state, or outcome attribution is ambiguous — multiple skills touched the
  same artifact and blame can't be isolated). `unknown` invocations are **counted in `N(s)` but
  excluded from `S(s)`/`Fl(s)`/utility computation** — they are not silently treated as success.

**Logged fields** (`skill_outcome` event, one per invocation once resolved):

| field | type | meaning |
|---|---|---|
| `invocation_id` | uuid | foreign key to the invocation |
| `outcome_status` | enum | `success` \| `failure` \| `reworked` \| `unknown` |
| `resolved_at` | timestamp | when the outcome was determined |
| `resolver` | string | `completion_contract` \| `review_verdict` \| `human` \| `timeout` |
| `evidence_ref` | string | pointer to the review comment / commit / task event that justified it |

`S(s) = count(outcome_status = success)`, `Fl(s) = count(outcome_status = failure)`, both scoped
to skill `s` within the window.

**Resolution window**: an invocation that stays `unknown` more than 14 days (configurable) past
`invoked_at` is finalized as `unknown` and excluded going forward rather than left open
indefinitely — this bounds how long a skill's score can be "pending."

### 1.3 Rework-after-use count — `Rw(s)`

**Definition**: a `reworked` outcome is a *specific, narrower* claim than failure: the artifact
was accepted at the time (task/review proceeded) but a human or a later pipeline step had to
redo or fix what the skill produced, discovered after acceptance. This matters because it is
exactly the case raw accept/reject counts miss — a skill that looks "successful" on first pass but
silently generates work downstream is not actually providing value, and SkillOps-style raw
utility (synthesis §b) would never see this.

**Detection triggers** (any one is sufficient to log a `reworked` outcome against the original
invocation, retroactively, via `invocation_id` lookup by artifact fingerprint):

1. **Explicit reviewer rework**: a `kanban_request_changes` call on a task/artifact where the
   changed file(s) or deliverable match what this invocation produced, and the review reason text
   references the skill's output.
2. **Re-touch by a later worker**: a subsequent task (same artifact, same file paths) is created
   or dispatched whose body indicates it is fixing/redoing the prior skill's output (e.g. a
   follow-up task created with `parents=[this_task]` and a title/body matching a fix pattern, or a
   git commit on the same file within the rework-detection window with a message matching
   `fix|revert|redo|correct` against a commit that touches lines the skill's invocation produced).
3. **Human correction comment**: a human comment on the task or a downstream task explicitly
   stating the skill's output needed correction.

**Rework-detection window**: 30 days (configurable) after `resolved_at = success`. A fix landing
outside that window is not attributed to the skill (too much time/context has passed for
confident attribution) — this bounds false attribution of stale skills to recent fixes.

**Logged fields** (`skill_rework` event):

| field | type | meaning |
|---|---|---|
| `invocation_id` | uuid | which original invocation is being reworked |
| `detected_at` | timestamp | when the rework was logged |
| `trigger` | enum | `review_request_changes` \| `retouch_task` \| `human_comment` |
| `rework_ref` | string | pointer to the triggering task/commit/comment |
| `severity` | enum | `minor` (small fix) \| `major` (substantial redo) — informational, not scored differently in v1 (see §7 for future extension) |

When a `skill_rework` event lands against an invocation whose `outcome_status` was `success`, that
invocation's `outcome_status` is updated to `reworked` (one UPDATE, not a duplicate record — an
invocation has exactly one current outcome state at any time, with the transition logged in an
append-only history table for auditability).

### 1.4 Net success/failure for scoring

For the utility score (§4), `reworked` is treated as a failure, because the point of utility
measurement is "did this skill's output stand without further work" — not "did it look fine for a
moment":

```
net_success(s, stratum)  = count(outcome_status = success)              [in that stratum]
net_failure(s, stratum)  = count(outcome_status = failure)
                          + count(outcome_status = reworked)            [in that stratum]
```

`unknown` is excluded from both. This is the only place rework enters the formula; §1.2's four-way
state split is preserved in storage so `reworked` can always be distinguished from `failure` for
diagnostics (e.g. "this skill fails outright" vs. "this skill needs a human to clean up after it"
are different problems even though both count against utility).

---

## 2. Why raw rates are not enough: the confounding problem

If utility were just `net_success(s) / (net_success(s) + net_failure(s))` pooled across all of
`s`'s invocations, a skill that's disproportionately invoked on hard tasks looks worse than one
invoked on easy tasks, independent of actual skill quality — exactly the gap the synthesis note
flags in both source papers (§b: SkillOps pools indiscriminately; Skill-V scopes to applicable
tasks but still pools across difficulty within that scope). A retirement or ranking policy built
on the raw rate would punish skills for being used where they're needed most (e.g. a debugging
skill invoked only on complex failures should not be judged against a skill invoked only on
trivial boilerplate).

The fix has two parts: (1) stratify by task category/difficulty so each invocation is compared
only against similar invocations, and (2) normalize each stratum's rate against a baseline so that
being-good-at-hard-tasks and being-good-at-easy-tasks are measured on the same scale.

---

## 3. Stratification

### 3.1 Task category

`task_category` is a free-text-but-controlled-vocabulary field already present on kanban tasks
(or inferable from the assignee profile, e.g. `code-review`, `research`, `design`, `ops`). No new
field is required at the task level — this design reuses whatever categorical label Hermes already
has for task type. If a task has no category, it falls into category `uncategorized`, which is its
own stratum (not merged into others) so it doesn't contaminate categorized baselines.

### 3.2 Difficulty bucket

Four buckets, assigned at task-creation time by whichever signal is available, in priority order:

1. **Explicit tag** (preferred): the task creator/orchestrator sets `difficulty_bucket` directly
   in the task body/metadata (`trivial` / `standard` / `complex` / `research`). Orchestrators
   already make this judgment implicitly when writing task specs; this just asks them to record
   it as a field.
2. **Computed proxy** (fallback, when no explicit tag exists): a difficulty proxy score from
   cheaply available signals —
   `proxy = w1*parent_count + w2*estimated_turns + w3*file_count + w4*prior_attempt_count`
   (weights `w1..w4` are a config detail out of scope for this design — the point is the proxy
   exists and is deterministic, not that its exact weighting is fixed here) — then bucketed into
   terciles computed over a trailing window of tasks in the same category (so "complex" means
   complex *relative to other tasks of that category*, not an absolute global scale, which avoids
   e.g. all `research` tasks being globally "complex" by category alone).
3. **Unknown**: if neither signal is available, bucket is `unknown` — its own stratum, excluded
   from baseline computation for other buckets (see §3.3) so a handful of unlabeled tasks don't
   dilute real baselines.

A **stratum** is the pair `(task_category, difficulty_bucket)`. All utility computation below
happens per-stratum first, then aggregated.

### 3.3 Baseline success rate per stratum

For stratum `k = (category, bucket)`, the baseline is the **leave-one-out global net success
rate** for that stratum — i.e. the net success rate of *all other* invocations (any skill, or no
skill at all, if Hermes logs skill-less task outcomes in the same event stream) in stratum `k`,
excluding the skill currently being scored:

```
b_k(s) = [ Σ_{s' ≠ s} net_success(s', k) ] / [ Σ_{s' ≠ s} (net_success(s', k) + net_failure(s', k)) ]
```

Leave-one-out matters specifically when one skill dominates usage in a stratum (e.g. only one
skill is ever invoked on `research`/`complex` tasks) — without excluding `s` from its own
baseline, `b_k(s)` would just converge to `s`'s own rate and the lift (§4) would always be ~0,
making the correction vacuous exactly where it's most needed. If excluding `s` leaves fewer than
`min_baseline_n` (default 10) invocations in the stratum, `b_k` is instead taken from the next
broader scope available: same category across all buckets, then global average, in that order —
logged as `baseline_scope = stratum | category | global` so it's always clear which scope was
used.

---

## 4. The utility score

### 4.1 Shrinkage within a stratum

Raw per-stratum rates are unstable at low `n` (a skill invoked twice in a stratum, 1 success 1
failure, is not reliably "50%"). Each stratum rate is shrunk toward that stratum's baseline using
a Beta-Binomial posterior mean with pseudo-count `α` (default `α = 5`):

```
util_k(s) = ( net_success(s,k) + α · b_k(s) ) / ( net_success(s,k) + net_failure(s,k) + α )
```

As `n_k(s) = net_success(s,k) + net_failure(s,k) → 0`, `util_k(s) → b_k(s)` (no evidence ⇒ assume
average). As `n_k(s)` grows, `util_k(s)` converges to the raw rate. This single shrinkage step
does the job a separate "minimum sample size" gate would otherwise need — a skill with 2
invocations in a stratum simply can't move its score far from baseline, rather than needing a
hard cutoff rule.

### 4.2 Per-stratum lift

```
lift_k(s) = util_k(s) − b_k(s)
```

Positive lift = skill outperforms the stratum baseline (good, even if the raw rate looks low
because the stratum is hard). Negative lift = skill underperforms its own stratum's baseline (bad,
even if the raw rate looks high because the stratum is easy). This is the core de-confounding
step: it is no longer "success rate" but "success rate *relative to what any skill achieves on
tasks this hard*."

### 4.3 Aggregate utility score

Strata are combined via invocation-weighted mean of lift (weighting by `n_k(s)`, the skill's own
invocation count in that stratum, so strata the skill is actually used in dominate its score —
not strata it barely touches):

```
U(s) = [ Σ_k n_k(s) · lift_k(s) ] / [ Σ_k n_k(s) ]
```

`U(s)` is on the same scale as a probability difference, roughly in `[-1, 1]` (in practice much
narrower — single-digit-to-low-double-digit percentage points for most real skills). `U(s) > 0`
means the skill is a net positive relative to what's achievable on the tasks it's actually used
for; `U(s) < 0` the opposite; `U(s) ≈ 0` means it performs about as well as anything else would on
those tasks (not necessarily "neutral" in an absolute sense — a skill at `U ≈ 0` on uniformly
`research`/`complex` tasks is still doing real work, just not more than baseline).

### 4.4 Confidence / reporting

Report `U(s)` alongside:
- `N(s)` total invocations (the raw count from §1.1), and
- `N_scored(s) = Σ_k n_k(s)` (invocations actually contributing to `U(s)`, i.e. excluding
  `unknown` outcomes), and
- `coverage = N_scored(s) / N(s)`.

If `N_scored(s) < 20` (configurable), the score is labeled **provisional** — shown, not hidden,
but flagged so a retirement/ranking decision doesn't fire on noise. This directly answers the gap
the synthesis note flags in SkillOps (§d: no minimum-sample guard on `θ_u`-based retirement) by
making low-sample status an explicit, visible flag on the utility score itself rather than a
silent threshold buried in a retirement rule.

### 4.5 Measurement window

All counts are computed over a trailing window (default: last 90 days or last 200 invocations,
whichever is larger, per skill) rather than all-time, so a skill's score reflects its current
version's behavior and recent task mix, not its entire history. Window choice is a config detail;
what matters for this design is that it is a *trailing* window, recomputed periodically, not a
cumulative all-time counter (an all-time counter would make an old skill's past failures
permanently drag down a since-improved version — see the lineage note in §6).

---

## 5. Worked numeric example

Two skills, deliberately chosen with **opposite difficulty mixes**, to show the correction doing
real work — a reader should be able to recompute every number below from the formulas in §4.

**Baselines** (computed from the rest of the population per §3.3, assumed already resolved):

| stratum | baseline `b_k` |
|---|---|
| `(code, easy)` | 0.90 |
| `(code, medium)` | 0.70 |
| `(code, hard)` | 0.50 |

Shrinkage `α = 5`.

**Skill A** — mostly used on easy tasks:

| stratum | `net_success` | `net_failure` | `n_k(A)` | raw rate |
|---|---|---|---|---|
| easy | 34 | 6 | 40 | 0.850 |
| medium | 6 | 4 | 10 | 0.600 |
| hard | 0 | 0 | 0 | — |

**Skill B** — mostly used on hard tasks:

| stratum | `net_success` | `net_failure` | `n_k(B)` | raw rate |
|---|---|---|---|---|
| easy | 0 | 0 | 0 | — |
| medium | 5 | 5 | 10 | 0.500 |
| hard | 26 | 14 | 40 | 0.650 |

**Naive (unstratified) comparison**, i.e. what you'd get *without* this design:

```
raw(A) = (34+6) / (34+6+6+4) = 40/50 = 0.800
raw(B) = (5+26) / (5+26+5+14) = 31/50 = 0.620
```

Naive reading: **A looks clearly better than B** (80% vs 62%). This is the exact failure mode the
synthesis note warns about — A's advantage is just that it's never used on hard tasks.

**Stratified, shrunk, baseline-corrected computation**:

Skill A:
```
util_easy(A)   = (34 + 5·0.90) / (40 + 5) = 38.5/45 = 0.8556
lift_easy(A)   = 0.8556 − 0.90 = −0.0444

util_medium(A) = (6 + 5·0.70) / (10 + 5) = 9.5/15 = 0.6333
lift_medium(A) = 0.6333 − 0.70 = −0.0667

U(A) = (40·(−0.0444) + 10·(−0.0667)) / (40+10)
     = (−1.778 + −0.667) / 50
     = −2.444 / 50 = −0.0489
```

Skill B:
```
util_medium(B) = (5 + 5·0.70) / (10 + 5) = 8.5/15 = 0.5667
lift_medium(B) = 0.5667 − 0.70 = −0.1333

util_hard(B)   = (26 + 5·0.50) / (40 + 5) = 28.5/45 = 0.6333
lift_hard(B)   = 0.6333 − 0.50 = +0.1333

U(B) = (10·(−0.1333) + 40·(0.1333)) / (10+40)
     = (−1.333 + 5.333) / 50
     = 4.0 / 50 = +0.0800
```

**Result**: `U(A) ≈ −0.049`, `U(B) ≈ +0.080`. **The ranking flips**: once corrected for the fact
that A only ever faces easy/medium tasks and still underperforms the baseline on both, while B
spends most of its time on hard tasks and *beats* the hard-task baseline substantially, B is the
better-performing skill — the opposite of what the naive 80%-vs-62% comparison suggested.

This is the worked proof that stratified, baseline-normalized lift (not raw success rate) is the
right quantity to rank, retire, or surface skills by.

---

## 6. Interaction with lineage/versioning (forward reference)

The synthesis note (§e) recommends porting Skill-V's `ν_i` shape (skill version + evidence-case
IDs) for lineage, which is separate design work. The one coupling point this design assumes:
**utility is computed per `(skill_id, skill_version)` pair, not per `skill_id` alone** (see
`skill_version` field in §1.1). When a skill is revised, its utility counter effectively resets —
old-version invocations stay attributed to the old version for history, but a new version starts
with no evidence and its score is `provisional`/baseline-anchored (§4.1, `n=0 → util=b_k`) until
it accumulates its own invocations. This prevents a genuinely improved revision from inheriting a
prior version's bad score, and prevents a regressed revision from coasting on a prior version's
good score — each version is judged only on its own evidence.

---

## 7. Explicit non-goals / open items

This design deliberately does **not** specify:
- The exact proxy-difficulty weights (`w1..w4` in §3.2) — a config/tuning detail, not a metric
  definition question.
- Whether/how `severity` (`minor`/`major`, §1.3) should weight the rework penalty differently —
  v1 treats all rework as equally failing; weighting by severity is a reasonable v2 extension but
  changes the formula in §1.4 and should be its own decision once real rework data exists to
  calibrate against.
- How `U(s)` feeds a retirement decision (threshold, hysteresis, archive-vs-delete) — that is the
  subject of the separate retirement-policy design item flagged in the synthesis note (§d, Open
  Item 2); this document only produces the *score* that policy would consume, including the
  `provisional` flag (§4.4) that a retirement policy must not ignore.
- Live telemetry collection, storage schema migrations, or the actual event-logging code — all of
  §1's "logged fields" tables are a specification for a future implementation task, not a claim
  that this logging exists yet.
