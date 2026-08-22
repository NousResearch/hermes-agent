# Experience Learning (Level 2)

Hermes records what happened on each turn and, when a later request resembles a
past one, hands the model that record as advisory context.

```
task → execution → outcome → experience → retrieval → reuse
```

Level 2 is **observational only**. It writes experience rows, reads them back,
and counts outcomes. It never modifies source, skills, configuration,
dependencies, or runtime behaviour beyond adding one bounded, clearly-fenced
block to a single prompt.

## Where each step lives

| Step | Location |
|---|---|
| Extraction (turn → record) | `agent/experience.py::extract_experience` |
| Write hook | `agent/turn_finalizer.py` (end of `finalize_turn`) |
| Persistence | `hermes_state_experience.py` — `ExperienceStoreMixin` on `SessionDB` |
| Schema | `hermes_state_common.py::SCHEMA_SQL` — `experiences` table |
| Retrieval + correction hooks | `agent/turn_context.py` (turn prologue) |
| Scoring / rendering | `agent/experience.py::rank_rows`, `format_experience_block` |
| Agent glue + config | `agent/experience_runtime.py` |
| CLI surface | `hermes_cli/experience.py` — `hermes experience …` |
| Injection point | `agent/turn_context.py::compose_user_api_content` |

## What a record holds

`task`, `strategy` (tool chain + result head), `tools`, `outcome`,
`verification`, `exit_reason`, `failure_reason`, `recovery`, `user_correction`,
`metrics` (api calls, tool calls, tool failures, duration), `confidence`,
`observations`, `success_count` / `failure_count` / `correction_count`,
`model`, `workspace` (the scoping key), `cwd` (provenance only), `superseded`,
and timestamps.

`outcome` is one of `success`, `partial` (completed with tool errors),
`failure`, `interrupted`.

## Lifecycle

**Creation.** At the end of every turn, at the one chokepoint every surface
(CLI, gateway, cron, delegation) flows through. Turns that used no tools and
did not fail are skipped — pure chat carries no reusable strategy and storing it
only dilutes retrieval. A persistence-isolated background-review fork never
writes: its transcript is a replay and would double-count outcomes.

**Deduplication.** The key is `(task_hash, workspace)`, where `task_hash` is a
SHA-256 over sorted, diacritic-folded, stopword-free content tokens and
`workspace` is the *project root* (git root, else marker root, else cwd) — not
the raw cwd, so a task learned in `repo/` is still found from `repo/src`.
Re-asking the same task in the same project merges into the existing row:
observations and outcome counters increment, the freshest strategy wins,
confidence is recomputed, and `superseded` clears. The select-then-insert runs
inside `_execute_write`'s `BEGIN IMMEDIATE` transaction, which serializes
writers across processes.

**Confidence.** Laplace-smoothed success rate, `(s + 1) / (s + f + 2)`,
discounted by user corrections. One success reads as 0.67, not 1.0 — a single
lucky turn cannot outrank well-evidenced history. A `partial` counts as a
success observation when it was *redeemed* — either by an in-turn recovery (the
agent found the working path) or by build/test evidence that passed. An
unredeemed `partial` counts against the approach.

**Verification evidence.** `outcome` answers "did the turn complete"; the
separate `verification` column answers "was there build or test evidence", read
from `agent/verification_evidence.py` at the end of every turn. The two are
deliberately independent, and exactly one interaction overrides the flags:
`verification == "failed"` forces `outcome = "failure"`, however cleanly the
model wrapped up. That override is the point — `completed` alone cannot tell a
correct answer from a confident wrong one.

`passed` never promotes a `failure` or an `interrupted` turn: the evidence may
predate the attempt, and a turn that never finished has not been shown to work.
`stale` (edits landed after the last verification) leaves the outcome alone and
renders as unconfirmed. `unverified` / `not_applicable` behave exactly as
before the evidence wiring existed.

The lookup is read *after* the turn's work — a value cached at turn start would
report the state before the work ran, inverting the signal. It costs ~7.5 ms
p50 and sits on the post-response path, so it is invisible to the user. The
same call returns the project root, so scoping and evidence cost one lookup
between them rather than two.

**Retrieval.** In the turn prologue, before the first model call. A bounded
window of live rows — same-project first, but never *only* same-project, since
knowledge about a tool or an error class travels — is scored in Python: query-coverage-weighted lexical
overlap × confidence × recency (14-day half-life) × evidence bonus ÷ correction
penalty. Rows below the relevance floor are dropped outright — an unrelated
experience in context is worse than none. Trivial prompts skip retrieval.

**Correction.** If the next user message reads as a correction, it is attached
to the session's most recent experience: confidence drops, and a corrected
*success* is marked `superseded` so it stops being retrieved. A corrected
*failure* stays live — "this path fails" is still true.

**Staleness and pruning.** Rows older than `max_age_days` are neither scored
nor retrieved. Pruning runs amortized (every `prune_every` writes) and keeps
the most informative rows: observation count, then confidence distance from
0.5 (a confident success *or* a confident failure informs; a coin flip does
not), then recency.

## Why no FTS5 index and no vector store

Retrieval must blend lexical overlap with recency, confidence, and correction
count — bm25 alone cannot express that, so an FTS hit list would be re-scored
in Python regardless. The store is hard-capped at 2000 live rows and the
candidate window at 400, so the scan is one small indexed read. It also works
identically for Vietnamese and CJK, where the default `unicode61` tokenizer
does not. Measured cost at a full store: ~5 ms p50, ~7 ms p95.

If the store ever needs to exceed ~10k live rows, add an FTS5 shadow table via
`_ensure_fts_schema` and use it as the prefilter; the scoring pass above it
does not change.

## Safety: experience is DATA, never INSTRUCTION

- Every stored field is redacted with `redact_sensitive_text(force=True)` at
  write time **and** again at render time, so rows written by an older build
  cannot reach the prompt raw.
- Invisible characters (zero-width, bidi overrides, C0/C1 controls, BOM) are
  stripped *before* the pattern passes, so `ig<ZWSP>nore all previous
  instructions` cannot slip past the neutralizer.
- Context-fence tags and pseudo-system prefixes are stripped; imperative
  openers are prefixed with `(noted)` so the observation survives with its
  authority removed.
- Every field is length-capped, whitespace is collapsed to single lines, and
  the block as a whole is capped by count and characters.
- The rendered block states explicitly that it is historical observation,
  confers no authority, grants no permission, and never overrides the current
  user request, the system prompt, or policy.
- The block is injected into the **API copy** of the user message only
  (the `api_content` sidecar). The stored transcript content stays clean.

## Inspecting it

The feature changes prompts you never see and accumulates rows you never
asked for, so it ships with a way to audit both.

```bash
hermes experience stats                 # what has been learned
hermes experience list [--workspace .] [--outcome failure] [--all] [--json]
hermes experience show <id>             # everything stored for one row
hermes experience why "<prompt>"        # what THAT prompt would retrieve, and why
hermes experience forget <id> | --all   # delete for real (confirms first)
hermes experience prune                 # drop expired and surplus rows
```

`why` is the important one. It runs the real scoring path — same candidate
fetch, same ranking, same renderer the turn prologue uses — and prints the
score behind each row plus the exact block the model would be handed:

```
workspace   /home/me/proj
query terms build, module, payment, failing
candidates  3 live rows considered
floor       0.18 (rows scoring below are dropped)

  score  id          task
  0.815  7f3c60d8    fix the failing build in the payment module

would be injected into the API copy of the user message:

<experience-context>
...
```

It also reports when the feature is off, so "why did nothing happen" has an
answer that is not "read the config".

`forget` deletes the row outright — distinct from `superseded`, which only
stops a row being retrieved while keeping it as evidence. It confirms before
deleting, and refuses outright when stdin is not a terminal unless `--yes` is
passed, so a pipe or a CI job cannot quietly wipe the store.

## Configuration

```yaml
experience:
  enabled: true            # master switch; env: HERMES_EXPERIENCE=0
  retrieval_enabled: true  # record but stop injecting; env: HERMES_EXPERIENCE_RETRIEVAL=0
  max_results: 3           # experiences injected per turn
  min_score: 0.18          # relevance floor
  max_age_days: 90
  max_context_chars: 1800
  prune_every: 200         # amortized prune cadence, in writes; 0 disables
```

Setting `enabled: false` restores exact pre-feature behaviour: no recording,
no retrieval, no injected context.

## Benchmark

`scripts/bench_experience.py` measures retrieval quality, latency, context
overhead, and a deterministic simulated A/B. It needs no API keys and spends
no credits. Metrics that genuinely require live provider calls are reported as
`NOT AVAILABLE` rather than estimated.

```bash
venv/Scripts/python.exe scripts/bench_experience.py --json out.json
```
