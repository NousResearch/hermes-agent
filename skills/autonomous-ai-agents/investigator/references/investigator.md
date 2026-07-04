# Investigator — design notes

## Layering (primitive vs orchestrator)

- **`next-best-questions` (the ranker)** is a pure, **report-only** primitive: given `(prompt + evidence)`
  it returns the next-best questions ranked by value of information. It never acts.
- **`investigator` (this skill)** is the **orchestrator**: it calls the ranker, answers the top
  questions with a full Hermes agent, folds facts back into one growing context, re-ranks, and
  responds. The split keeps the safe primitive callable anywhere and isolates the privileged loop.

Cross-skill call is in-process: the investigator resolves the ranker's `scripts/` via `HERMES_HOME`
(or `INFOGAIN_SCRIPTS_DIR`) and `import infogain`, then calls `infogain.run(problem, cfg, evidence=)`
and reads `result["all_scored"]`. The grounded answerer/responder delegate to the `ask` skill via
`model_utils.dispatch_single` (a full agent, isolated context).

## Capability ladder

Default is **full agency** (`act`). The ladder only **down-scopes**, and it maps onto the answerer's
toolsets + a prompt directive — it does **not** reinvent a permission system.

| level | toolsets | directive | reversibility |
|---|---|---|---|
| **act** (default) | file, web, terminal | none | unattended — may take real actions |
| **experiment** | file, web, terminal | "reversible experiments only (scratch/worktree)" | reversible by intent |
| **read** | file, web | "read-only; action-needing → NOT_FOUND" | no side effects |

v1 caveat: down-scoping below `act` is **instruction-level** (the directive), since Hermes toolset
read-only granularity is not yet verified. `act` (the default) is unaffected. A future hard sandbox for
`experiment` would pin `answer_cwd` to an auto-created git worktree.

The ladder also gates **answer-artifact capture** (`artifact_write` in `CAPABILITIES`): under
`read`, the answerer prompt omits the "write your answer to <run_dir>/answer-<fp>.json"
instruction — the read directive says "do NOT modify files", and a self-contradictory prompt is
worse than losing per-answer durability. The tombstone journal is written by the loop process
itself and applies at every capability level.

## Routing (v1.2 — triage, judgment, derived)

Prior to 1.2.0 every above-floor question went down one path: research (`grounded_answer`). 1.2.0
adds two more and a batch classifier that chooses between them, all gated behind `cfg["triage"]`
(default off — programmatic callers see byte-identical behavior until they opt in):

- **derived** — the ranker's own derive-or-ask step (`infogain.py`, `auto_derive`) sometimes
  answers a question from the accumulated evidence without any agent call at all
  (`recommendation == "DERIVED"` plus a `derived_answer` string). The loop consumes these
  directly, before the floor/top-K filter, tombstoning them `via="derived"`. `auto_derive` is
  only turned on in `_rank_cfg` when triage is on — triage off means the rank_cfg passed to
  `infogain.run` is byte-identical to 1.1.0.
- **findable vs judgment** — one batch `triage_batch` call classifies the round's top-K
  questions: FINDABLE ("an observable fact a tool-using agent could discover") stays on the
  research path; JUDGMENT ("a preference/decision with no discoverable ground truth") goes to
  `judgment_call`, which makes one conservative, autonomous decision (prefer
  reversible/standard/least-surprising) and tombstones it `via="assumed"` with a `rationale`.
- **fail-open by construction** — an unrouted question (triage off, triage call failed, or no
  valid in-range index) defaults to FINDABLE; duplicate entries after the first valid route are
  ignored. A JUDGMENT question whose judge call fails (dispatch error, bad JSON, `CANNOT_DECIDE`,
  or a hedge matching `_JUDGE_HEDGE_RE` — copied from `pipeline._HEDGE_RE`, not imported, same
  precedent as `fp()`) falls back to research once. Nothing is ever silently skipped.
- **`max_assumes`** bounds how many JUDGMENT routes may be *accepted* per run (counting
  tombstones resumed from a prior journal) — past the cap, JUDGMENT questions go to research
  instead of the judge.

## Evidence-phrasing contract

Every tombstone's `evidence` string follows one of four fixed shapes, so downstream readers (the
responder, `refine_prompt`, a human reading `tombstones.jsonl`) can tell provenance apart by
string alone without needing the `via` field:

| Shape | `via` | Meaning |
|---|---|---|
| `q -> a` | `research` | a fact the grounded answerer discovered |
| `q -> a (derived during analysis)` | `derived` | the ranker itself answered it from existing evidence, no agent call |
| `q -> decision (assumed: rationale)` | `assumed` | an autonomous judgment call — a decision, not a discovered fact |
| `q -> (known gap: ...)` | whichever route produced the NOT_FOUND | unresolved; carried forward as a known gap |

Every new tombstone also carries the ranked question's optional, nullable `value`, maximum answer-
branch `stakes`, and `recommendation`, attached centrally by `_tombstone`. Old journal tombstones
without these fields remain valid, and readers must use optional access and treat them as `None`.
The `iterate()` result's additive `unresolved_key_questions` list surfaces NOT_FOUND gaps whose
value reaches `key_gap_threshold` (default `0.40`) and always includes the single highest-value gap;
entries are deduplicated by question and value-sorted. Older result dicts without this key remain
valid as well.

The stakes-aware final responder is gated by `cfg["stakes_aware_respond"]` and is OFF by default
pending an A/B evaluation before any default change. It can be enabled with config key
`investigator.stakes_aware_respond`, env var `INVESTIGATOR_STAKES_AWARE_RESPOND=on`, or CLI flag
`--stakes-aware-respond on`. When enabled, `iterate()` passes the tombstones and
`unresolved_key_questions` through the existing three-argument responder config. `respond()` buckets
them into **Established facts**, **Minor open gaps**, and (only when nonempty) **⚠️ Unresolved key
questions**. It proceeds non-blocking, states the assumption used for each key gap, and collects those
assumptions in a closing **Material risks — assumptions to confirm** section. With no key gaps, it
omits both the key-gap bucket and material-risk instruction.

`respond()` is content-gated on the inferred-evidence markers: any evidence string containing
`"(assumed:"` or `"(derived"` triggers the "treat as inferred, not observed" framing and its
assumptions/known-gaps ledger; absent those markers its prompt is byte-identical to pre-1.2.0 (a
pure-research run produces the same responder prompt it always did). `refine_prompt()` always asks
for an `## Assumptions` section, and adds the `## Open questions` instruction only when an evidence
string contains `"(known gap:"`.

## Durability (v1.2)

An investigation is resumable via `cfg["run_dir"]` / `--run-dir` — artifact-based (the
resilient-planner drive.py pattern), deliberately NOT a resumable-script engine flow: the journal
is the domain artifact itself (tombstones), so replay+memoize machinery would be a second journal
with nothing extra to say. Mechanics: `tombstones.jsonl` (header record carries `problem_fp`;
stale-problem journals rotate to `.stale`), fp-normalized answered-set, per-question
`answer-<fp>.json` artifacts read before stdout (see `scripts/answerer.py`). Callers that loop
(relentless-solve) pass a per-cycle run dir; their own step memoization sits ABOVE this journal
and never re-enters a completed clarify phase.

## Vantage handling (the cross-cutting thread with the ranker's vantage family)

The ranker's **vantage** family (when enabled) emits questions whose answer is *access-relative* —
`a(question, vantage)` differs by environment / server / identity / credential / token. For such a
question the Investigator should:

1. **Identify the vantage axis** the question names (which env/credential/POV would change the answer).
2. **Investigate from the relevant vantage(s)** it has access to (a vantage = an access/capability
   requirement — being on server B, using token T).
3. If **multiple vantages are reachable**, investigate from each and **report the diff** — the variation
   across vantages is itself the finding (e.g. "prod and staging configs diverge").
4. If **only one is reachable**, flag the answer as **vantage-conditional** rather than absolute.

v1 records the vantage axis and investigates from the current vantage; multi-vantage comparison is a
fast-follow (needs the Investigator to acquire/switch credentials, which is an `act`-level capability).
