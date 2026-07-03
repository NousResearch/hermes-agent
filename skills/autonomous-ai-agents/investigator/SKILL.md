---
name: investigator
description: >
  Use when a task is underspecified and you want an agent to autonomously RESOLVE the unknowns before
  answering — not just list them. Calls the next-best-questions ranker for the next-best questions, then
  researches the top ones with a full Hermes agent (full agency by default — all tools), folds each
  distilled fact into one continuously-growing context, re-ranks, and repeats until it converges, then
  produces the final response. Records answered facts and known gaps as tombstones. Capability is full
  (act) by default; `--capability experiment|read` down-scopes for caution. Best where a clarification
  SHAPES the work (build/spec) or the answer is researchable. Triggers: "figure out what I'm missing
  and just do it", "investigate and answer", "resolve the unknowns then respond", "refine this prompt",
  "improve my prompt", "whittle down the unknowns and give me a better prompt", "triage the unknowns",
  "research what you can, judge the rest".
version: 1.2.0
author: agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [investigator, clarifying-questions, value-of-information, grounded-research, autonomous, ollama]
    related_skills: [next-best-questions, ask]
    config:
    - key: investigator.k
      description: Top-K questions to research per round (by rank)
      default: 6
      prompt: How many questions should the investigator research per round?
    - key: investigator.max_rounds
      description: Max investigate-then-re-rank rounds before responding
      default: 3
      prompt: Max investigation rounds?
    - key: investigator.capability
      description: Default capability — act (full agency) | experiment (reversible) | read (read-only)
      default: act
      prompt: Default capability level for the investigator?
    - key: investigator.triage
      description: Route each round's questions to derived/findable/judgment before answering (off = today's research-only behavior)
      default: false
      prompt: Enable triage routing (derived/findable/judgment)?
    - key: investigator.triage_model
      description: Model alias used for the batch FINDABLE/JUDGMENT classification call
      default: fast
      prompt: Which model should classify questions as findable vs judgment?
    - key: investigator.judge_model
      description: Model alias used for autonomous judgment-call decisions
      default: deepseek
      prompt: Which model should make judgment-call decisions?
    - key: investigator.max_assumes
      description: Max autonomous judgment-call decisions accepted per run (including resumed) before overflow routes to research
      default: 6
      prompt: Max autonomous assumptions per run?
    - key: investigator.output
      description: Final output — response (classic final response) | prompt (refined prompt only) | both
      default: response
      prompt: Which output should the investigator produce — response, prompt, or both?
---

# Investigator — resolve the unknowns, then respond

## Overview

This is the **orchestrator** layer that sits on top of the report-only `next-best-questions` ranker. The
ranker decides *what is worth clarifying*; the Investigator goes and **answers it**, then responds.

The loop is **one continuously-growing, append-only context**:

```
tombstones = []                              # answered facts + known gaps
for round in range(max_rounds):
    evidence = facts(tombstones)             # the shared growing context
    ranked   = next-best-questions.run(problem, evidence)   # next-best questions, given everything known
    top      = [q for q in ranked if value >= floor and not answered][:K]    # top-K BY RANK
    if not top: stop "converged"
    for q in top: tombstones += grounded_answer(q)       # full Hermes agent, distilled fact back
final = respond(problem, evidence)           # best response over the enriched context
```

Each round conditions on the *entire* accumulated context, so the model's implicit posterior sharpens
as facts accrue — which is why we **always append** and keep tombstones clean, high-signal facts.

## When to Use

**Use it** when the task's unknowns are *researchable* and a clarification would *shape the work* —
vague build/spec/integration tasks against a real project ("set up CI for this repo", "add export
to the reports page"), where the answerer can go read the codebase/environment and the final
response should be grounded in what it finds.

**Don't use it** for well-specified tasks (the ranker will converge immediately — wasted rounds),
for questions only the user can answer (it surfaces those as clarifying questions rather than
guessing — but if *most* unknowns are user-only, just run `next-best-questions` and ask), or when a
capable agent would naturally self-investigate anyway (the A/B showed the loop is redundant there —
its distinctive value is systematic coverage + user-only constraint surfacing).

### Example (abridged)

```
$ python3 scripts/iterate.py --problem "Set up CI for this repository." --k 2 --max-rounds 2

round 1: rank -> 2 questions worth researching
  ? Which test suites/commands must CI run?        -> ANSWERED: pytest via tests/run.py (README)
  ? Target platform — GitHub Actions or other CI?  -> ANSWERED: GitHub repo, no existing workflows
round 2: rank (with 2 facts folded in) -> top value 0.21 < floor -> stop: converged

FINAL RESPONSE: a .github/workflows/ci.yml running `python3 tests/run.py` on push/PR ...
TOMBSTONES: 2 ANSWERED, 0 NOT_FOUND   stop_reason: converged (natural)
```

## How to run

```bash
# Inside the hermes container, FROM the user's project dir (so the answerer researches the real repo):
python3 ${HERMES_SKILL_DIR}/scripts/iterate.py --problem "<task>"
python3 ${HERMES_SKILL_DIR}/scripts/iterate.py --problem "<task>" --k 6 --max-rounds 3 --capability read
python3 ${HERMES_SKILL_DIR}/scripts/iterate.py --problem "<task>" --triage on --output prompt
python3 ${HERMES_SKILL_DIR}/scripts/iterate.py --problem "<task>" --run-dir $HERMES_HOME/state/inv-<slug>
python3 ${HERMES_SKILL_DIR}/scripts/iterate.py --problem "<task>" --dry-run   # loop logic, no model calls
```

## Durability (`--run-dir`)

A live run is expensive (K questions × rounds × full agent researches); `--run-dir` makes it
**resumable** (artifact-based, the drive.py pattern — no engine dependency):

- Each tombstone is appended to `<run_dir>/tombstones.jsonl` as it lands (line 1 is a header
  `{"kind": "header", "problem_fp": ...}`); re-running with the same dir + problem reloads them,
  the answered-filter skips those questions (fp-normalized: case/punctuation variants dedup),
  and the result reports `n_resumed`. A journal for a *different* problem is rotated to
  `.stale` and the run starts fresh. `rounds`/`k_capped` count the current invocation only.
- The answerer agent is additionally instructed to write each distilled answer to
  `<run_dir>/answer-<fp(question)>.json` (`{"answer": "..."}`); the artifact is read before
  stdout parsing — a timeout or a stdout misclassification no longer loses the answer. Under
  `--capability read` this instruction is **omitted** (the read directive forbids writes; a
  coherent prompt beats per-answer durability) — the journal itself is written by the loop
  process and is unaffected.

Without `--run-dir`, behavior is the previous fully-in-memory run. Callers that loop (e.g.
relentless-solve) pass a per-cycle dir so a crash mid-clarify resumes instead of re-researching.

## Module layout

`scripts/iterate.py` — the convergence loop, tombstone journal, CLI. `scripts/answerer.py` —
the `ask`-skill seam: `grounded_answer`/`respond` dispatch, the stdout-salvage fallback
(`_extract`), and the answer-artifact capture. iterate re-exports answerer's names for
back-compat.

## Capability ladder

Default is **full agency** — it answers questions by any means (read, experiment, real action),
unattended. `--capability` only **down-scopes**:

| `--capability` | answerer tools | meaning |
|---|---|---|
| **act** (default) | file, web, terminal | full agency, unattended — current behavior |
| **experiment** | file, web, terminal | restricted by directive to **reversible** experiments (scratch/worktree) |
| **read** | file, web | **read-only** — inspect/search only; action-needing questions return NOT_FOUND |

Capability maps to the answerer's toolsets + a prompt directive (`CAPABILITIES` in `scripts/iterate.py`)
— it does not build a separate permission system. See `references/investigator.md`.

## Triage routing (`--triage`)

Each round, before questions are answered, every top-K question is routed down exactly one of
three paths:

| Route | Trigger | Handler | Tombstone `via` |
|---|---|---|---|
| **derived** | the ranker itself marks the question `DERIVED` with a `derived_answer` (no agent call) | consumed directly from the rank result | `derived` |
| **findable** | triage classifies it as an observable fact a tool-using agent could discover | `grounded_answer` (full Hermes agent research, unchanged) | `research` |
| **judgment** | triage classifies it as a preference/decision with no discoverable ground truth | `judgment_call` (conservative, autonomous decision) | `assumed` |

The derived route only fires when triage is on (it turns on the ranker's `auto_derive` step);
with triage off, every question is FINDABLE — today's pre-1.2.0 behavior, unchanged.

**Fail-open guarantees** — every failure mode routes to research, never to a silent skip:
- Triage disabled (`--triage off` / cfg `"triage"` absent or falsy) — no triage call is made,
  every question is FINDABLE.
- The triage call fails, returns malformed JSON, or omits a valid index for a question — that
  question falls back to FINDABLE. Duplicate entries after the first valid route are ignored.
- A JUDGMENT question's judge call fails (dispatch error, malformed JSON, `CANNOT_DECIDE`, or a
  hedge like "the prompt doesn't specify...") — it falls back to `grounded_answer` once.

**`max_assumes` cap** — once `max_assumes` (default 6) judgment calls have been accepted in a run
(counting ones resumed from a prior `--run-dir` journal), further JUDGMENT-routed questions go to
research instead of the judge — this bounds how much of the final output rests on autonomous
decisions rather than discovered facts.

**CLI default vs programmatic default** — `--triage` defaults to `on` and `--output` (below)
defaults to `prompt` on the CLI (the human prompt-refinement entry point); the programmatic
config-key defaults are `off` / `response` — a caller that doesn't pass `triage`/`output` (e.g.
relentless-solve today) sees behavior identical to 1.1.0 until it opts in.

## Output modes (`--output`)

| Mode | Produces | Default when |
|---|---|---|
| `prompt` | the refined prompt only (`refine_prompt`) | CLI default |
| `response` | the classic final response (`respond`), unchanged from pre-1.2.0 | programmatic default (`cfg["output"]` absent) |
| `both` | the refined prompt, then the final response generated FROM the refined prompt (not the original problem) | — |

### The refined-prompt contract

`refine_prompt` rewrites the ORIGINAL prompt into a self-contained, improved version:
- Every ANSWERED fact (researched or derived) is folded in as an explicit constraint or context.
- Every assumed decision is stated as an explicit choice made in the prompt, not a footnote.
- The original intent is preserved; no scope is invented beyond what the evidence establishes.
- The rewritten prompt always ends with an `## Assumptions` section — each assumed decision and
  its rationale, phrased so a human reviewer can **veto** it before the prompt is used.
- If any NOT_FOUND gaps remain, the prompt also ends with an `## Open questions` section — each
  gap stated as "unspecified — implementer may choose" or carried forward as a question.

**Quality bar**: re-ranking the refined prompt through `next-best-questions` fresh should leave a
near-empty bucket — that is the working definition of "the unknowns have been whittled down."

When `--run-dir` is set, `prompt`/`both` modes also write `refined-prompt.md` to the run dir.

## Per-question outcomes (tombstones)

- **ANSWERED** (`Q → A`) — a discovered fact enters the context.
- **NOT_FOUND** (`Q → gap`) — recorded as a known gap; the final response proceeds with a stated
  assumption. (No revival machinery yet — v1.)
- **user-only** — a genuine preference no investigation can resolve → surfaced as a clarifying question.

## Status (v1, validated with caveats)

End-to-end value is **task-dependent** (de-confounded A/B: helps where a clarification shapes the work,
redundant where a capable agent self-investigates). The ranking it relies on is validated in the
agentic domain (realized-change ρ≈0.66). See `next-best-questions/references/evsi-validation-findings.md`.

## Dependency

Depends on the **next-best-questions** ranker (imported in-process, resolved via `HERMES_HOME` or
`INFOGAIN_SCRIPTS_DIR`) and the **ask** skill's `model_utils` (the grounded answerer/responder run a
full Hermes agent via `dispatch_single`; resolved via `HERMES_HOME` or `ASK_SCRIPTS_DIR`).

## Verification

- Loop logic + journal + artifacts (no network): `python3 tests/test_iterate.py` (55 tests).
- Live (in container): `python3 scripts/iterate.py --problem "<task>"` produces a final response;
  `--capability read` confirms down-scoping (and produces no `answer-*.json`); interrupt a
  `--run-dir` run after one tombstone and re-run to confirm resume (`n_resumed` > 0).
- End-to-end A/B harness: `evals/validate_wrapper.py` (baseline vs wrapper, blind-judged).
