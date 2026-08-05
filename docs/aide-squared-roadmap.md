# AIDE² Self-Evaluation Roadmap

This document is the canonical status + future-work plan for the AIDE²
self-evaluation primitives in ``agent/``. It exists so reviewers can see
**what works today**, **what is intentionally a stub**, and **what
follow-up work is planned** before any of this becomes a production
self-improvement system.

> **Status as of this PR**: only the data layer is functional. All
> execution paths are intentional ``NotImplementedError`` stubs.

---

## Background

The original Weco AI AIDE² paper describes a recursive self-improvement
agent that:

1. Records task outcomes with separate agent-visible (public) and hidden
   objective (private) signals.
2. Runs a heterogeneous eval suite against the agent at fixed cost budgets.
3. Uses LLM-as-judge to score outputs without exposing the judge prompt to
   the agent.
4. Drives an outer-loop engineer that proposes SKILL.md mutations, validates
   them via the eval suite, and only retains mutations that improve the
   private score.

Hermes' integration of these ideas is split across four modules:

| Module | Role |
|---|---|
| `agent/experience_ledger.py` | Persistent public/private score ledger |
| `agent/eval_harness.py` | Eval definition loader + metric framework |
| `agent/hermes_squared.py` | Outer-loop cycle orchestrator |
| `agent/delegation_evolution.py` | Bandit / fork strategy dispatch |

## What Works Today (functional)

These pieces have tests, are fully implemented, and persist correctly:

- **ExperienceLedger**
  - Record / save / load ``SkillEval`` records
  - Aggregate ``SkillSummary`` (avg public, avg private, success rate,
    cost, days-since-last-eval, staleness score)
  - Reward-hacking detection (`public_private_gap`, `is_suspected_reward_hack`)
  - Stale / needs-improvement / top-N / worst-N queries
  - UTF-8-safe persistence (Windows-footgun fix)

- **EvalHarness — structural pieces**
  - Load eval definitions from ``evals.json`` or ``evals.yaml``
  - Default-eval generation when none exist
  - Deterministic ``private_check`` via subprocess (working)
  - Budget enforcement (cost > budget → failure)
  - Reward-hack detection (public–private gap, near-perfect public +
    failing private)
  - Ledger recording when execution succeeds
  - Stub-state reporting when execution raises
    (``EvalResult.not_implemented=True``)

- **DelegationEvolution — selection algorithm**
  - Bandit-weighted strategy selection with exploration bonus
  - Stagnation detection and counter
  - Strategy fork (pick untried strategy on stagnation)
  - Lineage tracking
  - Persistent state (scores + lineage, bounded history)

- **Hermes² — cycle orchestrator**
  - Read ledger → find worst skills → generate proposals
  - Proposal strategy selection from symptoms (reward-hack, high
    correction rate, low success rate, general optimize)
  - Cycle budget enforcement
  - Report serialization
  - Graceful stub-state handling (proposals rejected, SKILL.md untouched)

## What Is Intentionally a Stub

These raise ``NotImplementedError`` until the corresponding Phase is
landed. Calling them in production would corrupt user data or fabricate
ledger entries, so they refuse to run instead:

| Function | Module | Phase |
|---|---|---|
| ``_simulate_task_execution`` | EvalHarness | Phase 3 |
| ``_run_llm_judge`` | EvalHarness | Phase 3 |
| ``_dispatch_strategy`` | DelegationEvolution | Phase 3 |
| ``_fork_strategy`` | DelegationEvolution | Phase 3 |
| ``_run_validation_eval`` | HermesSquaredEngine | Phase 3 |
| ``_apply_mutation`` | HermesSquaredEngine | Phase 4 |

The stub contract is:
- Tests assert the function raises ``NotImplementedError``.
- ``EvalHarness.run_eval`` catches the exception, sets
  ``EvalResult.not_implemented = True``, and **does not record into the
  ledger** (so callers don't get fake eval data).
- ``HermesSquaredEngine._apply_proposal`` catches the exception, marks
  the proposal as ``rejected_stub``, and **does not modify SKILL.md**
  (so the user's skills are not corrupted by string-append mutations).

## Phase Plan

### Phase 1 — Honest stubs ✅ (this PR)

- Convert all `random.uniform(...)` simulators into ``NotImplementedError``
  stubs.
- Rewrite PR description to make the stub status explicit.
- Add stub-state tests that pin the contract above.
- Delete the string-append ``_apply_mutation`` (was actively harmful).
- **Why this is enough to merge today**: the data layer is real, the
  algorithm layer is real, and the contract that prevents fabrication /
  corruption is tested. Future phases can replace the stubs without
  touching working code.

### Phase 2 — Signal producer

Goal: every skill invocation produces a ``SkillEval`` entry.

Tasks:
- New module ``agent/skill_eval_producer.py`` with a single public API:
  ``record_skill_invocation(skill_id, public_signal, private_signals)``.
- Hook from Hermes' turn-completion path (likely
  ``run_agent.py`` turn-finalizer or ``gateway/run.py`` tool-call
  return) to call the producer.
- Public signal source: agent's own self-assessment.
- Private signal sources (in priority order):
  1. **User correction detection**: regex / heuristic on the user's
     next message ("不对", "wrong", "重新", "redo", etc.) within a
     short window after the skill ran.
  2. **Rework count**: same task re-issued within N minutes.
  3. **Reuse success**: next invocation of the same skill succeeded.
- Backpressure: producer must not block turn completion — write to
  ledger async or batch.

Risk: where to hook. Hermes' turn loop is the natural seam but
requires careful study of `run_agent.py` to find the right insertion
point without invalidating the prompt cache.

### Phase 3 — Real eval runner

Goal: ``EvalHarness._simulate_task_execution`` and
``_run_llm_judge`` produce real results.

Tasks:
- Replace ``_simulate_task_execution`` with a call to
  ``auxiliary_client`` that starts an isolated chat session running
  ``ev.prompt``, captures the output, and returns the billed cost.
- Replace ``_run_llm_judge`` with an ``auxiliary_client`` call to a
  blind judge prompt (the agent never sees it).
- Hardening for ``private_check`` security:
  - Reject shell commands that try to write outside a sandbox path.
  - Whitelist known-safe commands (``test -f``, ``python3 -c "..."``).
  - Or run in a Docker container / gVisor sandbox.
- Round-trip ledger entries to reflect real measured costs.

Risk: auxiliary_client integration depends on model availability and
may incur real API cost on every cycle. Tests must use the
``hermes-home`` mocking pattern already established in
``tests/agent/test_eval_harness.py``.

### Phase 4 — Real LLM-driven mutation

Goal: ``HermesSquaredEngine._apply_mutation`` returns LLM-generated
SKILL.md content, not hard-coded string appendages.

Tasks:
- Replace ``_apply_mutation`` with an ``auxiliary_client`` call to a
  rewriting LLM. Prompt template:
  - Current SKILL.md
  - Eval summary (private score, correction rate, failure mode)
  - Proposed strategy (``add_validation`` / ``rewrite_skill`` /
    ``fundamental_rewrite`` / ``optimize``)
  - Instruction: "Output ONLY the new full SKILL.md content, no
    commentary."
- Apply via file_ops V4A patch system (already exists in
  ``file_ops.py``) for atomic write + rollback on failure.
- Backup ``SKILL.md`` to ``SKILL.md.bak`` before apply.
- Remove the 4 hard-coded strategy branches; let the LLM decide.

Risk: the rewriting LLM might generate worse SKILL.md than the
original. Phase 4 must include **strict acceptance**: every mutation
is run through ``EvalHarness.run_eval`` first; only mutations that
improve private score are kept. If validation fails, restore from
backup.

### Phase 5 — Concurrency, CLI, observability

Goal: production-safe operation.

Tasks:
- File locking for the three JSON files (``portalocker`` for
  cross-platform support; ``fcntl.flock`` on POSIX, ``msvcrt`` on
  Windows).
- Or replace JSON with SQLite (cleanest concurrency story; bonus
  queryability).
- CLI subcommands:
  - ``hermes eval run <eval_id>``
  - ``hermes eval list``
  - ``hermes ledger show <skill_id>``
  - ``hermes evolve cycle`` (manual trigger of Hermes²)
- Metrics export: structured ``state/evolution_metrics.json`` written
  by every cycle (rejection rate, cost per cycle, accepted mutations).

## Why We Stopped at Phase 1

The temptation was to push all five phases into one PR. We chose not
to because:

1. **Phases 2–4 each require touching different parts of Hermes** —
   Phase 2 needs turn-loop hooks (core), Phase 3 needs
   auxiliary_client (production runtime), Phase 4 needs LLM
   coordination (cost + latency). Bundling them hides which phase is
   actually unblock-able for the maintainer.
2. **The maintainer can land Phase 1 today** — it's a pure stub-ization
   plus honest documentation. The other phases deserve their own PR
   with their own design discussion.
3. **The stubs are load-bearing** — Phase 2's producer cannot ship
   without Phase 1's stub contract, otherwise we'd start writing
   half-real eval data into the ledger.

## How To Review This PR

If you only have five minutes:

1. Read the docstring of each of the four modules — they each have a
   clear `⚠️ STUB IMPLEMENTATION WARNING ⚠️` block.
2. Read ``tests/agent/test_aide_squared_stubs.py`` — it pins the
   stub contract.
3. Run ``pytest tests/agent/test_aide_squared_stubs.py`` — it should
   take <1 second and pass.

If you have more time, read this roadmap and the inline comments on
each stub function. Every stub says what the real implementation will
do and which Phase it belongs to.

## Out of Scope (Deliberately)

The following were considered and explicitly rejected for this PR:

- **Automatic cron scheduling.** Hermes² should run on a schedule the
  user controls via existing ``hermes cron`` machinery, not embedded
  in this PR. The docstring shows how to wire it.
- **Web UI.** The data model is JSON; a UI can be added later.
- **Multi-agent consensus.** AIDE² is single-agent recursive; we
  mirror that scope. Federation across devices is a separate concern.
- **Telemetry / opt-out gating.** Per project policy, no analytics
  land without explicit opt-in. The ExperienceLedger is local-only.