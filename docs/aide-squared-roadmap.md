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

### Phase 2 — Signal producer ✅ (this PR)

- New module `agent/skill_eval_producer.py` with public API
  `SkillEvalProducer.record_turn(TurnSignals)` + `record_batch`.
- New package `agent/signal_sources/`:
  - `user_correction_detector.py` — multi-language regex pre-filter
    (EN/CN/ES/FR/DE), pluggable via `reset_patterns`.
  - `rework_detector.py` — sliding-window retry counter
    (`count_recent`, `count_rework_retries`, `filter_window`).
  - `reuse_tracker.py` — per-skill reuse history with
    `mark_invocation` / `lookup_reuse_outcome` (immediate-only or
    majority-outcome).
- New `agent/hermes_eval_hook.py` — `wrap_turn(...)` reference
  integration showing how a turn-finalizer should call the producer.
  Never raises on producer-side errors — failures are logged at
  WARNING so a broken ledger never breaks a turn.
- Private-score heuristic: `public_signal − 0.4 (corrected)
  − 0.15 × rework_count − 0.2 (reuse_failed)`, clamped to [0, 1].
  Documented as a placeholder for the LLM judge that Phase 3 will
  plug in.
- 49 new tests across `tests/agent/test_skill_eval_producer.py`,
  `tests/agent/test_hermes_eval_hook.py`, and
  `tests/agent/test_signal_sources/`.
- **Why this is enough to merge today**: the producer is fully
  testable in isolation; the runtime hook (one `wrap_turn(...)`
  call inside the turn finalizer) is left to the maintainer to
  avoid risking the prompt cache contract.

### Phase 3 — Real eval runner ✅ (this PR)

- New module ``agent/eval_runner.py``:
  - ``EvalInvocation`` (frozen dataclass) — all inputs for one eval round.
  - ``PromptResult`` / ``PrivateCheckResult`` — structured outputs.
  - ``EvalRunner`` Protocol — the surface ``EvalHarness`` needs.
  - ``DefaultEvalRunner`` — production implementation:
    - ``execute_prompt`` calls :mod:`agent.auxiliary_client.call_llm`
      and extracts OpenAI-style usage (prompt_tokens /
      completion_tokens).
    - ``run_private_check`` runs ``private_check`` via a hardened
      subprocess: argv-explicit invocation of ``/bin/sh -c <cmd>``,
      no ``shell=True``, environment restricted to a small safe
      allowlist, and a dangerous-token regex filter that blocks
      ``sudo`` / ``curl`` / ``wget`` / ``ssh`` / ``python`` /
      ``bash`` etc. unless ``allow_unsafe_private_check=True``.
  - Custom exceptions (``PrivateCheckError``) raised on dangerous
    tokens so callers can distinguish "blocked" from "timed out".
- New module ``agent/llm_judge.py``:
  - ``JudgeScore`` dataclass — score, reasoning, success, error,
    model.
  - ``LLMJudge`` Protocol.
  - ``DefaultLLMJudge`` — calls ``call_llm`` with a fixed judge
    prompt template; ``parse_score_text`` is robust against JSON /
    prose / fenced blocks and rejects out-of-range scores.
- ``EvalHarness`` accepts ``runner=`` and ``judge=`` constructor
  kwargs (dependency injection). The default production runner is
  ``DefaultEvalRunner`` and the default judge is
  ``DefaultLLMJudge``; tests inject fakes without touching
  auxiliary_client.
- ``_simulate_task_execution`` and ``_run_llm_judge`` are now real
  implementations; the Phase 1 stubs are removed.
- 41 new tests across ``tests/agent/test_eval_runner.py`` and
  ``tests/agent/test_llm_judge.py`` covering: prompt execution happy
  / failure paths, private check happy / failure / timeout /
  dangerous-token / env-filter paths, judge parse variants
  (clean JSON, prose-wrapped, fenced, out-of-range, nested braces),
  judge failure paths.
- Existing Phase 1 stub tests updated to reflect the new injection
  contract: ``TestEvalHarnessStub`` now asserts that ``runner=`` and
  ``judge=`` constructor args replace the defaults, rather than
  asserting ``NotImplementedError``.
- **Why this is enough to merge today**: the harness's *execution*
  path now goes through real LLM calls and real subprocesses. The
  default runner will fail loudly when no LLM provider is configured
  (rather than fabricating scores), and the dangerous-token filter
  closes the Phase 1 RCE footgun on ``private_check``.

### Phase 4 — Real LLM-driven mutation ✅ (this PR)

- New module ``agent/skill_muter.py``:
  - ``MutationContext`` (frozen dataclass) — all inputs the mutator
    needs (skill_id, current_content, strategy, scores, notes,
    model_kwargs).
  - ``MutationProposal`` — new_content + reasoning + success/error.
  - ``ApplyResult`` — applier's structured output (success,
    backup_path, error).
  - ``SkillMuter`` / ``SkillMuterApplier`` Protocols — the LLM-bound
    and side-effect-bound surfaces.
  - ``DefaultSkillMuter`` — calls ``auxiliary_client.call_llm`` with
    a structured prompt that includes the skill's evaluation
    summary and the chosen strategy; ``parse_mutation_response``
    handles raw output / fenced blocks / trailing
    ``<reasoning>...</reasoning>`` sections.
  - ``DefaultSkillMuterApplier`` — backup current ``SKILL.md`` to
    ``SKILL.md.bak``, write new content, rollback on failure. No
    diff application (LLM-generated diffs are unreliable; a full
    file is trivially atomic).
- ``HermesSquaredEngine`` accepts ``mutator=`` and ``applier=``
  constructor kwargs (dependency injection).
- ``_apply_mutation`` delegates to ``SkillMuter.mutate`` with the
  skill's ``SkillSummary`` so the LLM can tailor the rewrite to
  the skill's known symptoms (reward-hacking detection,
  correction rate, etc.).
- ``_validate_proposal`` writes the mutation via
  ``SkillMuterApplier.apply`` (creating the backup), runs the eval,
  and rolls back from the backup regardless of outcome. The
  applier's ``rollback`` is the success path; if no backup was
  created, falls back to writing the in-memory original.
- ``_apply_proposal`` permanently writes the mutation via the
  same applier and records the new eval entry in the ledger with
  ``lineage=proposal_id``.
- ``_run_validation_eval`` synthesizes a one-off ``EvalDefinition``
  from the skill's summary, injects it into the harness, and
  returns the measured ``private_score`` + ``cost_usd``. Falls back
  to a structured failure (not an exception) when the eval harness
  is not wired.
- 15 new tests across ``tests/agent/test_skill_muter.py``
  covering: prompt construction, parse variants (raw, fenced,
  reasoning section, empty), mutator happy / failure paths, and
  applier happy / refuse / rollback / new-skill scenarios.
- Existing Phase 1 stub tests updated to reflect the new
  contract: ``TestHermesSquaredStub`` now asserts that
  ``_apply_proposal`` surfaces failures as ``proposal.status =
  "apply_failed"`` (rather than raising), and that
  ``_run_validation_eval`` returns structured failure (rather than
  raising).
- **Why this is enough to merge today**: the SKILL.md is no longer
  mutated by string-append heuristics; it is regenerated by an
  LLM bound through a backup-then-write applier that rolls back on
  failure. Users can cron Hermes² without fear of permanent
  corruption.

### Phase 5 — Concurrency, CLI, observability ✅ (this PR)

- ``Aide2Metrics`` frozen dataclass: structured per-cycle metrics
  (cycle_id, timestamps, proposal counts, rejection_rate,
  total_cost_usd, duration_sec, skill_deltas dict). Suitable for
  Prometheus pushgateway, structured JSON logs, Datadog, etc.
  ``EvolutionReport.to_metrics()`` converts the report to metrics.
- Concurrent proposal validation: ``asyncio.gather`` validates all
  proposals in parallel. Budget tracking is accurate — cost is summed
  after all validations return. Apply phase is sequential (file writes).
  ``_validate_proposal`` exception handling via ``return_exceptions=True``.
- ``HermesAide2CLI`` class: thin CLI wrapper with ``run_cycle()``
  and ``run_status()`` methods. Reads ``~/.hermes/`` as default.
- ``hermes aide2 run [--budget USD] [--max-proposals N]``: runs one
  full improvement cycle synchronously.
- ``hermes aide2 status``: reads the latest ``evolution_report.json``
  and prints a formatted summary (cycle id, acceptance rate, cost,
  per-skill delta scores).
- ``main()`` entry point: usable as ``python -m agent.hermes_squared``
  or imported from the module.
- 9 new tests covering ``Aide2Metrics`` round-trip,
  ``EvolutionReport.to_metrics()`` delta derivation,
  concurrent validation with ``asyncio.gather``,
  exception capture in gather results,
  ``HermesAide2CLI`` status/cycle/main paths.

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