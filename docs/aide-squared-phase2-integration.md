# AIDE² Phase 2 — Integrating SkillEvalProducer

Phase 1 shipped the data layer (ExperienceLedger + EvalHarness +
HermesSquared + DelegationEvolution). Phase 2 ships the **producer** —
the bridge that turns real Hermes turns into ``SkillEval`` entries.

This document is the integration guide for the maintainer. It assumes
you will land the producer + producer tests in this PR; what is **not**
in this PR is the runtime hook (the call site inside ``run_agent.py``
or ``gateway/run.py``) because the right hook point depends on runtime
considerations the maintainer should own.

## What's in this PR

| File | Purpose |
|---|---|
| `agent/skill_eval_producer.py` | Core producer — `SkillEvalProducer` + `TurnSignals` dataclass |
| `agent/signal_sources/__init__.py` | Package marker |
| `agent/signal_sources/user_correction_detector.py` | Heuristic correction detector (EN/CN/ES/FR/DE) |
| `agent/signal_sources/rework_detector.py` | Sliding-window rework counter |
| `agent/signal_sources/reuse_tracker.py` | Per-skill reuse → outcome tracker |
| `agent/hermes_eval_hook.py` | Worked example: `wrap_turn()` for turn-finalizers |
| `tests/agent/test_skill_eval_producer.py` | Producer tests |
| `tests/agent/test_signal_sources/` | Detector unit tests |

## What you (the maintainer) wire next

A 3-line insertion somewhere in the turn loop. Recommended site:
``run_agent.py``'s turn-finalizer (the function that runs after each
agent turn completes). If that's not reachable cleanly, the gateway
tool-call return wrapper is a fine fallback.

### Option A — wrap_turn at the turn finalizer

```python
# In run_agent.py, after a turn completes successfully:
from agent.hermes_eval_hook import wrap_turn
from pathlib import Path

record = wrap_turn(
    skill_id=detected_skill_id,      # skill used in this turn, if any
    task_id=current_task_id,
    task_family=current_task_family,
    hermes_home=Path.home() / ".hermes",
    public_signal=agent_self_score,   # 0.0–1.0
    cost_usd=billed_cost,
    tokens_in=tokens_in,
    tokens_out=tokens_out,
    duration_sec=turn_duration,
    success=turn_succeeded,
    user_messages=[m.text for m in last_3_user_msgs],
)
```

### Option B — call the producer directly

If `wrap_turn`'s defaults don't fit, instantiate `SkillEvalProducer`
yourself and call `record_turn(TurnSignals(...))`. The producer is
the single source of truth for signal aggregation.

### Option C — gateway tool-call wrapper

Wrap the gateway's tool-call return so each tool invocation that maps
to a skill produces an eval. Higher volume per turn but coarser
granularity — one record per tool call, not per turn.

## Why the integration is not in this PR

- The right hook point requires understanding the prompt cache
  contract — adding work inside an existing turn loop risks breaking
  the prompt caching guarantee. The maintainer should own that
  decision.
- The producer's correctness is fully testable in isolation; the
  hook is one line of glue code that the maintainer can review
  alongside their other turn-loop changes.
- If Phase 2 ends up needing an extra field (e.g. `request_id` for
  correlation with gateway logs), exposing it via `TurnSignals` is a
  one-line addition — we don't want to lock the signature before the
  hook site is known.

## What the producer does (for reviewers)

For each `record_turn` call, the producer:

1. Validates inputs (`public_signal` in [0,1], `cost_usd >= 0`, etc.).
2. Computes three private signals:
   - **`user_corrected`**: runs the follow-up user messages through
     `user_correction_detector.detect`. CJK + EN + ES + FR + DE
     patterns by default. `user_corrected_override=True` skips the
     detector.
   - **`rework_count`**: counts how many `task_id` retries happened
     in the last 10 minutes (default). Uses caller-supplied
     `rework_events` plus an in-memory tracker. `rework_count_override`
     skips the computation.
   - **`reuse_success`**: looks up the next entry in the per-skill
     reuse history after the current timestamp. `None` if no further
     reuse yet. `reuse_success_override` skips the lookup.
3. Estimates a private score via a transparent heuristic:
   `public_signal − 0.4 (if corrected) − 0.15 × rework_count − 0.2 (if reuse_failed)`.
   Clamped to [0, 1]. This is a placeholder for the LLM judge that
   Phase 3 will plug in via `EvalHarness._run_llm_judge`.
4. Writes one `SkillEval` to the ledger. `record_turn` saves on every
   call by default; `record_batch` saves once at the end.
5. Logs at INFO with the signal values so a maintainer can audit
   what was recorded.

## Testing

```bash
.venv/bin/python -m pytest tests/agent/test_skill_eval_producer.py \
                              tests/agent/test_signal_sources/ -v
```

Tests cover:

- Validation (reject out-of-range `public_signal`, empty ids, etc.)
- Each detector in isolation (EN/CN positive + negative cases)
- Producer end-to-end with `tmp_path` HERMES_HOME
- Batch save semantics (`auto_save=False` + flush)
- Overrides vs detector paths
- Private score heuristic clamping

## Phase 3 will hook into this

Once Phase 3 lands `EvalHarness._run_llm_judge` and
`_simulate_task_execution`, the producer's private-score heuristic
stays in place for raw turn records (it's the per-turn signal). For
eval-driven scores, the caller can override `public_signal` and
`user_corrected` from the eval result and let the producer compute
the rest. Alternatively, callers can directly construct a `SkillEval`
and skip the heuristic — both paths are supported.

## Out of scope (Phase 2 deliberately does not)

- Connecting to a real turn finalizer (`run_agent.py` change).
- Persisting per-skill reuse history across restarts — currently
  in-memory only. A future Phase 5 PR can persist it alongside the
  ledger.
- Cross-device aggregation. That's Phase 2 of the federation plan,
  not AIDE² Phase 2.
- An async producer. Calls are sync; if write latency matters at
  high volume, use `record_batch` and call from a worker thread.