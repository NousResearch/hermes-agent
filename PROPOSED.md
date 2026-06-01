# PROPOSED: Production-Grade Autonomous Coding Agent Evaluation in Hermes Agent

**Date**: 2026-06-01  
**Branch context**: `agent-harness-hitl` (current)  
**Goal**: Embed the full research findings (SWE-bench-style real-world evaluation, GOAL.md fitness functions from goal-md, deepeval-style agentic metrics, monotonic improvement loops, dual scoring, production telemetry, iterations.jsonl) directly into the Hermes Agent core loop and extension points.

## Executive Summary

Hermes Agent already has a strong foundation via the `agent/hitl_harness/` module (Fitness Wizard → GOAL.md generation, State Locking for monotonic progress, Multi-Agent Verification, Layered Observability, Graduated Autonomy). 

The proposal does **not** replace this work. Instead it:
- Makes the existing harness production-hardened and first-class.
- Adds missing pieces from the research (deepeval agentic metrics, dual scoring, `iterations.jsonl` as a core artifact, cron-driven Continuous mode, native SWE-bench integration).
- Exposes everything through the existing skill/plugin/config system so users and sub-agents can opt in without forking the core loop.

## Research Alignment

| Research Finding                  | Current Hermes Support                  | Proposed Enhancement |
|-----------------------------------|-----------------------------------------|----------------------|
| GOAL.md fitness functions (goal-md) | `fitness_wizard.py` generates GOAL.md  | First-class `GOAL.md` loader + executor skill; dual scoring |
| Monotonic improvement + iterations.jsonl | `state_locking.py` + config entry     | Promote `iterations.jsonl` to core session artifact; automatic logging in every loop |
| Agentic metrics (deepeval)        | Partial via observability plugin       | Native `deepeval`-style metrics (Task Completion, Tool Correctness, Step Efficiency, Plan Adherence, Goal Accuracy) inside fitness functions |
| SWE-bench / real GitHub issues    | `mini_swe_runner.py`                   | Tight integration: harness can run SWE-style tasks as fitness components |
| Production telemetry feedback     | Layered observability + plugins        | Bidirectional loop: runtime metrics (error rate, latency, incidents) → GOAL.md refinement |
| Continuous / Converge modes       | Graduated autonomy (L0–L4)             | Explicit `mode: continuous | converge | supervised` in GOAL.md + cron integration |
| Dual scoring (metric + instrument) | Not explicit                           | Core primitive in fitness evaluation |

## Proposed Changes

### 1. Core Loop Integration (`run_agent.py` + `agent/conversation_loop.py`)

- Add optional `goal_md_path` / `goal_md_content` parameter to `AIAgent`.
- On session start: load `GOAL.md` (from repo root, `~/.hermes/goals/<session>/GOAL.md`, or explicit path).
- On every iteration end: invoke harness `on_iteration_end()` which now **always** runs the fitness function defined in GOAL.md (or falls back to default production readiness composite).
- Guarantee: only commit/accept changes that improve the primary score (enforced by State Locking).
- New hook: `on_fitness_evaluated(score, components, dual_instrument_score)`.

**Files touched**:
- `run_agent.py` — extend `__init__` and `run_conversation`
- `agent/conversation_loop.py` — call harness at iteration boundaries
- `agent/agent_init.py` — auto-instantiate enhanced `HITLHarness` when `hitl.enabled: true`

### 2. New / Enhanced `agent/hitl_harness/` Components

#### 2.1 `fitness_engine.py` (new)
- Executable fitness function runner.
- Supports both deterministic metrics (coverage, security scan pass %, perf) and deepeval-style LLM-as-a-judge / agentic metrics.
- Implements **dual scoring**: primary score + instrument_trust_score.
- Parses `GOAL.md` YAML frontmatter + action catalog.
- Exposes `evaluate(trajectory, context) -> FitnessResult`.

#### 2.2 `iterations_logger.py` (new)
- First-class `iterations.jsonl` writer (one line per iteration: timestamp, before/after scores, action, accepted/reverted, rationale).
- Integrated with `hermes_state.py` so sessions can query historical iterations.
- Automatic compression / archival policy.

#### 2.3 `deepeval_adapter.py` (new)
- Thin adapter that brings deepeval metrics into Hermes without requiring the external package (or uses it when available).
- Implements: `TaskCompletion`, `ToolCorrectness`, `StepEfficiency`, `PlanAdherence`, `GoalAccuracy`, `ArgumentCorrectness`, plus G-Eval custom criteria.
- Registered as a core metric provider in the fitness engine.

#### 2.4 Enhance existing files
- `fitness_wizard.py` — add production templates (`production_readiness`, `security_hardening`, `maintainability`, `observability_coverage`).
- `state_locking.py` — expose `dual_score` gate.
- `layered_observability.py` — add production telemetry layer (pull from observability plugin + runtime hooks).

### 3. Skill & Plugin Surface

- New built-in skill: `goal-driven-development` (in `skills/` or promoted from `optional-skills/`).
  - Commands: `/goal create`, `/goal run`, `/goal status`, `/goal converge`.
  - Loads `GOAL.md` and drives the agent using the fitness engine.
- Observability plugin enhancements: emit fitness scores, dual instrument trust, and iteration deltas as structured events.
- Achievements plugin: new achievements for "monotonic streak", "production fitness milestone", "SWE-bench resolution".

### 4. Cron & Continuous Mode

- Extend `cron/` scheduler to support `mode: continuous` GOAL.md jobs.
- A cron job can load a `GOAL.md`, run the autonomous loop overnight, and report via Telegram/email/webhook when target reached or stalled.
- Use existing `cronjob` skill infrastructure.

### 5. SWE-bench / Real-World Task Integration

- Promote `mini_swe_runner.py` integration: the fitness engine can include "SWE-bench resolution rate on private corpus" as a metric component.
- Add `swe_bench` toolset / environment that reuses Hermes execution backends (local, docker, modal).

### 6. Configuration & UX

Update `hermes_cli/config.py` (and example configs):

```yaml
hitl:
  enabled: true
  goal_md:
    auto_load: true
    default_path: null          # or ~/.hermes/goals/{session_id}/GOAL.md
    dual_scoring: true
  metrics:
    include_deepeval: true
    production_readiness_weight: 0.4
    security_weight: 0.25
    maintainability_weight: 0.2
    agentic_efficiency_weight: 0.15
  iterations:
    log_path: "~/.hermes/iterations/{session_id}.jsonl"
    max_without_improvement: 5
  mode: converge | continuous | supervised
```

CLI additions:
- `hermes goal create "Improve production readiness of checkout flow"`
- `hermes goal run --converge`
- `hermes goal status`

### 7. Documentation & Migration

- New docs: `website/docs/guides/goal-driven-development.md`
- Update `AGENTS.md` and `CODEBASE_MAP.md`
- Migration guide for existing users of the HITL harness.

## Implementation Phases (Recommended)

**Phase 1 (High impact, low risk)**  
- Promote `iterations.jsonl` to first-class artifact + automatic logging.  
- Add `deepeval_adapter.py` and wire into existing fitness wizard.  
- Dual scoring primitive in `state_locking.py`.

**Phase 2**  
- `fitness_engine.py` + GOAL.md loader skill.  
- Cron Continuous mode support.

**Phase 3**  
- Production telemetry feedback loop.  
- Full SWE-bench integration + public examples.

## Non-Goals / Out of Scope (for this proposal)

- Replacing the existing HITL harness architecture.
- Forcing every session to use GOAL.md (always opt-in or config-driven).
- Bundling the full deepeval package (adapter only).

## Success Metrics

- 100% of autonomous runs produce `iterations.jsonl` with monotonic score improvement.
- Dual instrument trust score > 0.7 on production tasks.
- Measurable reduction in "good enough but brittle" code reaching production (tracked via post-deploy signals).

---

This proposal turns the excellent existing HITL harness work into a **production-grade, research-backed autonomous improvement engine** while staying faithful to Hermes' existing plugin/skill/config philosophy.

Ready to implement? I can start with Phase 1 files immediately.