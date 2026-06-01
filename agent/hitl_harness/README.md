# HITL Agent Harness

Production-grade autonomous improvement engine for Hermes Agent.

## Components

| File | Purpose |
|------|---------|
| `__init__.py` | `HITLHarness` — central orchestrator with lifecycle hooks |
| `fitness_wizard.py` | `FitnessFunctionWizard` — intent → GOAL.md generation |
| `fitness_engine.py` | `FitnessEngine` — executable fitness function runner |
| `state_locking.py` | `StateLockingOptimizer` — monotonic improvement enforcement |
| `deepeval_adapter.py` | `DeepevalAdapter` — agentic metrics without external deps |
| `iterations_logger.py` | `IterationsLogger` — first-class `iterations.jsonl` logging |
| `continuous_mode.py` | `run_continuous_goal()` — converge/continuous/supervised loops |
| `telemetry_feedback.py` | `TelemetryFeedback` — production signals → fitness function |
| `goal_commands.py` | CLI commands: `goal create`, `goal run`, `goal status` |
| `graduated_autonomy.py` | `GraduatedAutonomyGate` — L0–L4 autonomy levels |
| `multi_agent_verification.py` | `MultiAgentVerification` — confidence-calibrated verification |
| `layered_observability.py` | `LayeredObservability` — 4-layer explanation stack |

## Quick Reference

```python
from agent.hitl_harness import HITLHarness, HITLHarnessConfig
from agent.hitl_harness.continuous_mode import run_continuous_goal
from agent.hitl_harness.telemetry_feedback import TelemetryFeedback

# The harness auto-attaches to AIAgent when hitl.enabled: true
agent = AIAgent(..., hitl_config={"enabled": True})

# Or run a standalone continuous improvement loop
result = run_continuous_goal(
    goal_path="~/.hermes/goals/my-goal/GOAL.md",
    mode="converge",
    target_score=0.95
)

# Connect production telemetry
feedback = TelemetryFeedback(agent)
feedback.on_deployment(agent.session_id)
```

## Lifecycle Hooks

The harness wires into the agent loop via:

- `on_session_start(session_id, system_message)`
- `on_iteration_start(messages)`
- `pre_tool_call(tool_call)` → block reason or None
- `post_tool_call(tool_call, result)` → annotated result
- `on_iteration_end(response, trajectory)` → mutated response
- `on_session_end()`

See `conversation_loop.py` for the call sites.
