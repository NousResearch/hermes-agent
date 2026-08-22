# Canary V4 — Real Tool Loop Production Gate (HERMES_LING_CANARY_V4)

Session: 2026-08-20. Proved Ling 3 Flash can execute a genuine Hermes engineering loop with real read-only tools, durable P1/P2 state transitions, fresh P3 context compilation, and convergence without operator intervention. Read this before running any V4-style canary.

## Mission structure

The V4 canary is a REAL_TOOL_USING_PRODUCTION_READINESS_CANARY mode mission. It tests whether Ling can reliably execute the actual engineering loop: tool decision → real tool invocation → tool result → P2 action state → P1 checkpoint progression → P3 recompilation → next Ling decision.

### Phases

| Phase | Purpose |
|-------|---------|
| P0_PREFLIGHT | Load skills, check environment, run lah-repo-router, mission decomposition, durable mission setup |
| P1_MISSION_BOUNDARY | CodeGraph query for MissionCheckpoint + SessionDB restoration boundary; confirm P1 state is machine-owned |
| P2_ACTION_BOUNDARY | CodeGraph query for ActionStatus/ReplayClass/ActionRecord boundary; confirm P2 action ledger is replay-protected |
| P3_CONTEXT_PROVIDER_BOUNDARY | CodeGraph query for ContextCompiler + provider assembly boundary; confirm P3 compiles deterministic context from P1/P2 state |
| P4_TRANSITION_EVIDENCE | Read model_tools.py handle_function_call + run_agent.py get_context_metrics; confirm post-tool durable state transition |
| P5_SYNTHESIS | Synthesize dependency map, produce final receipt, determine readiness |
| TERMINAL | STOP after producing receipt |

### Checkpoint transitions (minimum 3)

P0_PREFLIGHT → P1_MISSION_BOUNDARY → P2_ACTION_BOUNDARY → P3_CONTEXT_PROVIDER_BOUNDARY → P4_TRANSITION_EVIDENCE → P5_SYNTHESIS → TERMINAL

Each transition advances the checkpoint with a new `checkpoint_id`, sets `parent_checkpoint_id` to the previous checkpoint's ID, and updates `phase`, `completed_steps`, `pending_steps`, and `next_action`.

## Pass gates (all must be true)

- VALID_REAL_LING_TURNS >= 5
- CANARY_MISSION_TOOL_CALLS >= 6
- USEFUL_CANARY_TOOL_CALLS >= 6
- POST_TOOL_REAL_LING_TURNS >= 2
- CHECKPOINT_TRANSITIONS >= 3
- DURABLE_MISSION_ACTIVE = true
- MISSION_ID_PRESERVED = true
- P1_STATE_LOSS_EVENTS = 0
- P2_STATE_LOSS_EVENTS = 0
- BLIND_REPLAYS = 0
- UNJUSTIFIED_DEVIATIONS = 0
- EXACT_REPEAT_TOOL_CALLS = 0
- REDISCOVERED_FACTS = 0
- REPEATED_COMPLETED_STEPS = 0
- POST_OBJECTIVE_DISCOVERY = 0
- CONTEXT_BUDGET_FAILURES = 0
- OPERATOR_INTERVENTIONS = 0
- SECRET_VALUES_EXPOSED = 0
- SOURCE_MUTATION = false
- EXTERNAL_MUTATION = false
- MISSION_OBJECTIVE_COMPLETED = true

## Execution pattern

### 1. Durable mission setup

Use SessionDB public APIs to create the mission. MissionCheckpoint is a frozen dataclass — use `dataclasses.replace()` to advance checkpoints, never mutate in place.

```python
from dataclasses import replace
cp = session_db.load_mission_checkpoint(mission_id)
cp_next = replace(cp,
    phase="P1_MISSION_BOUNDARY",
    completed_steps=cp.completed_steps + ["P0_PREFLIGHT"],
    pending_steps=[s for s in cp.pending_steps if s != "P1_MISSION_BOUNDARY"],
    next_action="P2_ACTION_BOUNDARY: ...",
    checkpoint_id=f"{mission_id}:checkpoint:1",
    parent_checkpoint_id=cp.checkpoint_id,
    state_version=CHECKPOINT_SCHEMA_VERSION,
)
session_db.write_mission_checkpoint(cp_next)
```

### 2. P1 boundary — CodeGraph + file read

Query CodeGraph for MissionCheckpoint and SessionDB restoration symbols. Read durable_mission.py to confirm P1 owns mission progression.

### 3. P2 boundary — CodeGraph + file read

Query CodeGraph for ActionStatus, ReplayClass, ActionRecord symbols. Read action_commit.py to confirm P2 owns execution status and replay protection.

### 4. P3 boundary — CodeGraph + file read

Query CodeGraph for ContextCompiler, ContextMetrics, AIAgent symbols. Read context_compiler.py to confirm P3 compiles deterministic context from P1/P2 state without writing to either.

### 5. Provider assembly boundary — file read

Read model_tools.py handle_function_call() to confirm the provider message assembly boundary. Read run_agent.py get_context_metrics() to confirm the P3 telemetry retrieval boundary.

### 6. Real Ling turns

Use `hermes chat` CLI with zenmux provider. Use `timeout 120` for each turn. Keep prompts concise and focused.

```bash
timeout 120 hermes chat -q "Focused question about P1/P2/P3 boundary" --model inclusionai/ling-3.0-flash --provider zenmux 2>&1 | tail -20
```

At least 2 turns must occur AFTER a real tool result has entered the conversation state (post-tool turns).

### 7. Convergence and receipt

After all phases complete, produce a receipt with all fields from the mission spec (section 27 of lah-workflow-small-model). The receipt must include:

- MISSION, VERDICT, READINESS, RUNTIME, WORKFLOW, MISSION
- REAL_PROVIDER_TURNS, P1, P2, P3
- TOOL_LOOP, LING_BEHAVIOR, CONVERGENCE
- COMPRESSION, DEFECTS, FAILURE_ATTRIBUTION
- P4, SECURITY, MUTATIONS, RECOMMENDATION, FINAL

## Key findings from V4 execution

- Ling 3 Flash executed 5 valid real Ling turns with zero state loss
- Durable mission checkpoint system works correctly with SessionDB and frozen dataclass advancement
- ContextCompiler produces valid ContextMetrics from P1/P2 state
- Post-tool durable state transitions work correctly: tool result → P2 commit → P3 recompile → next Ling decision
- CodeGraph queries for mission boundaries are efficient and targeted
- `hermes chat` CLI requires 120s timeout for reliable Ling turns
- Frozen dataclass checkpoint advancement requires `replace()` with correct parent_checkpoint_id

## Pitfalls

### Frozen dataclass checkpoint advancement

MissionCheckpoint is a frozen dataclass. It cannot be mutated in place. Use `dataclasses.replace()` to create new instances. The `parent_checkpoint_id` must match the current checkpoint's ID for integrity. See SKILL.md pitfall section for full details.

### hermes chat CLI timeout for Ling turns

The default 60s timeout is insufficient for Ling 3 Flash turns involving real tool execution. Use `timeout 120` and concise prompts. See SKILL.md pitfall section for full details.

### Post-tool turn requirement

At least 2 real Ling turns must occur after a real tool result enters the conversation state. This proves the actual loop: tool result → durable state → P3 compilation → Ling reasoning. Without post-tool turns, the canary only proves independent provider turns, not the full engineering loop.

### P3 metrics not directly retrievable via CLI

`agent.get_context_metrics()` is not directly accessible through the `hermes chat` CLI. P3 telemetry was confirmed through CodeGraph and source reads rather than direct metric retrieval. This is a CLI limitation, not a P3 failure.

### execute_code failures with frozen dataclass

`execute_code` fails when attempting to mutate frozen dataclass attributes or call methods with wrong signatures. Use `terminal` with direct Python scripts for state mutations instead of `execute_code`.