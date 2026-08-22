# Canary V3 — Real Mission Readiness Gate (HERMES_LING_CANARY_V3)

Session: 2026-08-20. Executed a realistic read-only engineering mission
through the durable execution path to certify Ling 3 Flash for daily Hermes use.
Read this before running any V3-style canary.

## Mission structure

The V3 canary is a REAL_MISSION_CANARY_MEASURE_AND_CERTIFY mode mission.
It executes an actual engineering audit (not synthetic prompts) and measures
P1/P2/P3 metrics against strict pass gates.

### Phases

| Phase | Purpose |
|-------|---------|
| P0_PREFLIGHT | Load skills, check environment, run lah-repo-router, mission decomposition |
| P1_DURABLE_START | Initialize durable mission, verify checkpoints, bind session |
| P2_EXECUTION_FLOW | Execute realistic engineering mission, track P1/P2 metrics |
| P3_CONTEXT_FLOW | Track context telemetry per Ling turn, measure compression |
| P4_COMPRESSION_CONTINUITY | Observe compression/session rotation effects on state |
| P5_RISK_ASSESSMENT | Identify defects, risks, and P4 candidates |
| P6_SYNTHESIS | Produce final readiness report and receipt |
| TERMINAL | STOP after producing receipt |

### Pass gates (all must be true)

- VALID_REAL_LING_TURNS >= 8 (or mission naturally completes with strong evidence)
- DURABLE_MISSION_ACTIVE = true
- MISSION_ID_PRESERVED = true
- P1_STATE_LOSS_EVENTS = 0
- P2_BLIND_REPLAYS = 0
- UNJUSTIFIED_DEVIATIONS = 0
- REPEATED_COMPLETED_STEPS = 0
- POST_OBJECTIVE_DISCOVERY = 0
- CONTEXT_BUDGET_FAILURES = 0
- SECRET_VALUES_EXPOSED=*** OPERATOR_INTERVENTIONS = 0
- Mission objective successfully completed

## Execution pattern

### 1. Durable mission setup

Use SessionDB public APIs to create the mission:

```python
session_db.create_session(session_id, "canary-v3")
session_db.create_mission(mission_id, root_session_id=session_id)
checkpoint = MissionCheckpoint(
    mission_id=mission_id,
    checkpoint_id=f"{mission_id}:checkpoint:0",
    parent_checkpoint_id=None,
    state_version=CHECKPOINT_SCHEMA_VERSION,
    objective=objective,
    phase="P0_PREFLIGHT",
    completed_steps=[],
    pending_steps=["P1_DURABLE_START", ...],
    next_action="P1_DURABLE_START: verify checkpoint and bind session",
    status="ACTIVE",
    canonical_repo=repo_path,
    repo_observed_head=sha,
    codegraph_project=repo_path,
)
session_db.write_mission_checkpoint(checkpoint)
```

Verify binding: `session_db.get_mission_for_session(session_id)` must return
the correct mission_id.

### 2. Agent construction

Use canonical ZenMux credential resolution only:

```python
_env = load_env()
agent = AIAgent(
    model="inclusionai/ling-3.0-flash",
    provider="zenmux",
    api_key=_env["ZENMUX_API_KEY"],
    base_url="https://zenmux.ai/api/v1",
    session_db=session_db,
    session_id=session_id,
    mission_id=mission_id,
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    enabled_toolsets=[],  # read-only mission
    max_iterations=5,
    verbose_logging=False,
)
```

Note: `agent.session_id` is set dynamically by `init_agent()` and is not
recognized by static analysis (pyright). Use `getattr(agent, 'session_id', fallback)`
to avoid false positives.

### 3. Harness execution

Use DurableBenchmarkHarness for turn-level observation:

```python
harness = DurableBenchmarkHarness(agent, session_db, mission_id)
obs = harness.run_turn(user_message)
metrics = agent.get_context_metrics()
```

After each turn, record:
- checkpoint_before vs checkpoint_after (for P1 transitions)
- metrics raw/compiled/hot/warm/recent tokens (for P3)
- result status (for P2)

### 4. Metrics collection

P1: mission_id_changes, checkpoint_transitions, next_action_loss_events
P2: actions_recorded, committed, failed, blind_replays
P3: per-turn raw/compiled/hot/warm/recent tokens, compression_count
Convergence: total_tool_calls, useful/low_info, exact/semantic repeats, rediscovered_facts

### 5. Receipt format

The receipt must include all fields from the mission spec (section 27):
MISSION, VERDICT, RUNTIME, MISSION, REAL_PROVIDER_TURNS, P1, P2, P3,
LING_BEHAVIOR, CONVERGENCE, COMPRESSION, DEFECTS, P4, SECURITY, MUTATIONS,
RECOMMENDATION, FINAL.

FINAL must be exactly one of:
LING_CANARY_V3_PASS | LING_CANARY_V3_PASS_WITH_LIMITATIONS |
LING_CANARY_V3_FAIL | LING_CANARY_V3_INSUFFICIENT_EVIDENCE |
CANARY_BLOCKED_DURABLE_RUNTIME | CANARY_BLOCKED_REPO_AUTHORITY |
CANARY_BLOCKED_PROVIDER | CANARY_BLOCKED_RUNTIME_DEFECT

## Key findings from V3 execution

- Ling 3 Flash executed 10 realistic engineering turns with zero state loss
- Durable mission checkpoint system works correctly with SessionDB
- ContextCompiler produces valid ContextMetrics (raw/compiled/hot/warm/recent)
- No compression triggered at moderate context levels (expected)
- Canonical credential resolution via load_env()/get_env_value() works correctly
- The convergence governor bootstrap gate activates correctly before discovery

## Pitfalls

### session_id attribute not recognized by static analysis

`agent.session_id` is set dynamically in `init_agent()` and pyright flags it
as unknown. Use `getattr(agent, 'session_id', SESSION_ID)` to avoid false
positives. The attribute works at runtime.

### Certified architecture context data files may not exist

The `certified-architecture-context-and-resume` skill references data files
(`data/certified-architecture-context.json`, `data/architecture-fingerprint.json`,
`data/mission-resume-packet.json`) and scripts that may not exist on disk.
Before relying on the startup sequence, verify these files exist. If missing,
use the convergence governor bootstrap gate and CodeGraph directly as a fallback.

### reduction_percent can be negative

Compiled context tokens often exceed raw transcript tokens because the compiled
context includes the full system prompt, tool definitions, and conversation
history. A negative reduction_percent is expected and does not indicate a bug.

### Read-only missions have zero tool calls

When `enabled_toolsets=[]`, the agent makes no tool calls. Convergence metrics
for tool calls will be zero. This is expected for read-only audit missions.