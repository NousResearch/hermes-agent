# REAL_RUNTIME_MEASURE_ONLY Benchmark Execution Pattern

Session: HERMES_LING_REAL_CONTEXT_COMPILER_RUNTIME_BENCHMARK_V1 (2026-08-20)

## Telemetry Surface

`AIAgent.get_context_metrics()` returns a `ContextMetrics` object with:
- `raw_transcript_tokens` — raw token count before compilation
- `compiled_context_tokens` — tokens after ContextCompiler compilation
- `hot_state_tokens` — hot (frequently accessed) state tokens
- `warm_state_tokens` — warm (recently accessed) state tokens
- `recent_conversation_tokens` — tokens from recent conversation turns
- `reserved_headroom` — reserved context headroom (typically 20% of context window)
- `compression_count` — number of compressions applied
- `compression_distance_turns` — turns since last compression

## Per-Turn Metrics Capture

Use `step_callback` (not `stream_callback`) to capture metrics after each turn:

- `step_callback(turn_num, prev_tools)` — called after each API call in the conversation loop
- `stream_callback` — called for text deltas during streaming, NOT per turn

The `build_turn_context()` function (in `agent/turn_context.py`) is called once at conversation start, not per turn. Therefore:
- `ContextCompiler` metrics are captured once and do NOT update per turn
- `compiled_context_tokens` will be the same value across all turns
- `raw_transcript_tokens` reflects the current transcript state and may change

## Durable Mission Checkpoint System

The durable mission system (`agent/durable_mission.py`) requires:
1. A `SessionDB` instance passed as `session_db` to `AIAgent`
2. The session must be bound to a mission in the SessionDB
3. Without these, `restore_mission_for_turn()` raises `MissionCheckpointRequiredError`

For benchmark missions that don't need durable checkpoints, omit `mission_id`, `session_id`, and `checkpoints_enabled` from the AIAgent constructor.

## Common Pitfalls

### compiled_context_tokens=0
If `compiled_context_tokens` is 0 for all turns, this means the ContextCompiler produced an empty compiled context. This can happen when:
- The conversation is cut short by iteration budget before the compiler populates metrics
- The `build_turn_context()` was called but the compiler returned no messages

### Durable Mission Requires SessionDB
Error: "durable mission requires SessionDB"
Fix: Either provide a `SessionDB` instance or omit `mission_id`/`checkpoints_enabled`.

### Session Not Bound to Durable Mission
Error: "session is not bound to durable mission"
Fix: The session must be created in the SessionDB and bound to a mission before the durable mission system will activate. For measurement-only missions, disable checkpoints.

### Iteration Budget Exhausted
If the conversation reaches `max_iterations` without completing, the result will have `completed: false`. For complex tasks, increase `max_iterations` or scope the task more tightly.