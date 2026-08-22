# Compression Canary Test Pattern

A class-level pattern for certifying that Hermes context compression preserves mission continuity via MISSION_STATE + CompressionRecoveryGate.

## When to Use

When a mission requires proving that real context compression (not mocked) preserves mission state and that the recovery gate correctly restores execution continuity.

## Prerequisites

- `ZENMUX_API_KEY` must be available in the environment (required for real compression via zenmux provider)
- Hermes compression must be enabled in config (`compression.enabled: true`)
- The `mission_state.py`, `compression_recovery.py`, and `conversation_compression.py` modules must be present

## Setup

1. Set up MISSION_STATE at `~/.hermes/mission_state.json` with all required fields:
   - `mission_id`, `objective`, `canonical_repo`, `codegraph_project`
   - `current_phase`
   - `>= 5 known_facts`, `>= 3 completed_steps`
   - `current_blocker`, `blocking_unknown`, `next_discriminating_evidence`
   - `exactly ONE next_action`
   - `>= 2 forbidden_retries`

2. Record PRE_COMPRESSION snapshot from machine state (do not rely on LLM recollection).

## Triggering Real Compression

Real compression requires crossing the token threshold (50% of model context length by default). For Ling 3 Flash (262K context), this is ~131K tokens.

Options to trigger compression:
- Run a sufficiently long conversation that naturally crosses the threshold
- Temporarily lower `compression.threshold` in config for testing (restore after)
- Use the `/compress` slash command for manual compression (not automatic)

**Note**: `compress_context()` requires a valid zenmux API key. Without `ZENMUX_API_KEY`, real compression cannot be triggered.

## Verification Sequence

After compression, verify:

1. **Compression marker detected**: `[CONTEXT COMPACTION — REFERENCE ONLY]` present in messages
2. **Session rotated**: `session_id` changed after compression
3. **MISSION_STATE reloaded**: State persisted to disk and reloaded post-compression
4. **Recovery gate activated**: `CompressionRecoveryGate.detect_compression()` returns True
5. **State preserved**: All fields (objective, canonical_repo, codegraph_project, current_phase, known_facts, completed_steps, current_blocker, blocking_unknown, next_action, forbidden_retries) match pre-compression values
6. **Action continuity**: `FIRST_MATERIAL_ACTION_AFTER_RECOVERY == PRE_COMPRESSION_NEXT_ACTION`

## Receipt Format

Produce a receipt with these required fields:
- `REAL_COMPRESSION.crossed` (must be true for certification)
- `RECOVERY_GATE.detected`, `recovery_complete`
- `STATE_PRESERVATION` for all fields
- `ACTION_CONTINUITY.match`
- `POST_COMPRESSION_20_CALL_METRICS` (classify first 20 post-compression tool calls)
- `FINAL`: `REAL_COMPRESSION_CONTINUITY_CERTIFIED` or `REAL_COMPRESSION_CONTINUITY_FAILED` or `BLOCKED_REAL_COMPRESSION_NOT_REACHED`

## Pitfalls

- **Missing ZENMUX_API_KEY**: The most common blocker. Check `~/.hermes/.env` and environment. Without it, compression cannot be triggered via the zenmux provider.
- **Threshold too high**: Default 50% of 262K = 131K tokens. May require many conversation turns. Lower the threshold temporarily for testing.
- **Mocking compression**: Unit tests and mocked compression do NOT satisfy the real compression requirement. Must observe actual `compress_context()` execution.
- **Manual recovery invocation**: Calling `execute_recovery()` directly does NOT satisfy the requirement. The recovery gate must detect compression automatically.

## Related

- `references/convergence-governor-pattern.md` — P9 Context Compaction Continuity
- `references/behavioral-certification-pattern.md` — SNAPSHOT -> DECISION -> RECEIPT pattern