# Achilli Tendo

Agent session checkpoint and resume for Hermes Agent.

Tendo serializes the agent's working state at session boundaries so it can
be resumed later -- surviving crashes, provider outages, or intentional
pauses. Named after the *tendo* (tendon) that transfers force when the
primary attachment fails.

## What It Does

1. **Session checkpointing via `on_session_finalize`**: When a session ends
   (CLI quit, /new, /reset, gateway session expiry), Tendo serializes:
   - Full session transcript (messages from state.db)
   - Active background processes and their session IDs
   - Active subagent IDs and their goals
   - Current working directory
   - Timestamp and completeness score

2. **Checkpoint listing via `list_checkpoints`**: Browse checkpoint history
   with metadata -- when created, session ID, reason, completeness.

3. **State management**: Checkpoints stored in
   `~/.hermes/checkpoints/agent-state/<session_id>.json`.

## What It Cannot Do (Yet)

- **Resume is model-mediated**: Tendo can create checkpoints, but "resuming"
  means creating a new session and injecting the checkpoint transcript as
   context. The agent sees its past work and continues. This is not a
   true "hot resume" -- the new session has a fresh context window.

- **Background processes**: If a background process was alive at checkpoint
   time, Tendo records its PID. On resume, the process is NOT restarted.
   The agent must re-spawn it.

- **File handles**: Open file handles cannot be serialized. Tendo records
   which files were recently read/written but cannot "re-open" them.

- **YantrikDB session context**: Tendo attempts to record the YantrikDB
   session ID (if one exists via recall). On resume, the agent should
   restart its YantrikDB session for full continuity.

## Enabling

```bash
hermes plugins enable achilli-tendo
# or edit ~/.hermes/config.yaml:
plugins:
  enabled:
    - achilli-tendo
```

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `ACHILLI_TENDO_MAX_CHECKPOINTS` | `50` | Max checkpoints to retain (oldest evicted) |
| `ACHILLI_TENDO_DIR` | `~/.hermes/checkpoints/agent-state` | Checkpoint directory |
| `ACHILLI_TENDO_DISABLE` | `unset` | Set to `1` to disable checkpointing |

## Architecture

```
Session runs (CLI / gateway)
    |
    v
CLI quit / /new / /reset / GC
    |
    v
on_session_finalize fires (WORKS -- confirmed in cli.py line 972, 6542)
    |
    v
Tendo._on_session_finalize()
    |
    +-- Saves transcript to JSON
    +-- Records background processes
    +-- Records active subagents
    +-- Writes checkpoint file

Later: Agent can list_checkpoints, read JSON, create new session with context
```

## Checkpoint Format

```json
{
  "session_id": "abc123",
  "checkpoint_time": "2026-06-03T19:30:00Z",
  "reason": "session_finalize",
  "message_count": 42,
  "working_directory": "/d/HermesPlace",
  "background_processes": {},
  "active_subagents": [],
  "yantrikdb_session_id": null,
  "completeness_score": 0.85,
  "continuation_prompt": "Session resumed from checkpoint..."
}
```

## Dependencies

- Requires Hermes Agent >= 0.15.1 (for `on_session_finalize` hook)
- Optional: YantrikDB (for session context continuity)
- No external packages required
