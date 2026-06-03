# Achilli Refrain

Memory contradiction detection and cognitive harmonization for Hermes Agent.

Inspired by Debussy's compositional technique of the *refrain* — a recurring
thematic return that transforms the material each cycle. Refrain periodically
returns to the agent's memory, consolidating and harmonizing what it finds.

## What It Does

1. **End-of-session consolidation**: Hooks into `on_session_end`. On each
   turn boundary, checks enough time has elapsed since the last scan (debounce
   default: 1 hour). If so, triggers a `yantrikdb think` consolidation +
   conflict scan.

2. **Status dashboard via `refrain_status`**: Reports the number of open
   conflicts, last consolidation scan results, and a memory health score
   (contradictions / total memories).

3. **Conflict resolution via `resolve_conflict`**: Interactive tool that
   presents conflicting memories to the agent and offers resolution
   strategies: keep_a, keep_b, keep_both, merge.

4. **Memory debouncing**: Since `on_session_end` fires every turn (not just
   at session termination — this is a Hermes core behavior), Refrain
   implements time-based debouncing. It will not run more than once per
   configurable interval regardless of how many turns occur.

## Dependencies

**Requires YantrikDB MCP server.** Refrain calls YantrikDB's MCP tools
(`think`, `conflict`, `memory`) via the agent's tool dispatch. Without
YantrikDB running, Refrain's end-of-session scan will fail gracefully with
a warning.

If you don't use YantrikDB, Refrain will load without errors but its
consolidation scans will be no-ops.

## Enabling

```bash
hermes plugins enable achilli-refrain
# or edit ~/.hermes/config.yaml:
plugins:
  enabled:
    - achilli-refrain
```

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `ACHILLI_REFRAIN_DEBOUNCE` | `3600` | Minimum seconds between consolidation scans |
| `ACHILLI_REFRAIN_RUN_PATTERN_MINING` | `0` | Set to `1` to enable pattern mining (slower) |
| `ACHILLI_REFRAIN_DISABLE` | `unset` | Set to `1` to disable consolidation scans |

## Limitations

- **Debounce required**: `on_session_end` fires every turn in
  `conversation_loop.py`, not once at session termination. Without
  debouncing, a 20-turn conversation would trigger 20 consolidation scans.
  Refrain implements debouncing, but this means consolidation happens on
  the *first* turn after the debounce window expires, not truly "at session
  end."

- **No LLM for auto-resolution**: Conflicts are presented to the agent for
  manual resolution via the `resolve_conflict` tool. Refrain does not
  auto-resolve using LLM (that would require `ctx.llm` and adds cost).
  Future versions may add optional LLM-assisted resolution.

## Architecture

```
Agent runs conversation (N turns)
    |
    v
on_session_end fires (every turn)
    |
    v
Refrain checks: debounce elapsed?
    |
    +-- No  -> return (skip)
    +-- Yes -> yantrikdb think (consolidation + conflict scan)
                  |
                  v
              results logged, conflicts flagged
```

On the next agent message, if conflicts were found, the agent will see
them and can use `resolve_conflict` to address them.
