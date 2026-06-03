# Achilli Chorus

Multi-agent output harmonization and conflict resolution for Hermes Agent.

When you spawn multiple subagents in parallel (via `delegate_task` with a
`tasks` array), each child returns independently. The parent agent must then
manually synthesize the results — comparing, merging, resolving conflicts.

Chorus automates this synthesis.

## What It Does

1. **Result collection**: Every `subagent_stop` event is intercepted. The
   child's summary and metadata are buffered in an in-memory batch.

2. **Harmonization via `ctx.llm`**: When all siblings in a batch have
   completed, Chorus uses the host's LLM (via `ctx.llm`) to:
   - Identify conflicting recommendations
   - Detect overlapping work
   - Find gaps (neither child addressed a required component)
   - Generate a unified recommendation
   - Flag items requiring human review

3. **Status dashboard via `chorus_status`**: Returns the current state of
   the harmonization buffer — how many siblings completed, conflict count,
   synthesis readiness.

4. **Synthesis trigger via `on_session_end`**: Checks if a complete batch
   is ready for harmonization. If so, runs the LLM synthesis and injects
   the result via `ctx.inject_message()`.

## Important Design Notes

- `subagent_stop` provides the child's **summary string**, not the raw
  output text. Chorus works with what the summary provides. Detailed output
  harmonization should be done by the parent agent after reading full results.

- Chorus uses `ctx.llm` for synthesis (~500-2000 tokens per harmonization
  call). This draws from the host's API quota. For tight budgets, set
  `ACHILLI_CHORUS_MAX_TOKENS` to limit synthesis length.

- `ctx.inject_message()` is used to deliver harmonized results to the
  parent agent. If called mid-turn (agent is running), it interrupts the
  turn. Chorus only fires on `on_session_end` (turn boundary) to avoid
  this.

## Batch Detection

Chorus considers subagents to be in the same "batch" if they complete within
a configurable time window (default 60s between first and last child).
Children completing outside this window are treated as separate batches.

## Enabling

```bash
hermes plugins enable achilli-chorus
# or edit ~/.hermes/config.yaml:
plugins:
  enabled:
    - achilli-chorus
```

## Dependencies

- Requires Hermes Agent >= 0.15.1 (for `subagent_stop` hook and `ctx.llm`)
- Requires `delegate_task` (built-in) for spawning subagents
- Incompatible with other plugins that use `ctx.inject_message()` (only one
  plugin can safely inject mid-turn)

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `ACHILLI_CHORUS_BATCH_WINDOW` | `60` | Max seconds between first and last child to be treated as same batch |
| `ACHILLI_CHORUS_MAX_TOKENS` | `1000` | Max tokens for LLM synthesis call |
| `ACHILLI_CHORUS_DISABLE_LLM` | `unset` | If set to `1`, skip LLM synthesis; collect and report only |

## Architecture

```
Parent spawns [child A, child B, child C]
      |
      v
child A finishes -> subagent_stop -> Chorus collects
child B finishes -> subagent_stop -> Chorus collects
child C finishes -> subagent_stop -> Chorus collects (batch complete)
      |
      v
on_session_end -> Chorus runs LLM synthesis -> injects harmonized result
      |
      v
Parent sees harmonized summary in next turn
```
