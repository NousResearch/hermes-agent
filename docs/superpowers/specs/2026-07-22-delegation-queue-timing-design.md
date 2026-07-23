# Delegation Queue Timing Design

## Goal

Show truthful queue wait and execution durations in the subagent roster so queued work is not mistaken for running work.

## User-visible behavior

Each child row displays timing for its current lifecycle state:

- Pending: `queued 42m`
- Running: `running 3m · queued 42m`
- Completed: `completed in 8m · queued 42m`
- Failed or interrupted: preserve the terminal state and show execution plus queue history when both timestamps exist.

Queue history remains visible after execution starts and after terminal completion.

## Lifecycle data

Delegation records carry three distinct timestamps:

- `queued_at`: task accepted for delegation and exposed as pending.
- `started_at`: worker receives an execution slot and begins running.
- `ended_at`: worker reaches a terminal state.

Durations derive from these timestamps:

- Pending queue duration: `now - queued_at`
- Final queue duration: `started_at - queued_at`
- Running duration: `now - started_at`
- Final execution duration: `ended_at - started_at`

Values must never be negative. Missing or malformed timestamps omit only the unavailable duration and preserve existing roster rendering.

## Architecture

### Delegation lifecycle

Capture `queued_at` at the point where `delegate_task` accepts and records a child before scheduler admission. Capture `started_at` only at the authoritative transition from pending to running. Capture `ended_at` at the existing terminal transition.

Single-child and batch delegation use the same lifecycle fields. No timestamp is inferred from Telegram message time, roster refresh time, process discovery, or a renderer-local clock origin.

### Async records

Persist lifecycle timestamps in the long-lived async delegation record used by `_async_delegation_watcher`. Finished children disappear from the active registry, so terminal roster rendering must retain the record timestamps instead of depending on `list_active_subagents()`.

Existing records without the additional fields remain renderable.

### Roster rendering

A pure timing formatter receives child state, lifecycle timestamps, and `now`. It returns compact timing text without mutating records.

The background roster watcher continues owning one edited-in-place bubble. Its idle refresh advances pending and running durations under existing throttling; this feature must not create additional Telegram sends.

## Error handling

- Missing timestamp: omit unavailable duration.
- Non-numeric timestamp: treat as missing.
- `started_at < queued_at`: omit queue duration.
- `ended_at < started_at`: omit execution duration.
- Pending child with a stray `started_at`: render from authoritative state and valid fields; do not silently relabel lifecycle state.
- Terminal child without `ended_at`: preserve terminal state and show only durations that can be calculated safely.

No compatibility configuration or migration is added. Graceful rendering of in-memory records created before deployment is required because the gateway may observe mixed record shapes during shutdown or tests.

## Testing

Focused tests cover:

1. Pending child increments `queued` duration from `queued_at`.
2. Running child displays execution duration and frozen queue duration.
3. Completed child displays final execution duration and queue history.
4. Failed/interrupted terminal states retain honest markers and timing.
5. Single-child and batch records carry the same lifecycle fields.
6. Missing, malformed, and inverted timestamps fail safely.
7. Background roster refresh updates elapsed text by editing the existing bubble, not sending another message.
8. Existing records without lifecycle timestamps retain current rendering.

Run focused delegation lifecycle, subagent roster, async watcher, and gateway progress tests. Use a temporary `HERMES_HOME` where live plugins could affect imports.

## Scope

Included:

- Delegation lifecycle timestamps
- Long-lived async delegation records
- Subagent roster timing renderer
- Focused regression tests

Excluded:

- Scheduler policy or concurrency changes
- Queue prioritization
- New model tools or tool-schema fields
- New user configuration
- Heartbeat, todo, workflow-plugin, or ordinary tool-progress redesign
- Additional Telegram messages

## Acceptance criteria

- Pending row clearly says `queued` and shows live wait duration.
- Running row shows `running` duration plus preserved queue duration.
- Terminal row preserves queue history when timestamps exist.
- A task queued for 42 minutes and running for seconds cannot appear as merely `0s`.
- Timing derives from authoritative lifecycle transitions.
- Single and batch delegation behave consistently.
- Legacy/missing timestamp records render without failure.
- Focused tests pass.
- Gateway restart boundary is reported because runtime code is startup-loaded.
