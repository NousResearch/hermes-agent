# OpenClaw signal mapping

Use this reference when Agent Pulse runs inside an OpenClaw-style runtime.

## Goal

Estimate availability from visible low-cost signals, not hidden self-judgment.

## Cheap signal sources

### runningTask
Set `runningTask=true` when any of these are true:
- a foreground tool task is still underway
- a background `exec` or process run is still active
- work has clearly started and has not been delivered yet

### queuedMessages
Approximate from visible unmet demand:
- `0` when there is no obvious backlog
- `1-2` when there is one pending ask or a small follow-up queue
- `3+` only when multiple distinct asks are genuinely waiting

Do not count routine system notices, async completion banners, or heartbeat noise as queued user work.

### blocked / waitingExternal
Set `blocked=true` when progress is stopped on a missing dependency, failed command, permissions gate, or human decision.
Set `waitingExternal=true` when the blocker is outside the agent, for example waiting on the user, a remote service, CI, or another system.

### recentState
Use the nearest honest label:
- `working` when actively advancing a task
- `blocked` when stopped on a dependency
- `waiting` when paused for someone else
- `idle` when no active work is underway

### activeProject
Set `activeProject=true` when there is an in-flight workstream even if no tool is running this exact second.

### pendingActions
Count concrete undelivered next actions already implied by current work.
Examples:
- edits made but not reported
- verification still pending
- a message still needs to be sent
- a push or deploy step is still outstanding

### hasStartedWork
Set `hasStartedWork=true` once the agent has begun doing the requested task, even if the next step is quiet.

### deliveryDue
Set `deliveryDue=true` when the agent is in the middle of producing a result that should land soon and interruption would meaningfully disrupt it.

## Quick mapping advice

- Recently active and still carrying an undelivered result usually means `light` or `busy`, not `idle`.
- Waiting on the user is often `blocked` plus `waitingExternal=true`, which keeps interruptibility higher than hard-blocked tool failure.
- If evidence is weak, return `unknown` instead of pretending certainty.
