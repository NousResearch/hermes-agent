# Spec delta: cron process isolation

## ADDED Requirements

### Requirement: supervised agent execution
Agent-backed cron runs MUST execute in a child process group that the parent scheduler registers before authorizing execution. The parent MUST own result handling, runtime state, execution-ledger transitions, and cleanup.

#### Scenario: child handshake and registration
- **WHEN** an agent-backed job is dispatched
- **THEN** the child sends `ready`, waits for parent authorization, and the parent registers it before sending `start`
- **AND** a child that exits before producing a result is recorded as failed after it is reaped

### Requirement: authoritative inactivity termination
When inactivity timeout fires, the scheduler MUST interrupt the agent and use a bounded wait. If the worker remains non-cooperative, an isolated child MUST terminate rather than wait indefinitely on a running `Future`.

#### Scenario: non-cooperative inactivity timeout
- **GIVEN** `HERMES_CRON_TIMEOUT` is positive and the agent reports no activity for that limit
- **WHEN** the agent ignores `interrupt()`
- **THEN** timeout cleanup returns within the termination grace bound
- **AND** the isolated child exits so the parent can reap it and release scheduler ownership

### Requirement: bounded wall-time cleanup
When `cron.max_runtime_seconds` is positive, the parent MUST terminate the child with bounded SIGTERM/SIGKILL escalation and join it before removing process registration. A value of zero MUST remain an unlimited wall-time opt-out.

#### Scenario: wall timeout escalation
- **WHEN** a child exceeds its configured wall limit
- **THEN** the parent marks it cancelling, sends termination, escalates when needed, joins/reaps it, and reports a timeout

### Requirement: cwd and script isolation
The process-isolation path MUST preserve the existing cwd lock/restore contract, and script-only jobs MUST retain their existing script timeout behavior without agent construction.

#### Scenario: script-only job
- **WHEN** a job has `no_agent=True`
- **THEN** it executes through the script path and is not shortened by the agent wall-clock setting
