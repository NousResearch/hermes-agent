type: spec-delta
project: hermes-agent (upstream PR)
capability: gateway-deferred-restart

# Spec Delta: gateway-deferred-restart

## ADDED Requirements

### Requirement: Deferred restart request protocol
The gateway CLI SHALL provide a `hermes gateway defer-restart` subcommand with no
parameters. The CLI SHALL hand off the request as a protected request file
(`<hermes_home>/restart_requests/pending-<request_id>.json`) containing
schema_version, request_id, service_identity, target_generation (pid + process
start time), requested_at, and actor_ref (audit-only), written atomically
(0700 directory, 0600 file, tmp+rename+directory fsync) before signaling the
target process with SIGUSR1 as a wake-up only. The request file SHALL NOT carry
CLI-chosen commands, exit codes, delays, or restart methods. The gateway SHALL
refuse symlinked paths and verify file owner and regular-file type.

The acceptance linearization point is the moment the gateway has verified
identity and authorization, atomically claimed the request, and reserved restart
budget such that the restart can be scheduled. Writing the pending file is not
acceptance. If request_restart() declines to schedule, the request SHALL NOT be
recorded as ACCEPTED. Acceptance SHALL be persisted before the generation's
final exit.

Exit codes: 0 = accepted or idempotent duplicate, 2 = usage error, 3 = no
matching gateway target, 4 = target lacks a supervisor-managed restart path,
5 = refused by cross-process cooldown, 6 = outcome unknown (could not prove
acceptance).

#### Scenario: agent triggers deferred restart in a chat turn
- WHEN a chat session's agent executes `hermes gateway defer-restart` via the terminal tool
- THEN the CLI passes resolution/pre-check/cooldown, writes the request file, signals the target, and exits 0 once acceptance of its request_id is confirmed for the target generation

#### Scenario: no matching gateway
- WHEN no gateway process matches the resolved profile identity
- THEN the CLI exits 3 without writing a request file or signaling

#### Scenario: request idempotency scope
- WHEN the same CLI invocation internally retries with its generated request_id
- THEN the gateway returns the existing decision without consuming new cooldown budget
- AND independent CLI invocations are separate requests subject to the cooldown

### Requirement: Turn-completion delivery boundary
The gateway SHALL NOT exit for a deferred restart until the active-work count
reaches zero, where active work includes running agent turns, and a turn's final
reply submission to the platform send API is awaited before its slot is released.
The delivery guarantee is bounded to "platform send API accepted" and does NOT
claim platform delivery or user-visible receipt.

#### Scenario: final reply completes before restart
- WHEN a deferred restart is accepted while an agent turn is running
- THEN the gateway waits for the turn's final reply send to complete and only then proceeds to stop with the service restart exit code

#### Scenario: timeout force-completion
- WHEN the turn does not complete within `agent.restart_after_turn_timeout`
- THEN the gateway force-completes using existing recovery semantics and no lossless guarantee applies on this path

#### Scenario: acceptance handoff when no active work
- WHEN a deferred restart is accepted while active work is already zero
- THEN the acceptance record is persisted before the gateway exits and the CLI can read that record even after the process is gone

### Requirement: Acceptance confirmation bound to target generation
The CLI SHALL confirm acceptance by observing a persisted acceptance record for
its own request_id attributable to the resolved target generation (PID + process
start time). A stale or unattributable restart_requested/draining state SHALL NOT
count as confirmation. When confirmation cannot be proven within the confirmation
window (~5s), the CLI SHALL exit 6 and callers SHALL treat the outcome as unknown
(no immediate unconditional retry).

Runtime status (restart_requested and similar live state) MAY be reset when a new
generation starts, but accepted_request_id history SHALL be retained with a
sufficient retention period and SHALL NOT be cleared at new-process startup.

#### Scenario: confirmation observed
- WHEN the gateway persists ACCEPTED for the CLI's request_id under generation G
- THEN the CLI exits 0 reporting the request as accepted

#### Scenario: stale status not accepted
- WHEN status shows an older restart state predating the CLI's request
- THEN the CLI does not treat it as confirmation and keeps polling until the window closes

#### Scenario: outcome unknown
- WHEN the confirmation window closes without an attributable acceptance record
- THEN the CLI exits 6 and does not immediately re-fire

### Requirement: Cross-process cooldown and idempotency
The deferred restart channel SHALL enforce a cross-process cooldown (default
600s) keyed by a stable gateway service identity: (canonical hermes home,
supervisor type, supervisor service id). Process generation identity SHALL be
used only for target binding and PID-reuse protection, never as the cooldown key.
Within a cooldown window, one service identity SHALL NOT gain additional deferred
restart budget via generation changes, sibling profiles on the same host process,
or duplicate requests (invariant I2). Multiplexed profiles sharing a host process
SHALL resolve to one service identity, authorization scope, and cooldown scope.

#### Scenario: second request inside cooldown window
- WHEN a deferred restart was accepted for service identity S at T0 and a new request for S arrives at T0+120s
- THEN the CLI exits 5 and no request file is signaled through

#### Scenario: cooldown survives gateway restart
- WHEN the gateway restarts (new PID and start time) and a new request arrives within the cooldown window
- THEN the request is refused with exit 5 because the service identity is unchanged

#### Scenario: multiplexed profiles share budget
- WHEN two profiles resolve to the same gateway host process and both request deferred restart within the cooldown window
- THEN at most one request is accepted

### Requirement: Supervisor capability pre-check
Before writing the request file, the CLI SHALL verify that the resolved target is
bound to a supervisor-managed restart path (launchd job loaded with KeepAlive /
restart semantics; systemd unit with a restart policy covering the service exit
code, accounting for RestartPreventExitStatus/SuccessExitStatus combinations, and
not start-limited; or s6 service under supervise). The check SHALL be bound to
the resolved signal target. When no supported supervisor path exists (including
Windows in v1), the CLI SHALL refuse with exit 4 and SHALL NOT signal the process
to die. The precheck promise is limited to "a trusted supervisor-managed restart
path existed at request-acceptance time", not a permanent guarantee.

#### Scenario: launchd managed gateway
- WHEN the resolved target is the process of a loaded launchd job with KeepAlive
- THEN the pre-check passes and the request proceeds

#### Scenario: unmanaged process
- WHEN the resolved target is a foreground/detached gateway with no supervisor restart path
- THEN the CLI exits 4 without signaling

#### Scenario: unsupported platform
- WHEN the command is run on Windows
- THEN the CLI exits 4 with an explicit "unsupported platform" message

### Requirement: SIGUSR1 compatibility preservation
Deferred request processing SHALL NOT change the behavior of pre-existing SIGUSR1
calls that carry no valid deferred request: when no deferred request is present,
the legacy restart behavior SHALL be preserved. When a deferred request is present
but fails validation, the gateway SHALL reject that request and SHALL NOT fall
through to legacy unconditional restart (invariant I3). Signal handlers SHALL NOT
perform blocking file I/O; request reading, locking, and state commit SHALL run in
an appropriate async context within the event loop.

#### Scenario: legacy signal without deferred request
- WHEN SIGUSR1 arrives and no valid deferred request file exists
- THEN the existing SIGUSR1 restart behavior applies unchanged

#### Scenario: invalid deferred request present
- WHEN SIGUSR1 arrives and a deferred request file exists but fails validation
- THEN the request is rejected and logged and no unconditional restart is triggered

### Requirement: Concurrency and crash semantics
At most one deferred request SHALL become the active restart request for a
generation; concurrent pending requests SHALL receive explicit outcomes
(accepted-scheduled, duplicate, cooldown-refused, or expired) and SHALL NOT be
silently dropped. Requests whose target_generation does not match the consuming
gateway SHALL NOT be consumed; a new generation SHALL NOT reinterpret a previous
generation's pending request as a fresh restart instruction. Pending files SHALL
have an expiry/cleanup rule, and cleanup SHALL NOT delete result records for
requests in accepted or outcome-unknown state. A crashed CLI after signaling
leaves the request file which the gateway may still consume; a gateway crash
before claiming leaves the file for expiry cleanup.

#### Scenario: stale generation request ignored
- WHEN a pending request file targets generation G1 and the running gateway is generation G2
- THEN G2 does not consume the request as a restart instruction

#### Scenario: crash after write before signal
- WHEN the CLI writes the request file and crashes before signaling
- THEN the request expires per the cleanup rule without triggering a restart

### Requirement: Guard extensions without relaxing existing guards
The existing terminal-tool lifecycle guard (G1) and CLI self-refusal (G2) SHALL
remain unchanged in their interception semantics; this change adds no new guard
surface and relaxes none. Normal chat turns remain allowed subject to
authorization, capability, and cooldown rules. A cron payload MAY drive the
deferred channel in v1: that is a KNOWN, documented coverage gap (the cross-
process cooldown bounds it to at most one restart per cooldown window per
service), left to a decoupled follow-up change.

#### Scenario: known coverage gap documented
- WHEN a reader consults the deferred restart documentation
- THEN the cron coverage gap and its cooldown bound are stated as a v1 limitation

#### Scenario: existing guard behavior unchanged
- WHEN any previously blocked command shape is submitted (restart/stop/uninstall forms, launchctl forms, kill forms, spliced/continuation/compound variants)
- THEN the existing scanner behavior is unchanged (still blocked)

### Requirement: Message recovery guarantee documentation
Documentation SHALL state: gateway shutdown persists successfully flushed pending
messages (shutdown_flush → `<hermes_home>/pending_messages/`) and the new process
recovers them into session history via recover_pending_to_db(); this guarantees
recoverability of successfully flushed data but does NOT guarantee automatic
re-dispatch as new turns, does NOT guarantee exactly-once delivery, and does NOT
cover abnormal termination (SIGKILL/power loss) before flush. Message re-execution
relies on platform redelivery or user resends.

#### Scenario: documented guarantee level
- WHEN a user reads the deferred restart documentation
- THEN the flush/recovery boundary, no-automatic-redispatch semantics, and platform-redelivery dependency are explicitly stated
