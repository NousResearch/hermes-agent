type: spec-delta
project: hermes-agent (upstream PR)
capability: gateway-deferred-restart

# Spec Delta: gateway-deferred-restart

## ADDED Requirements

### Requirement: Deferred restart CLI
The gateway CLI SHALL provide `hermes gateway defer-restart` with no
subcommand-specific parameters. It SHALL request restart acceptance and return
without waiting for drain completion, a replacement process, or gateway
readiness. Parser usage errors SHALL retain the parser's normal exit behavior.

#### Scenario: supervised gateway accepts the request
- WHEN an authorized chat turn invokes `hermes gateway defer-restart` for a supported supervised gateway
- THEN the CLI exits 0 only after receiving a valid positive acceptance response
- AND it states that acceptance is confirmed but restart completion and readiness are not

#### Scenario: unsupported argument
- WHEN an invocation supplies a delay, force, all-profiles, wait-ready, or unknown argument
- THEN parser validation rejects the invocation without sending a control mutation

### Requirement: Existing lifecycle protections remain unchanged
The existing CLI self-refusal and terminal lifecycle guard for ordinary gateway
restart, stop, uninstall, and equivalent self-targeting lifecycle commands SHALL
remain unchanged. The implementation SHALL NOT add a lifecycle-guard allowlist.

#### Scenario: ordinary self-restart remains blocked
- WHEN a gateway-hosted agent attempts an existing blocked restart, stop, or uninstall command shape
- THEN the existing guard behavior remains blocked
- AND `hermes gateway defer-restart` uses only its separate constrained control channel

### Requirement: Marked cron agent execution is refused before mutation
The CLI handler SHALL check the existing `HERMES_CRON_SESSION` provenance marker
before target resolution or any control mutation. A matched official cron
context SHALL exit non-zero and SHALL NOT send the deferred-restart verb. This
is a misuse guard for marked cron agent subprocesses, not a same-user permission
boundary. Unmarked no-agent and pre-agent script paths are an explicit v1
coverage gap; this change SHALL NOT claim complete cron provenance coverage.

#### Scenario: marked cron invocation is refused
- WHEN `HERMES_CRON_SESSION` identifies an official cron execution
- THEN the command exits 1 before any gateway mutation request

#### Scenario: unmarked scheduler script boundary is documented
- WHEN an official no-agent or pre-agent script runs without `HERMES_CRON_SESSION`
- THEN the v1 documentation identifies that path as outside this marker-based refusal guarantee

### Requirement: Target resolution fails closed
The CLI SHALL target a confirmed live gateway owned by the selected profile when
one exists. A nullable PID, missing socket response, timeout, starting owner,
held owner lock, malformed ownership record, or inaccessible identity SHALL NOT
be treated as proof of absence. Only when an own instance is positively known
not to exist MAY the CLI use the host
returned by `host_multiplexer_serving(selected_profile)`, and only when that
helper confirms that the host serves the profile and belongs to the same Hermes
tenant. The CLI SHALL reject an unreachable or ambiguous own instance,
standalone exclusion, cross-tenant result, unconfirmed host relationship, or
resolution failure. It SHALL NOT inherit a legacy default-home fallback as
identity proof. Identify and mutation SHALL target the same canonically verified
home.

#### Scenario: selected profile owns a live gateway
- WHEN the selected profile has a confirmed own live gateway
- THEN identify and mutation requests target that gateway's home

#### Scenario: selected profile is a confirmed satellite
- WHEN no own gateway exists and `host_multiplexer_serving()` confirms a same-tenant host serving the profile
- THEN identify and mutation requests target the host gateway's home
- AND successful output states that all profiles served by that gateway are affected

#### Scenario: target ownership cannot be confirmed
- WHEN an own instance is unreachable or ownership/host service relationship is ambiguous
- THEN the command exits 1 without guessing another home or sending a mutation

### Requirement: The receiving gateway enforces restart-capable supervision
The gateway SHALL expose a distinct `defer-restart` control verb while preserving
`pause-for-update` unchanged. Before scheduling restart, the receiving gateway
SHALL reject non-POSIX hosts. On a supported POSIX host it
SHALL require its own detected supervisor to be `systemd`, `launchd`, or
`external`, and SHALL require `is_gateway_supervisor_process()` to return true.
A failed check SHALL return a structured rejection and SHALL NOT call
`request_restart()`.

#### Scenario: supported supervisor
- WHEN the actual receiving gateway reports an allowed supervisor and the existing restart-capable supervisor helper returns true
- THEN the handler may request restart on that gateway's event loop

#### Scenario: instance changed after CLI preflight
- WHEN CLI preflight observed a supported instance but an unsupported manual instance receives the mutation
- THEN the receiving instance rejects the request without calling `request_restart()`

#### Scenario: unsupported supervisor
- WHEN the receiving gateway is manual, desktop, Windows-only, s6-without-supported-identity, missing, unknown, or inconsistent
- THEN the handler returns `accepted: false` with reason `unsupported_supervisor`
- AND it does not initiate shutdown

#### Scenario: Windows external-supervisor context remains unsupported
- WHEN a native Windows gateway carries the explicit external-supervisor marker
- THEN the handler rejects before event-loop handoff and does not call `request_restart()`

### Requirement: Restart acceptance has an explicit tri-state result
After supervisor admission, the handler SHALL validate the accepting runner's
already-loaded drain timeout. A non-finite budget SHALL produce a structured
`unsupported_drain_timeout` rejection before restart mutation; the handler SHALL
NOT reread, clamp, or alter the existing configuration.

The event-loop callback SHALL reject an ordinary stop already represented by
the runner's existing stop state before calling `request_restart()`. The check
SHALL run at callback execution time, including for a callback that runs after
the handler deadline, and SHALL return reason `already_stopping` without changing
the stop's restart intent.

After those admission checks, the handler SHALL marshal
`request_restart(detached=False, via_service=True)` onto the receiving gateway's
event loop. It SHALL distinguish a positive return, a negative return, and an
exception or bounded-wait timeout.

A positive return SHALL produce a result containing `accepted: true`, a positive
integer PID, an allowed supervisor string, and a finite non-negative
`drain_timeout`, with no `reason` field. A negative return SHALL produce
`accepted: false` with reason `already_stopping`. An exception or result timeout
SHALL cause the existing control protocol to return an error rather than infer a
negative or positive result.

#### Scenario: current runner accepts restart
- WHEN `request_restart()` returns true
- THEN the handler returns a valid positive acceptance result

#### Scenario: current runner already has a restart flow
- WHEN `request_restart()` returns false
- THEN the handler returns `accepted: false` and `reason: already_stopping`
- AND the response is not reported as a newly accepted request

#### Scenario: callback raises
- WHEN the event-loop callback raises before returning a result
- THEN the control response is an error and the CLI treats the outcome as UNKNOWN

#### Scenario: callback result is late
- WHEN no callback result is available within the handler's bounded wait
- THEN the control response is an error and the CLI treats the outcome as UNKNOWN
- AND the implementation does not claim that the queued callback was cancelled

#### Scenario: ordinary stop wins before callback execution
- WHEN an ordinary stop begins before the event-loop callback executes
- THEN the callback returns a non-new `already_stopping` rejection
- AND it creates no restart task and does not change the existing stop intent

#### Scenario: loaded drain budget cannot be represented
- WHEN the accepting runner's loaded drain timeout is non-finite
- THEN the handler returns `accepted: false` with reason `unsupported_drain_timeout`
- AND it does not call `request_restart()`

### Requirement: CLI success requires a strict positive ACK
The CLI SHALL exit 0 only when the response has `accepted is True`; a positive
integer PID that is not a boolean; a supervisor in the supported set consistent
with preflight; a finite non-negative numeric `drain_timeout` that is not a
boolean; and no `reason` field. Extra unrelated fields MAY be ignored.

A structurally valid negative ACK SHALL contain literal `accepted: false`, one
recognized reason from `already_stopping`, `unsupported_supervisor`, or
`unsupported_drain_timeout`, and only reason-appropriate optional metadata. Its
PID, supervisor, and timeout fields, when present, SHALL obey the same strict
type rules as the positive schema. Missing, wrongly typed, unknown-reason, or
contradictory negative results are UNKNOWN.

A valid negative result SHALL exit 1 as a rejection or non-new acceptance. A
missing, malformed, contradictory, transport-lost, or
otherwise unverifiable result SHALL exit 1 as UNKNOWN. UNKNOWN output SHALL say
that the gateway may already be processing the request and SHALL instruct the
caller to inspect status rather than automatically retry. The CLI SHALL NOT
retry or fall back to updater pause, a signal, or ordinary restart.

#### Scenario: valid positive ACK
- WHEN every strict success field is valid and consistent
- THEN the CLI exits 0 and reports the accepting PID and supervisor

#### Scenario: malformed false ACK
- WHEN a result contains `accepted: false` but is structurally contradictory or malformed
- THEN the CLI treats it as UNKNOWN rather than as a reliable rejection

#### Scenario: response is lost after acceptance
- WHEN the gateway may have accepted restart but the CLI receives no valid response
- THEN the CLI exits 1 with UNKNOWN wording and performs no automatic resubmission

### Requirement: Existing restart and delivery semantics are reused honestly
The feature SHALL reuse the current runner-local restart idempotency, active-work
wait, drain, service exit, session recovery, and delivery ledger behavior. It
SHALL NOT modify `request_restart()`, delivery persistence, cron scheduling, or
session recovery.

Success SHALL mean only that the current runner newly accepted a restart request.
It SHALL NOT mean that drain completed, a replacement is ready, the platform send
API completed before exit, or the reply is exactly-once. Eligible persisted
replies MAY be redelivered after restart and MAY carry ambiguity or duplication
semantics. No cross-process request deduplication, generation binding, persisted
request state, cooldown, audit, or new provenance guarantee is provided.

#### Scenario: active work delays shutdown after ACK
- WHEN restart is accepted while eligible active work remains
- THEN the acceptance response can return before shutdown
- AND stop proceeds only after the existing wait policy releases or reaches its existing bound

#### Scenario: duplicate request in one runner
- WHEN a second deferred request reaches a runner whose restart task is already started
- THEN no second restart task is created
- AND the second request is not reported as newly accepted

### Requirement: Unsupported and older gateways fail without fallback
An older gateway that does not implement `defer-restart`, or a gateway whose
capabilities cannot be confirmed, SHALL produce a non-success result. The CLI
SHALL NOT invoke `pause-for-update`, a signal, or ordinary restart as fallback.
The updater's existing `pause-for-update` behavior, including its ability to
pause a manual gateway for an update it will later resume, SHALL remain
unchanged.

#### Scenario: older gateway does not know the verb
- WHEN the selected gateway returns unknown verb or no compatible response
- THEN the CLI exits 1 without invoking another lifecycle mechanism

#### Scenario: updater pause compatibility
- WHEN the existing updater invokes `pause-for-update`
- THEN its existing handler and manual-gateway semantics remain unchanged

### Requirement: End-to-end verification is isolated from the installed gateway
The four-step E2E SHALL run a disposable checkout-backed canary with separate
`HERMES_HOME`, `HERMES_RUNTIME_DIR`, gateway lock directory, HOME/XDG state,
transport identity, ports, and durable conversation store. A small external
supervisor SHALL relaunch the exact reviewed canary child on exit 75 using
`--external-supervisor`. The verification SHALL NOT repoint, install, restart,
or modify the production gateway service.

#### Scenario: canary completes the four-step restart flow
- WHEN the candidate command is invoked from an active canary agent turn
- THEN evidence records positive acceptance from the old PID and a visible final reply on the canary transport
- AND the supervisor launches a different PID from the reviewed checkout
- AND a subsequent turn resumes the same durable canary conversation
