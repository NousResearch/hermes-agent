type: spec-proposal
project: hermes-agent (upstream PR)
task: gateway-deferred-restart-slim
status: draft after ChatGPT design review round 2 APPROVED

# Proposal: supervised gateway deferred restart

## Problem

An agent running inside the messaging gateway cannot safely execute the existing
`hermes gateway restart` command. The CLI self-refusal and terminal lifecycle
guard intentionally block a gateway-hosted process from stopping or restarting
its own host. Those protections must remain unchanged.

The current workaround opens an external desktop terminal and schedules
`sleep 30 && hermes gateway restart`. It works outside the gateway process tree,
but depends on UI automation and is not an appropriate upstream interface.

The earlier PRs #140993 and #141056 designed a separate persisted request
protocol, generation binding, cooldown, provenance, and audit system. Current
`main` now has a gateway-owned control socket, supervisor identity, an existing
`request_restart()` drain path, active-work waiting, and delivery recovery.
Re-implementing those facilities would duplicate the current architecture.

## Solution

Add one narrow CLI command:

`hermes gateway defer-restart`

The command asks the gateway serving the selected profile to accept a restart
through a new `defer-restart` control-socket verb. The actual gateway instance
must prove that it is managed by a supported restart-capable supervisor before
calling the existing `request_restart(detached=False, via_service=True)` path.

The command waits only for an acceptance response. It does not wait for drain,
a replacement process, or readiness. It has no delay, force, all-profiles, or
wait-ready parameters.

## Design

### Independent control verb

The updater's existing `pause-for-update` verb remains unchanged because the
updater may itself relaunch a manual gateway. The new command uses a distinct
`defer-restart` verb so an older gateway cannot silently execute updater
semantics and the receiving instance can enforce supervisor preconditions.

### Server-side supervisor gate

The receiving gateway accepts the request only on a supported POSIX host and
only when both conditions hold:

- its own detected supervisor is one of `systemd`, `launchd`, or `external`;
- `gateway.restart.is_gateway_supervisor_process()` confirms the process has a
  restart-capable exit-75 route.

Unsupported, missing, or ambiguous supervisor state is rejected without calling
`request_restart()`. The first version does not claim Windows, desktop, manual,
or s6 support. s6 can be added separately after its runtime identity is reported
consistently by the control socket.

The event-loop callback also rejects an ordinary stop that has already started.
This check happens when the callback executes, including after a late callback,
so a deferred request cannot change an unrelated stop into an exit-75 restart.

The positive ACK reports the accepting runner's already-loaded drain budget.
If that budget is non-finite and therefore cannot satisfy the v1 wire contract,
the handler rejects before calling `request_restart()`; it does not reread,
clamp, or modify the existing restart configuration.

### Explicit acceptance result

The event-loop handoff distinguishes three outcomes:

- `request_restart()` returns `True`: return `accepted: true`;
- it returns `False`: return `accepted: false`, reason `already_stopping`;
- it raises or no result is obtained within the bounded wait: raise from the
  handler so the existing control protocol returns `ok: false`.

A timed-out callback may still run later. Therefore a missing response is
UNKNOWN, not proof that the request did not execute. The CLI never retries or
falls back automatically.

### Profile target

The CLI prefers a confirmed live gateway owned by the selected profile. It uses
a shared host only when no own instance exists and the existing
`host_multiplexer_serving(profile)` helper explicitly confirms that the host
serves the profile and belongs to the same Hermes tenant. An unreachable or
ambiguous own instance, standalone exclusion, cross-tenant result, or failed
host resolution is rejected. The CLI never guesses the default home.

### Cron boundary

The command refuses `HERMES_CRON_SESSION` before target resolution or mutation.
This prevents misuse by marked cron agent subprocesses. Current no-agent and
some pre-agent script paths are not stamped with that marker and remain an
explicit v1 coverage gap. The check is not an OS permission boundary and does
not claim to stop a same-user process from clearing provenance or speaking to
the local control socket directly.

### Isolated end-to-end verification

The four-step E2E uses a disposable checkout-backed canary rather than the
installed production gateway. It has separate `HERMES_HOME`,
`HERMES_RUNTIME_DIR`, gateway lock directory, HOME/XDG state, transport identity,
and ports. A small external supervisor relaunches the reviewed canary child on
exit 75 with `--external-supervisor`. The canary persists its conversation store
across replacement and records acceptance from the old PID, visible final reply,
a different replacement PID, and a subsequent turn in the same canary
conversation. This proves the isolated transport used by the canary, not a live
Feishu/Telegram delivery unless a dedicated canary bot is separately used.

### Delivery boundary

The change reuses current active-work waiting and delivery recovery. It does not
add a platform-send barrier or second outbox. Eligible persisted replies may be
redelivered after restart and may carry ambiguity/duplication semantics. The
feature does not guarantee that a platform send API completed before exit,
exactly-once delivery, or recovery of content that was never persisted.

## Non-goals

- no changes to the blocking semantics for `gateway restart`, `stop`, or
  `uninstall`;
- no lifecycle-guard allowlist;
- no configurable delay or exact restart time;
- no persisted restart request/result protocol, request ID, generation binding,
  cooldown, audit trail, actor identity, or complete provenance framework;
- no changes to `request_restart()`, the delivery ledger, cron scheduler, or
  session recovery;
- no automatic fallback to `pause-for-update`, signals, or ordinary restart;
- no Windows, manual, desktop, or s6 support in this first version;
- no modification of the existing desktop Terminal workaround skill.

## Compatibility and rollback

Older gateways reject the unknown verb; the CLI reports a non-success result and
does not degrade to another restart path. Removing the new parser, CLI sibling,
control verb, and tests fully removes the feature without changing existing
restart/update behavior.

## Review record

The external ChatGPT discussion is archived outside the PR working tree:

- round 1: APPROVED WITH REQUIRED CHANGES; B1 false-success ACK and B2
  supervisor instance-replacement window;
- round 2: APPROVED; B1 and B2 CLOSED, no B3.

The review files are stored under the task workspace and are not part of the
upstream diff.
