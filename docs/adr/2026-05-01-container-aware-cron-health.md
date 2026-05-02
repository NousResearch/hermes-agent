# ADR: Container-aware cron gateway health

Date: 2026-05-01
Status: Accepted

## Context

Hermes cron jobs are executed by the gateway scheduler. Historically `hermes cron status` and `hermes cron list` inferred automatic execution from locally visible gateway PIDs. That works for a foreground gateway or a systemd/launchd service in the same process namespace.

It is misleading for containerized deployments where one supervised gateway container owns the scheduler and another CLI/container shares the same `HERMES_HOME` volume. In that topology the gateway PID may not be visible from the inspecting process namespace, but the shared gateway runtime lock can still prove that a gateway owner exists for that `HERMES_HOME`.

A PID-only check can therefore produce a false negative: "Gateway is not running — cron jobs will NOT fire" while the Docker-supervised gateway is actually ticking jobs. Worse, stale-PID cleanup must not remove metadata when the runtime lock is actively held by another namespace.

## Decision

Hermes cron diagnostics will treat the gateway runtime lock as the scheduler-owner signal for shared `HERMES_HOME` deployments.

`gateway.status.get_gateway_owner_status()` reports three states:

- `local_pid_running`: the gateway PID is visible and verified in the current namespace.
- `shared_lock_active`: a gateway runtime lock is held, but the PID is not locally visible or not inspectable; this is expected for shared-volume container/service-scope deployments.
- `not_running`: no active gateway runtime lock was found.

`get_running_pid()` remains PID-oriented for callers that need a local PID, but it no longer deletes PID/lock metadata merely because the PID is not visible while the runtime lock is active.

`hermes cron status` and `hermes cron list` use `get_gateway_owner_status()` so operators see container-aware status instead of a PID-namespace false negative.

## Consequences

Positive:

- Cron status is accurate for Docker/shared-volume deployments.
- Operators are less likely to start a second gateway and create platform polling conflicts.
- Gateway metadata is preserved when another namespace owns the runtime lock.

Tradeoffs:

- A held lock can only prove an owner holds the scheduler lock; it does not prove every platform adapter is healthy.
- Lock semantics are only reliable on filesystems that preserve advisory locks across the relevant processes/containers.

## Operational guidance

- Run exactly one supervised gateway owner per shared `HERMES_HOME`.
- In container deployments, prefer the Docker/Compose/systemd owner instead of launching ad hoc foreground gateways for cron ticks.
- If `hermes cron status` reports a shared runtime lock, inspect gateway logs and runtime health before replacing the gateway.
- Do not rely on this guarantee over unverified network filesystems.
