# Proposal: Repair cron process-isolation ownership

## Why
The isolated cron supervisor must remain truthful during shutdown and external-fire retries. A registry entry removed before reaping can hide a live child, an unbounded result poll can overshoot the configured wall limit, and an expired advisory fire claim can overwrite a still-live execution.

## What
Keep child registration until forced cleanup and confirmed reaping; bound supervisor polling by the remaining deadline; attach external fire claims to durable execution IDs and refuse reclamation while that execution owner is live; add regression coverage for these lifecycle invariants. Replace descendant enumeration with a fail-closed containment boundary: the preserved implementation allocates a dedicated verifiably owned Linux cgroup-v2 boundary, parks the child, assigns and verifies membership, then releases it to start the workload; a systemd transient unit/scope is a separately implemented alternative. Terminate only that opaque owned boundary and wait for emptiness plus child reaping. If any capability, ownership, or teardown step fails, report the exact unavailable/cleanup-failed status and retain ownership rather than claiming success. Process-group cleanup is only an explicitly-labelled best effort fallback. macOS and Windows explicitly do not claim equivalent detached-session containment.

## Narrowed platform contract
This change does not promise universal descendant cleanup. A hard detached-session-containment claim is valid only when Linux establishes a dedicated cgroup-v2 boundary, or when a separately implemented systemd transient unit/scope adapter proves that its control group is owned and terminable, before the workload may execute. The preserved diff implements the cgroup-v2 option only. The supervisor must prove each prerequisite at runtime: delegated child creation/removal, empty boundary, stable parent/identity, PID membership readback while parked, and cgroup-targeted teardown capability. A capability failure, permission error, assignment/readback failure, or identity mismatch before release fails closed for that claim; it never silently upgrades process-group signaling into equivalent containment. macOS and Windows are explicitly unsupported for this contract.

The fallback remains safe and useful but intentionally weaker: it records `unavailable` or `unsupported`, may terminate the spawned process group as best effort, and never reports that a detached `setsid` descendant was authoritatively contained. A revalidation, cgroup termination, emptiness, or reaping failure records `cleanup_failed` and retains the boundary ownership record for diagnosis instead of releasing it as successful cleanup.

## Architectural anchor
This change serves ADR 0006 (`/home/linh/hpladrs/0006-spec-driven-development-via-openspec.md`) and does not alter the accepted ADR layer.
