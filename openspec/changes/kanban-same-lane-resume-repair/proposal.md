# Guarded atomic same-lane resume

## Change classification

- `change_class`: `full-openspec`
- `intent`: Add one controller-only operation that can resume an existing Kanban card for remediation on that same card's own open PR without weakening the ordinary active-PR duplicate-lane guard and without publishing a phantom run.
- `classification_reason`: This changes durable task/run state, worker-process lifecycle, active-PR admission, concurrency behavior, and failure compensation. The blast radius includes recovery from legacy claimed/heartbeat-only runs and a live worker spawn, so bounded unit tests alone are insufficient.

## Problem

The ordinary dispatcher correctly suppresses a ready task when an active PR is already associated with its lane. That protection also strands the exceptional case in which the existing card must resume remediation on its own PR after review requested changes. Prior recovery attempts committed the card claim and opened a run before worker launch produced a trustworthy PID/session. A launch failure could therefore leave a running claim with heartbeats but no spawned-worker evidence.

The observed target is card `t_dfa23a41`, assigned to `flynn`, resuming its own PR `#155` after Crash run `1144` ended with `changes_requested`. Runs `1146` and `1149` were later controller reclaims with claimed/heartbeat-only evidence; they are housekeeping failures, not newer substantive work outcomes.

After the target worktree was restored, the target was again READY with no claim, current run, worker PID, or session, but retained `worker_started_at=85963388` from a terminal prior attempt. Because the admission guard correctly treats every non-null lifecycle handle as active, that isolated residue refused the authorized resume with `task_not_unclaimed`. The residue must be normalized by board-kernel lifecycle code, not by an operator-side database edit, and only after proving that no active claim, run, PID, or session exists.

## Proposed change

Introduce a deliberately invoked, controller-only same-lane resume operation whose initial authorization is bound specifically to task `t_dfa23a41` and PR `#155`. Any different task or PR is unauthorized by this change and must refuse; widening the allowlist requires a separate reviewed change. The invocation is bound to an immutable request containing the exact task, assignee, PR URL, base, head ref, head OID, preserved workspace, upstream, and required substantive review run/outcome. The operation:

1. accepts a one-use authorization/attempt ID for this exact target and acquires the task-scoped interprocess resume fence plus the board's existing serial write/claim lease;
2. validates every board, lane, PR-ownership, remote PR, worktree, upstream, and review-history guard while that lease is held, then atomically places the same unclaimed card in a durable `resume_quarantined` blocked state with no run;
3. launches a child behind a start gate, obtains a live PID, process-start fingerprint, and child-confirmed session ID, and revalidates every mandatory guard before publishing a claim, run handle, `spawned` event, or heartbeat;
4. atomically publishes the task claim, run, PID/start fingerprint, session, and `spawned` event;
5. releases the child only after publication commits; and
6. holds a task-scoped interprocess resume fence across rollback, full process-group termination, and compensation, and retains a durable one-use attempt receipt so concurrent contenders cannot launch sequential children for the same authorization; and
7. compensates a launch or publication failure by keeping the card quarantined and the child gated, terminating and confirming the full process tree dead, then atomically transitioning the same card to READY/unclaimed with `spawn_failed` evidence and no run handle or heartbeat.

Before admission, the exact authorized operation may atomically clear only an isolated stale `worker_started_at` fingerprint on a READY target. The kernel normalizer must refuse if any claim field, current/open run, worker PID, or session remains, and must append durable `lifecycle_normalized` evidence in the same transaction that clears the residue. It is not a generic active-handle bypass and does not rewrite historical run evidence.

The operation is not an automatic scan path and is not a general exception to active-PR suppression.

## Scope

In scope:

- controller/kernel code required for the explicit operation and gated launch handshake;
- typed immutable request and typed verification result;
- existing SQLite transaction/claim primitives and process-start fingerprinting;
- focused lifecycle, concurrency, compatibility, and compensation tests;
- read-only observer checks and a separately authorized exact-card live verification after all competing Flynn cards are terminal.

## Non-goals and prohibitions

This change SHALL NOT:

- alter ordinary `check_respawn_guard` / `_dispatch_lane_task` active-PR suppression for any task;
- permit a card to resume another task's PR;
- create or substitute a card, worktree, branch, or PR;
- edit Finance application source, Finance OpenSpec packages, or Finance tests as part of this controller change;
- mutate PR `#155`, merge it, close it, comment on it, change reviewers, push commits, or otherwise write GitHub state;
- interact with PR `#37` in any way;
- use direct/ad-hoc SQLite edits;
- deploy, change configuration, alter Hermes profiles/gateway/cron/systemd/host/network/Vault state, or change board schema;
- start Finance application services, schedules, connectors, provider/customer operations, or any external-effecting Finance runtime;
- mutate Finance/GL/proposal/review/outcome business data; or
- relax the existing serial Node/Flynn-lane posture.

The initial live capability is Linux-only because the pre-identity crash guarantee requires a hardened launcher wrapper using the host's parent-death primitive. Unsupported hosts must refuse rather than emulate weaker safety.

## Success criteria

The package is complete when implementation can prove all of the following:

- only the exact existing card may bypass its own active-PR suppression;
- every mandatory guard is checked under one claim lease;
- competing controller attempts produce at most one child and one published run;
- a real PID, process-start fingerprint, and child-confirmed session exist before run/spawn/heartbeat publication;
- success publishes one coherent task/run/`spawned` state atomically;
- failure cannot leave a claimed-only or heartbeat-only run;
- successful compensation returns the same card to READY/unclaimed and retains `spawn_failed` evidence;
- isolated terminal `worker_started_at` residue is cleared only by an atomic kernel lifecycle normalization event, while any active claim/run/PID/session remains fail-closed;
- uncertain child cleanup fails closed rather than making the card runnable beside a possible orphan; and
- all unrelated tasks still receive the ordinary active-PR suppression and serial-lane behavior.
