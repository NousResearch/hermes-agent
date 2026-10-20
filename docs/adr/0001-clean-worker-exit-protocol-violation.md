# ADR 0001: Clean worker exit is a sticky protocol violation

- Status: accepted
- Date: 2026-09-24

## Context

A Kanban worker can exit with code `0` while its card remains `running`. This
means no terminal Kanban action (`kanban_complete`, `kanban_request_review`, or
`kanban_block`) reached the board. Treating that exit as retryable can re-run a
worker which already failed to follow its required protocol.

The reaper can race with a terminal board action. Recovery therefore must not
close a run, append an event, or change task state after another actor has
already terminally transitioned the task.

## Decision

On first observed clean exit for a still-running claimed task, the worker
reaper performs one compare-and-swap update guarded by task ID, `running`
status, worker PID, and claim lock. The winner changes the card to `blocked`,
clears worker ownership, stores failure evidence, closes the active run with
outcome `protocol_violation`, and appends a visible `protocol_violation` event
with `sticky: true` in the same transaction.

A failed CAS has no follow-up effects. The winning protocol event participates
in sticky-block detection, so dispatcher recomputation never requeues or
auto-promotes the card. Only explicit `kanban unblock` releases it.

## Consequences

Operators receive a durable run outcome, worker evidence, and board event to
inspect before deciding whether to unblock. Existing nonzero crash, timeout,
and rate-limit paths remain unchanged. Worker nudge and exit-trailer behavior
remain inputs to exit classification.
