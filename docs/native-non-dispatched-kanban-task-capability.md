# Native non-dispatched Kanban task capability

This document records the native implementation of the local contract proven in AION governance PR #598 / issue #595.

## Capability

`hermes_kanban_create_non_dispatched_task` is exposed through the structured `kanban_create` tool when callers set:

- `initial_status: todo` or `initial_status: ready`
- `no_dispatch: true`
- `correlation_key: <stable exactly-one key>`
- `body` or `metadata` carrying `audit_task_contract.v1`

The ordinary `kanban_create` behavior for `running` / `blocked` remains unchanged.

## Native proof returned

The no-dispatch path returns a `proof` object containing:

- `task_id`
- `assignee`
- `status`
- `metadata`
- `no_dispatch`
- `correlation_key`
- `read_after_create_verified`
- `worker_execution_started: false`
- `dispatcher_pickup_count: 0`

## Fail-closed rules

The native path rejects:

- `running` as a substitute for `ready`
- `blocked` as a substitute for `todo`
- missing or false `no_dispatch`
- missing `correlation_key`
- duplicate non-archived `correlation_key`
- GitHub comment / regex / timeout verdict sources
- direct DB write markers as a contract-satisfying integration path
- dispatch leakage observed during read-after-create

## Dispatch suppression

The dispatcher claim path refuses to transition tasks with `no_dispatch=1` from `ready` to `running`; this preserves `ready` as a non-dispatched audit-queue state for this capability.

## Boundary

This implementation does not itself authorize live task creation by any AION workflow, Batch 9 retry, deployment, dispatcher/controller/cron startup, production readiness claims, or Flow OS DONE.
