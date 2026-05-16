# BO-012 — Dispatcher-native Kanban worker pilot

BO-012 verifies the upstream-intended Kanban worker path without restarting or reloading the live gateway.

## Scope

The pilot uses the existing `tests/stress/test_subprocess_e2e.py` local subprocess harness against an isolated temporary `HERMES_HOME`. It does **not** mutate the live Kanban DB, gateway process, secrets, BWS, or production/customer systems.

## Path verified

- A Kanban board is created in an isolated temp home.
- Tasks are created and assigned in Kanban.
- `dispatch_once(..., board="bo")` claims ready tasks and spawns real subprocess workers.
- The worker process receives dispatcher environment:
  - `HERMES_KANBAN_TASK`
  - `HERMES_KANBAN_WORKSPACE`
  - `HERMES_KANBAN_BOARD`
- The worker orients with `hermes kanban show <task> --json` before lifecycle mutation.
- The worker emits `hermes kanban heartbeat` progress events.
- The worker finishes with `hermes kanban complete --summary ... --metadata ...`.
- `task_runs` captures structured handoff metadata, including board pin, workspace, worker PID, iteration count, and orientation proof.
- Crash behavior remains observable: a spawned sleeper is killed, `detect_crashed_workers()` records a crashed run and requeues the task.

## Authority boundary

Kanban DB remains the execution ledger. This document and the stress harness are verification artifacts only; they do not become a competing authority surface.

## Live-system boundary

This pilot intentionally avoids live gateway restart/reload. Gateway subscription/notification routing is therefore not claimed as live-proven here. That surface requires a separately approved live-gateway exercise.
