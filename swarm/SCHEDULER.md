# Swarm Task Scheduler

DAG-based task scheduler for the swarm execution engine.

## Core Concepts

- **Tasks** are nodes in a directed acyclic graph. Each task has a unique ID, a prompt, and optional dependencies on other tasks.
- **Dependencies** are edges: `add_dependency(A, B)` means A cannot run until B completes.
- **States** follow the lifecycle: `pending -> running -> completed | failed | cancelled`.

## Thread Safety

All public methods acquire a `threading.Lock`. Workers call `mark_completed` / `mark_failed` from their own threads.

## Retry Logic

When a task fails, `mark_failed` increments its retry counter. If retries remain, the task returns to `pending` and becomes eligible for `get_ready_tasks` again. Once retries are exhausted, the task moves to `failed` and all downstream dependents are cancelled recursively.

## Scheduling Order

`get_ready_tasks` returns pending tasks whose dependencies are all completed, sorted by `created_at` (oldest first). The caller (orchestrator) is responsible for dispatching these to workers.

## Checkpointing

`to_dict()` / `from_dict()` serialize the full scheduler state for crash recovery.
