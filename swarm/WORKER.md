# Swarm Worker Pool

## Overview

`WorkerPool` manages a set of `SwarmWorker` instances across heterogeneous backends. All operations are thread-safe (guarded by a `threading.Lock`).

## Backends (strategy pattern)

| Backend | Class | Status |
|---------|-------|--------|
| `local` | `LocalWorkerBackend` | Stub (sleeps, returns placeholder) |
| `modal` | `ModalWorkerBackend` | Stub (raises `NotImplementedError`) |
| `ssh`   | `SSHWorkerBackend`   | Stub (raises `NotImplementedError`) |

Each backend implements `WorkerBackend`: `start_worker`, `run_task`, `stop_worker`.

## Key operations

- **add_worker / remove_worker** -- register or drain+remove a worker.
- **get_available_worker(capabilities)** -- first idle worker matching all requested capabilities.
- **assign_task(worker_id, task)** -- marks worker busy, runs the task in a daemon thread via the backend.
- **release_worker** -- force-mark a worker idle.
- **scale_up(n, backend) / scale_down(n)** -- elastic scaling within `SwarmConfig.max_workers`.
- **shutdown** -- joins all task threads, stops every worker.

## Task lifecycle through the pool

1. Caller picks a worker via `get_available_worker`.
2. Caller calls `assign_task` -- worker state becomes `busy`, task state becomes `running`.
3. Backend's `run_task` executes in a background thread.
4. On completion the worker returns to `idle` automatically.

## File

`swarm/worker.py`
