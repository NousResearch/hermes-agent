# Swarm Orchestrator

The `SwarmOrchestrator` is the top-level coordinator that ties together the scheduler, worker pool, model router, artifact store, and message bus. It owns no domain logic itself — it delegates to each subsystem.

## Architecture

```
SwarmOrchestrator
  |-- SwarmScheduler    (DAG task graph, dependency resolution, retries)
  |-- WorkerPool        (worker lifecycle, backend dispatch)
  |-- ModelRouter       (model selection per role, cost tracking)
  |-- ArtifactStore     (file storage with provenance)
  |-- MessageBus        (inter-agent pub/sub messaging)
```

## Usage

```python
from swarm import SwarmConfig
from swarm.orchestrator import SwarmOrchestrator

orch = SwarmOrchestrator(SwarmConfig(max_workers=4))

# Add individual tasks
t1 = orch.add_task("research", "Find info on X", role="researcher")
t2 = orch.add_task("write", "Write doc about X", role="executor", deps=[t1])

# Or add a plan (deps reference task names)
orch.add_plan([
    {"name": "plan",  "prompt": "Break down the problem", "role": "planner"},
    {"name": "code",  "prompt": "Implement solution",     "role": "executor", "deps": ["plan"]},
    {"name": "review","prompt": "Review the code",        "role": "critic",   "deps": ["code"]},
])

# Run synchronously
summary = orch.run()

# Or run in background
future = orch.run_async()
result = future.result(timeout=300)

# Cancel everything
orch.cancel()

# Inspect state
status = orch.get_status()
```

## Run Loop

1. Ensure workers are provisioned (auto-scales to `max_workers` if pool is empty).
2. Poll scheduler for ready tasks (all deps completed).
3. For each ready task, find an idle worker and select a model via the router.
4. Dispatch task to worker in a background thread.
5. On completion: store artifacts, update scheduler, broadcast on message bus.
6. On failure: scheduler handles retry (up to `max_retries`) or cancels downstream.
7. Loop until `scheduler.is_complete()` or `max_turns` reached.
8. Return summary dict with task counts, spend, artifact count.

## Key Methods

| Method | Description |
|--------|-------------|
| `add_task(name, prompt, role, deps, model)` | Create and schedule a single task |
| `add_plan(tasks)` | Add multiple tasks with name-based dependency resolution |
| `run(max_turns, poll_interval)` | Synchronous execution loop |
| `run_async()` | Returns `Future` for background execution |
| `cancel()` | Cancel pending tasks and shut down workers |
| `get_status()` | Combined status from all subsystems |
