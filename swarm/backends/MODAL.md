# Modal Swarm Backend

Cloud execution backend using [Modal](https://modal.com) sandboxes.

## Overview

`ModalSwarmBackend` provisions a dedicated Modal sandbox per swarm worker.
Tasks are executed by sending serialized prompts into the sandbox via
`sandbox.exec()`, collecting stdout/stderr, and returning structured
`SwarmResult` objects.

## Quick start

```python
from swarm.backends import ModalSwarmBackend
from swarm.types import SwarmWorker, SwarmTask

backend = ModalSwarmBackend(
    app_name="hermes-swarm",
    image="nikolaik/python-nodejs:python3.11-nodejs20",
)

worker = SwarmWorker(name="cloud-1", backend="modal")
backend.start_worker(worker)

task = SwarmTask(name="install deps", prompt="pip install requests && echo OK")
result = backend.run_task(worker, task)
print(result.success, result.output)

backend.stop_worker(worker)
```

## Constructor parameters

| Parameter  | Default | Description |
|------------|---------|-------------|
| `app_name` | `"hermes-swarm"` | Modal app namespace |
| `image`    | `"nikolaik/python-nodejs:python3.11-nodejs20"` | Docker image for sandboxes |
| `gpu`      | `None` | GPU type (e.g. `"T4"`, `"A10G"`) |
| `cpu`      | `1.0` | CPU cores per sandbox |
| `memory`   | `2048` | Memory in MB per sandbox |
| `timeout`  | `600` | Max sandbox lifetime in seconds |

## Methods

- **`start_worker(worker)`** — Provision a Modal sandbox. Returns `True` on success.
- **`run_task(worker, task)`** — Serialize the task prompt, execute in the sandbox, return `SwarmResult`.
- **`stop_worker(worker)`** — Terminate the sandbox.
- **`health_check(worker)`** — Exec `echo ping` in the sandbox; returns `True` if responsive.
- **`shutdown_all()`** — Terminate all active sandboxes.

## Integration with WorkerPool

Register the backend in `swarm/worker.py`:

```python
from swarm.backends import ModalSwarmBackend

_BACKENDS["modal"] = ModalSwarmBackend
```

Or instantiate with custom config and inject directly:

```python
pool._backends["modal"] = ModalSwarmBackend(gpu="T4", memory=4096)
```

## Requirements

- `modal` Python package (`pip install modal`)
- Modal account and token (`modal token new`)
- Import is lazy — the module loads without modal installed but raises
  `ImportError` on first use.
