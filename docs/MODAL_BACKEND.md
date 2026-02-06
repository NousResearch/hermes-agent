# Modal Backend

Hermes Agent uses [Modal](https://modal.com) for scalable, isolated cloud execution environments. There are two Modal integrations:

1. **Terminal Tool** (`tools/terminal_tool.py`) - For CLI/agent command execution
2. **Atropos Backend** (`atropos/backends/modal_backend.py`) - For batch RL training workloads



---

## Terminal Tool (CLI/Agent)

The terminal tool provides a simple interface for executing commands in Modal sandboxes.

### Configuration

Set environment variables:

```bash
export TERMINAL_ENV=modal
export TERMINAL_MODAL_IMAGE=python:3.11
export TERMINAL_MODAL_APP_NAME=hermes-sandbox
```

Or use a YAML config file (`modal_profiles.yaml`):

```yaml
profiles:
  default:
    image: python:3.11
    cpu: 1.0
    memory: 2048
    min_pool: 1
    max_pool: 5
    idle_timeout: 120

  gpu:
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    gpu: T4
    memory: 16384
    min_pool: 0
    max_pool: 2
```

### Features

| Feature | Description |
|---------|-------------|
| **Sandbox Pool** | Pre-warmed sandboxes for low latency |
| **Auto-scaling** | Grows/shrinks pool based on demand |
| **Idle Timeout** | Sandboxes auto-terminate when unused |
| **Profile Selection** | Different configs for different workloads |
| **Credential Injection** | `modal.Secret` integration |

### Usage

```python
from tools.terminal_tool import terminal_tool

# Simple command
output = terminal_tool("echo hello", task_id="my-task")

# With profile selection
output = terminal_tool("python train.py", task_id="training", profile="gpu")

# Cleanup when done
from tools.terminal_tool import cleanup_vm
cleanup_vm("my-task")
```

### Architecture

```
_ModalPoolManager (singleton)
    ├── "default" pool → [sandbox-0, sandbox-1, ...]
    └── "gpu" pool     → [sandbox-0, ...]

Each pool:
  - Maintains min_pool warm sandboxes
  - Scales up to max_pool on demand  
  - Background thread scales down idle sandboxes
```

---

## Atropos Backend (RL Training)

The Atropos backend is designed for high-throughput batch execution during reinforcement learning training.

### Key Concept: Slot-based Multiplexing

Instead of one sandbox per trajectory, multiple trajectories share sandboxes via **slots**:

```
Sandbox (1 container)
    ├── Slot 0 → Trajectory A (workspace: /data/slot_0)
    ├── Slot 1 → Trajectory B (workspace: /data/slot_1)
    └── Slot 2 → Trajectory C (workspace: /data/slot_2)
```

**Benefits**:
- Fewer containers = lower cost
- Shared warm-up time
- Better GPU utilization

### Configuration

```python
from atropos.backends.modal_backend import ModalSandboxConfig, ModalToolBackend

config = ModalSandboxConfig(
    name="default",
    image="python:3.11",
    cpu=1.0,
    memory=2048,
    slots_per_sandbox=10,  # 10 trajectories per container
    min_sandboxes=1,
    max_sandboxes=5,
)

backend = ModalToolBackend(config.with_app_name("my-training"))
```

### Multi-Profile Support

Different trajectory types can request different resources:

```python
backend = ModalToolBackend.with_profiles(
    app_name="rl-training",
    profiles={
        "default": ModalSandboxConfig(
            name="default",
            cpu=1.0,
            memory=2048,
        ),
        "pytorch-gpu": ModalSandboxConfig(
            name="pytorch-gpu",
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            gpu="T4",
            memory=16384,
        ),
    }
)

# CPU task
slot1 = await backend.acquire("traj-1", profile="default")

# GPU task
slot2 = await backend.acquire("traj-2", profile="pytorch-gpu")
```

### Batched Execution

The key optimization - execute many commands in parallel:

```python
# Acquire slots for multiple trajectories
slots = [await backend.acquire(f"traj-{i}") for i in range(50)]

# Execute batch across all slots in parallel
results = await backend.execute_batch([
    (slot, "bash", {"command": "python step.py"})
    for slot in slots
])

# Release slots
for slot in slots:
    await backend.release(slot)
```

### Architecture

```
ModalToolBackend
    └── _ModalMultiProfileManager
            ├── "default" → _ModalSandboxPool
            │                   ├── Sandbox 0 (slots 0-9)
            │                   └── Sandbox 1 (slots 0-9)
            │
            └── "pytorch-gpu" → _ModalSandboxPool
                                    └── Sandbox 0 (slots 0-9)
```

---

## Credentials

Inject secrets securely using Modal's secret management:

```bash
# Create secret in Modal dashboard or CLI
modal secret create my-api-key API_KEY=sk-xxx
```

```python
# Reference in config
config = ModalSandboxConfig(
    secrets=["my-api-key"],  # Modal secret names
    env_vars={"DEBUG": "1"},  # Additional env vars
)
```

## Troubleshooting

### "Modal package not installed"
```bash
pip install modal
modal token new  # Authenticate
```

### "Sandbox creation failed"
- Check Modal dashboard for quota limits
- Verify image exists and is accessible
- Check secret names are correct

### Shutdown errors
These are harmless warnings during Python interpreter shutdown:
```
[Modal] Error terminating ...: cannot schedule new futures after interpreter shutdown
```

The sandboxes will auto-terminate via Modal's idle_timeout anyway.
