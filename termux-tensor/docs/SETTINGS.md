# Termux Tensor Settings

## Core defaults

```text
TENSOR_HOST=127.0.0.1
TENSOR_PORT=9379
TENSOR_THERMAL_MODE=balanced
TENSOR_LOG_LEVEL=INFO
ACTIVE_MODELS=1
LAN_EXPOSURE=false
BACKGROUND_INFERENCE=false
```

## Model policy

- Keep one model loaded at a time.
- Prefer the smallest model that satisfies the task.
- Keep context conservative on 6 GB devices.
- Do not run parallel generation by default.
- Stop inference before long gaming, camera, charging, or other high-load sessions.

## Runtime policy

LiteRT-LM direct mode uses the native Android ARM64 CLI and CPU backend in Termux V1.

llama.cpp server mode is the persistent HTTP path for Hermes custom-provider integration.

GPU and NPU are not enabled by guessing. The runtime must actually expose the backend.

## Network

Bind local inference to 127.0.0.1. LAN access is disabled by design.

## Logs

```text
~/.hermes/tensor/logs/
```

## State

```text
~/.hermes/tensor/state/current-model
```

## Environment

Set overrides before starting the runtime:

```bash
export TENSOR_LITERT_BIN=/path/to/litert_lm_main
export TENSOR_SERVER_BIN=/path/to/llama-server
export TENSOR_HOST=127.0.0.1
export TENSOR_PORT=9379
```