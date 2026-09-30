# Hermes Agent — Google Tensor / Termux Edition

This directory is the Google Tensor / Pixel profile for Hermes Agent on ARM64 Android.

## V1 local LiteRT-LM integration

Hermes already supports OpenAI-compatible custom endpoints, so this profile uses the existing `custom` provider instead of changing the Hermes provider core.

### Diagnose
    bash termux-tensor/bin/tensor-doctor

### Import a local model
    bash termux-tensor/bin/tensor-lm import ~/storage/downloads/model.litertlm tensor-local

### Start local inference
    bash termux-tensor/bin/tensor-lm start tensor-local

The controller starts LiteRT-LM on `127.0.0.1:9379`, waits for `/v1/models`, and configures Hermes automatically.

### Test / status
    bash termux-tensor/bin/tensor-lm status
    bash termux-tensor/bin/tensor-lm test

### Start Hermes
    hermes

### Stop
    bash termux-tensor/bin/tensor-lm stop

## Configuration

The controller writes a custom Hermes endpoint equivalent to:

    model:
      provider: custom
      base_url: http://127.0.0.1:9379/v1
      api_key: none
      default: tensor-local

Hermes documents `provider: custom` and a local OpenAI-compatible `base_url` for self-hosted inference. LiteRT-LM exposes an OpenAI-compatible server with `/v1/models` and `/v1/chat/completions`.

## Google Tensor boundary

V1 does not claim direct Tensor TPU/NPU access from a normal Termux process. CPU inference is the baseline. GPU/NPU acceleration depends on the LiteRT-LM runtime, model, Android integration, and available delegates.

For true Android accelerator integration, V2 should use a native Android LiteRT-LM service and keep Hermes in Termux as the agent/control layer.

## Phone resource policy
- One active local model by default.
- No parallel Hermes workers by default.
- Explicit start/stop controls.
- PID and server logs under `~/.hermes/tensor/`.
- Use shorter context/model sizes on lower-RAM devices.
- Keep long-running gateway/background execution opt-in.

## Existing architecture
Android / Pixel / Google Tensor → Termux (aarch64) → Hermes Agent → local inference backend
- LiteRT-LM: preferred when a compatible native binary/model is available
- llama.cpp: CPU fallback

## V2 architecture
    Hermes / Termux
          |
          | localhost IPC
          v
    Android LiteRT-LM service
          |
       +--+--+
       |     |
      GPU   NPU

This keeps the accelerator path explicit: Termux controls Hermes, while the Android runtime owns hardware-specific delegates.