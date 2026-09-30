# Hermes Tensor Android V2

Native Android companion for the Hermes Termux Tensor profile.

Responsibilities:
- Host LiteRT-LM through the Android API.
- Own CPU/GPU/NPU delegate selection.
- Report the actual selected backend.
- Provide foreground-service lifecycle and stop controls.
- Later add thermal/battery policy and model management.

The current skeleton intentionally uses a CPU-only probe until the exact LiteRT-LM Android dependency/API is wired in. It must not fake NPU/GPU availability.

Next implementation: add LiteRT-LM Android dependency, implement Engine/model loading, implement real delegate probing, add localhost HTTP or Binder bridge, add foreground service notification, add Compose dashboard, then connect the Termux tensor-bridge client.