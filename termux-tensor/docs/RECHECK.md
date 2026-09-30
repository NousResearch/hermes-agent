# A-to-Z Recheck

## Architecture

- [x] Termux-first V1
- [x] Hermes remains agent layer
- [x] Local model directory
- [x] Unified tensor CLI
- [x] Diagnostics
- [x] Model import/select/current/remove
- [x] Thermal status
- [x] Persistent local-server fallback
- [x] Hermes custom endpoint configuration
- [x] Loopback-only default
- [x] Installation guide
- [x] Settings guide
- [x] Troubleshooting guide
- [x] Android V2 boundary documented

## Safety and correctness corrections

- Do not assume a `litert-lm serve` command exists.
- Do not claim Tensor NPU access from normal Termux.
- Do not expose the local server to the LAN by default.
- Do not use unsigned Hermes repositories.
- Do not treat Android background execution as guaranteed.
- Do not connect Hermes until direct model inference and the local HTTP endpoint have been tested separately.

## V1 runtime order

```text
Termux
  -> Hermes
  -> Tensor profile
  -> model selection
  -> runtime test
  -> localhost server if needed
  -> Hermes custom provider
```

## V2

Native Android LiteRT-LM Engine is the correct boundary for Android GPU/NPU/Tensor-specific acceleration. The current Android project skeleton is kept separate from the Termux V1 runtime.