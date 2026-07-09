---
name: hermes-native
description: |
  Hermes Native: sovereign VS Code inference/runtime skill + blueprint.
  Use when the user invokes `remoteUse.hermesNative`, references Hermes Native,
  WebLLM/WebGPU in VS Code, or wants a local Copilot-class primitive.
  Triggers: "hermes native", "webllm", "webgpu runtime", "remoteUse.hermesNative",
  "sovereign inference", "local runtime in vs code".
  Prevents: duplicate runtime projects, fallback to non-native approach,
  separate runtime instances when the existing one should be extended.
  Skips: creative UI polish until the runtime layer is hardened.
---

# Hermes Native

Hermes Native is the sovereign local inference/runtime surface inside the existing VS Code extension. Treat it as both a skill and a blueprint for the bounded private-client mesh.

## Prefixes

- `+æ://hermes-native`
- `vscode://hermes-native`
- `pc://mesh/victus/local/runtime/hermes-native`

## Canonical Commands

- `remoteUse.hermesNative`
- `remoteUse.webLLM` → delegates to `remoteUse.hermesNative`

## Files of Truth

- Extension entrypoint: `C:\æ\hermes-fork\vscode-remote-use\src\extension.ts`
- Surface template: `C:\æ\hermes-fork\vscode-remote-use\templates\web-llm.html`
- Installer: `C:\æ\hermes-fork\vscode-remote-use\scripts\install-hermes-vscode.ps1`
- Validated package: `C:\æ\hermes-fork\vscode-remote-use\vscode-remote-use-0.1.0.vsix` with publisher `hermes-agent`

## Surface Requirements

- One runtime instance only, not parallel projects.
- Must use local template discovery first (`findLocalTemplate`); fallback HTML only if no local template exists.
- Must preserve backward compatibility for existing `remoteUse.*` commands.
- Must not move inference work to a cloud backend.
- Must expose runtime state through the webview: `LOADING`, `READY`, `ENGINE_FAILED`, `INFERENCE_FAILED`.

## Invariants

- `remoteUse.hermesNative` is the first-class command.
- All “WebLLM/WebGPU” work happens inside `remoteUse.hermesNative`.
- Keep `remoteUse.webLLM` as a delegating alias only.
- State machine must be observable in the webview log.
- If the user asks for a “second instance,” refuse and extend this one.
