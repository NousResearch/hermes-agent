# Unsloth Studio Plugin

## Decision

Unsloth Studio is a local browser-based UI for training, running, exporting, and comparing open models. Hermes should launch and observe it as a local service, not copy its training UI into Hermes core.

## Implemented Surface

- `hermes unsloth-studio status`
- `hermes unsloth-studio start`
- `hermes unsloth-studio stop`
- `hermes unsloth-studio install-info`
- `/unsloth-studio status`
- `/unsloth-studio install-info`

The default start command binds to `127.0.0.1:8888`. Binding to `0.0.0.0` or another non-loopback host requires `confirm_public_host`.

## Source Findings

- Unsloth Studio is Beta and works on Windows, Linux, WSL, and macOS.
- The Studio quickstart launches with `unsloth studio -H 0.0.0.0 -p 8888`, then opens `http://localhost:8888`.
- Windows install is `irm https://unsloth.ai/install.ps1 | iex`.
- macOS, Linux, and WSL install is `curl -fsSL https://unsloth.ai/install.sh | sh`.
- Studio can train via QLoRA, LoRA, and full fine-tuning, and export to safetensors or GGUF for llama.cpp, vLLM, Ollama, LM Studio, and similar runtimes.
- The Unsloth repo describes Studio UI components as AGPL-3.0 while core Unsloth remains Apache-2.0. Treat Studio as a separate local service boundary.

## Boundary

The plugin does not run remote install scripts automatically. It returns install commands for the user and only starts a locally installed `unsloth` CLI.
