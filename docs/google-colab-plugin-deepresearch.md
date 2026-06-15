# Google Colab CLI Plugin Deep Research

## Decision

Use a standalone Hermes plugin around the official `google-colab-cli`, not a new core model tool. Colab is a compute backend with account state, quota, and billable accelerator allocation, so it belongs behind explicit plugin enablement and confirmation.

## Source Findings

- Google announced Colab CLI on 2026-06-05 for terminal and agent workflows. It supports remote execution, artifact recovery, interactive access, and accelerator provisioning.
- The official repository states that Colab CLI currently supports Linux and macOS only. Windows should use WSL for now.
- `colab run` is the safest default for one-shot jobs because it provisions, executes, and stops the VM unless `--keep` is set.
- Persistent sessions are useful for incremental work, but `colab new` creates a live rented VM and `colab stop` is required to avoid burning compute units.
- Agent automation must avoid interactive `colab repl`, `colab console`, `colab auth`, and `colab drivemount` unless stdin/browser handling is deliberately arranged.
- TRL `SFTTrainer` accepts text, prompt/completion, and conversational `messages` datasets. For Hermes-style traces, `messages` plus assistant-only loss is the preferred shape.
- QLoRA should use 4-bit quantization plus LoRA/PEFT rather than full fine-tuning on Colab GPUs.
- Irodori-TTS-Server is already OpenAI TTS API compatible and the Hermes repo already has an `irodori_tts` plugin. Colab should train or prepare artifacts; local Irodori serving should remain the runtime boundary.
- `llama-server` exposes an OpenAI-compatible chat completions endpoint. Hermes should connect to that endpoint as a model provider or custom OpenAI-compatible provider after a GGUF model is available.

## Implemented Surface

- `hermes google-colab status`
- `hermes google-colab sessions`
- `hermes google-colab run --gpu T4 --confirm script.py`
- `hermes google-colab sft-template --output-path script.py`

The `google_colab_run` tool requires `confirmed=true` before invoking `colab run`.

## Recommended Workflow

1. Enable the plugin: `hermes plugins enable google-colab`.
2. On Windows, install and authenticate Colab CLI inside WSL.
3. Generate an SFT script with `hermes google-colab sft-template --output-path runs/hermes_sft.py`.
4. Run the script with `hermes google-colab run --gpu T4 --confirm runs/hermes_sft.py`.
5. Upload the adapter to Hugging Face or download it locally.
6. Convert or publish GGUF separately if the target model architecture is supported by llama.cpp.
7. Serve the GGUF with `llama-server` and point Hermes at `http://127.0.0.1:8080/v1`.

## Boundaries

- Do not add Colab as a core tool.
- Do not hide billable allocation behind an automatic model tool call.
- Do not store HF tokens in generated scripts.
- Do not route Irodori synthesis through Colab for ordinary runtime use.
- Do not claim Windows native support while upstream states it is unsupported.
