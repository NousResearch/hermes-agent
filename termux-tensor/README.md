# Hermes Agent — Google Tensor / Termux Edition

This directory is the Google Tensor / Pixel profile for Hermes Agent on ARM64 Android.

## Architecture
Android / Pixel / Google Tensor → Termux (aarch64) → Hermes Agent → local inference backend

- LiteRT-LM: preferred when a compatible native binary/model is available
- llama.cpp: CPU fallback

Hermes stays the agent/orchestration layer. The Tensor profile does not modify the core agent loop.

## Acceleration boundary
A Tensor SoC does not automatically expose its TPU/NPU to a Termux process. CPU is the guaranteed baseline. GPU/NPU acceleration depends on the runtime, model, Android integration, and available delegates.

This profile therefore treats CPU as guaranteed, GPU as optional, and NPU as an optional Android-runtime integration.

## Target
- ARM64 / aarch64 Android
- Google Tensor / Tensor G2/G3/G4-class Pixel hardware
- Android 12+
- 6 GB RAM minimum for lightweight models
- 8–12 GB preferred for larger local models

## Install
Use the official Hermes Termux APT package, then run:

\`\`\`bash
pkg install git curl jq
git clone https://github.com/abdulraheemnohri/hermes-agent.git
cd hermes-agent
bash termux-tensor/bin/tensor-doctor
\`\`\`

## Local model layout
\`\`\`text
${HOME}/.hermes/models/
├── litertlm/
│   └── model.litertlm
└── gguf/
    └── model.gguf
\`\`\`

Set:
\`\`\`bash
export HERMES_LOCAL_MODEL_DIR="$HOME/.hermes/models"
\`\`\`

## Local endpoint
If a local runtime exposes an OpenAI-compatible endpoint:

\`\`\`bash
export HERMES_LOCAL_BASE_URL=http://127.0.0.1:9379/v1
export HERMES_LOCAL_MODEL=local
hermes model
\`\`\`

## LiteRT-LM
Prefer a native Android/Bionic build or compatible prebuilt binary. Do not assume \`pip install litert-lm\` works on Termux.

Example:
\`\`\`bash
litert-lm serve --host 127.0.0.1
\`\`\`

Then configure Hermes to use the local endpoint.

## Thermal policy
- one active model by default
- no parallel inference workers by default
- short context on 6 GB devices
- monitor temperature for long jobs
- explicit user-controlled background execution
- immediate stop remains available

## Scope
This profile does not modify the Hermes core loop, require root, require Docker, silently download models, or claim direct TPU access from Termux.
