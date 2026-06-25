# Hermes Agent IoT / Raspberry Pi 2 profile

This repository is a Pi2/IoT-oriented flavor of the original Hermes Agent source at:

`/media/matt/E/hermes-agent`

Goal: preserve native Hermes Agent architecture and feature compatibility while making the default Raspberry Pi 2 install small enough to be practical on ARMv7 / 1GB RAM.

This is not a separate mini-agent. The Python package path, CLI entrypoint, tools, plugins, gateway, memory, cron, MCP, ACP, and provider architecture stay intact. The Pi2 profile only changes default installation choices and default enabled tool surface.

## What is preserved

Kept as native Hermes functionality:

- `hermes` CLI entrypoint from `pyproject.toml`
- core agent loop and provider routing
- tools system and toolsets
- skills system
- persistent memory and session search
- cron scheduler
- delegation/subagents
- MCP and ACP code paths
- gateway and platform adapters
- plugins and memory provider plugins
- OpenAI-compatible endpoints, including local `llama.cpp` servers

## What is slimmed

The Pi2 profile avoids eager install/use of heavy features:

- browser automation runtime
- local Chromium/Playwright-style stacks
- image/video generation backends
- voice/STT dependencies such as faster-whisper
- TTS premium providers
- torch / sentence-transformers / chromadb by default
- dashboard/web UI extras unless explicitly requested
- messaging platform extras unless explicitly requested

The code remains present. Heavy features can be re-enabled later with `hermes tools` and the relevant Python extras.

## Install profiles

Use the native-compatible installer:

```bash
bash setup-pi2-minimal.sh --profile core
bash setup-pi2-minimal.sh --profile native
bash setup-pi2-minimal.sh --profile rag
```

Profiles:

- `core`: smallest practical Hermes CLI profile. Installs package through `pip install -e .[cli,pty]`, writes a config that disables heavy toolsets by default.
- `native`: core plus MCP/ACP/Home Assistant/SMS extras. Still disables browser/media/messaging tool surfaces by default.
- `rag`: native plus lightweight document helpers and Honcho optional dependency. Remote embeddings are recommended; local torch stacks are not installed by default.

## Recommended Pi2 install

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git

git clone --depth 1 https://github.com/matttest0080-prog/hermes-agent-iot.git
cd hermes-agent-iot
bash setup-pi2-minimal.sh --profile core

source ~/.hermes-venv/bin/activate
hermes setup model
hermes
```

Hermes requires Python `>=3.11,<3.14`. If your Raspberry Pi OS image ships older Python, install Python 3.11+ first.

## Local llama.cpp / OpenAI-compatible model

Pi2 can connect to a local or LAN OpenAI-compatible endpoint. For example, if `llama.cpp` server is running at `http://localhost:8080/v1`, configure Hermes through:

```bash
hermes setup model
```

or edit `~/.hermes/config.yaml` using the normal Hermes config commands.

Important: a 7B model on Raspberry Pi 2 is usually impractical because of RAM and speed. Prefer:

- a remote/OpenRouter/OpenAI-compatible provider, or
- a much smaller quantized model, or
- a stronger LAN machine running `llama.cpp` with Pi2 acting as the Hermes client.

## Memory and RAG posture

Default Pi2 memory:

- built-in Hermes memory
- session search
- SQLite/FTS-style lightweight local state

Optional RAG:

```bash
bash setup-pi2-minimal.sh --profile rag
```

The RAG profile intentionally avoids installing `torch`, `sentence-transformers`, and `chromadb` by default. For Pi2, prefer remote embeddings or cloud memory providers. If you explicitly want local embeddings, install them manually and expect high RAM/compile cost.

## Config templates

Installer templates live in:

```text
templates/config.pi2-core.yaml
templates/config.pi2-native.yaml
templates/config.pi2-rag.yaml
```

The installer copies one to `~/.hermes/config.yaml` only if that file does not already exist. Existing configs are not overwritten.

## Re-enabling features later

Use native Hermes controls:

```bash
hermes tools
hermes tools list
hermes tools enable browser
hermes tools enable image_gen
hermes config edit
```

Then install any missing extra dependencies shown by the feature/tool.

## Verification

After installation:

```bash
source ~/.hermes-venv/bin/activate
hermes --help
python -m py_compile cli.py run_agent.py model_tools.py toolsets.py
python - <<'PY'
from toolsets import resolve_toolset
print('hermes-cli tools:', len(resolve_toolset('hermes-cli')))
print('file tools:', resolve_toolset('file'))
PY
```

## Design rule

Allowed Pi2 slimming:

- default-off toolsets
- optional extras
- lazy imports/lazy install
- profile-specific config
- documentation of heavy paths

Avoided:

- deleting core modules
- patching source during install
- removing toolset definitions
- forking config schema
- hard-coding Pi2-only behavior into generic runtime paths
