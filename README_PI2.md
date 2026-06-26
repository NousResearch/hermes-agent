# Hermes Agent IoT / Raspberry Pi 2+ profile

This repository is a Pi2/IoT-oriented flavor of the original Hermes Agent source at:

`/media/matt/E/hermes-agent`

Goal: preserve native Hermes Agent architecture and feature compatibility while making the default Raspberry Pi 2 install small enough to be practical on ARMv7 / 1GB RAM.

This profile is primarily designed for embedded-system development: always-on controllers, sensor/automation nodes, lab devices, robotics gateways, home/industrial IoT boxes, and other constrained Linux deployments where low memory use, predictable dependencies, and remote-first AI services matter more than desktop-heavy local stacks.

Raspberry Pi 2 is the minimum/baseline target for this profile, not the maximum supported device. The same lightweight install path is suitable for Pi2-class or better Linux systems such as Raspberry Pi 3/4/5, Pi Zero 2 W, ARM64 SBCs, x86 mini PCs, and VMs. On stronger hardware, you can keep the safe Pi2 defaults or opt into heavier Hermes extras after install.

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
- `native`: core plus MCP/ACP/Home Assistant/MQTT/SMS extras. Still disables browser/media/messaging tool surfaces by default.
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

## Robotics applications

This fork can be used as a lightweight robot edge agent: it coordinates high-level tasks, talks to MQTT/HTTP/serial/ROS bridge layers, summarizes robot state, and connects a remote/LAN AI model to the robot controller. Keep hard real-time motor control, obstacle reflexes, and emergency-stop enforcement in an MCU or ROS controller.

Recommended split:

```text
Cloud/LAN LLM or operator UI -> Hermes Agent IoT -> MQTT/HTTP/serial bridge -> MCU/ROS controller -> robot body
```

See `ROBOTICS.md` for recommended robot architectures, MQTT topic conventions, safety limits, watchdog tasks, and next implementation steps.

## MQTT IoT tools

The native and rag profiles include lightweight MQTT support for embedded sensors and actuators through the `mqtt` toolset. Configure a broker with:

```bash
export MQTT_HOST=192.168.1.10
export MQTT_PORT=1883
# Optional:
export MQTT_USERNAME=iot-user
export MQTT_PASSWORD=secret
export MQTT_TLS=false
hermes tools enable mqtt
```

Available MQTT tools:

- `mqtt_publish`: publish sensor values or device commands
- `mqtt_subscribe_recent`: listen briefly for retained/new messages on a topic filter
- `mqtt_device_command`: publish a command and optionally wait for a state/ack topic

MQTT brokers do not provide history by default. `mqtt_subscribe_recent` returns retained messages and messages published while the tool is listening.

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

## Context and performance defaults

Upstream Hermes keeps a 64K runtime-context floor for reliable full tool use with local/Ollama models. The Pi2 profiles lower that floor through `agent.minimum_tool_context_length` so small local models can still run in a degraded, low-tool mode:

- `config.pi2-core.yaml`: `2048` tokens for tiny local chat / minimal tools
- `config.pi2-native.yaml`: `8192` tokens for broader native workflows
- `config.pi2-rag.yaml`: `8192` tokens locally, with remote/central RAG preferred

For full Hermes tool use, coding, or shared RAG, keep using a stronger LAN/cloud model with 64K+ context. The Pi2 override is an escape hatch for constrained local inference, not a claim that 2K can carry the complete Hermes tool surface.

## Memory and RAG posture

Default Pi2 memory:

- built-in Hermes memory
- session search
- SQLite/FTS-style lightweight local state

Optional RAG:

```bash
bash setup-pi2-minimal.sh --profile rag
```

The RAG profile intentionally avoids installing `torch`, `sentence-transformers`, and `chromadb` by default. For Pi2, prefer remote embeddings or cloud memory providers. `sqlite-vec` is optional; Raspberry Pi 2 / ARMv7 wheels are not currently available from PyPI/piwheels, so the installer skips it and falls back to SQLite FTS5/remote embeddings. The IoT branch also uses plain `uvicorn` instead of `uvicorn[standard]` to avoid `uvloop` source builds on ARMv7. If you explicitly want local embeddings/vector indexing, install them manually and expect high RAM/compile cost.

## Multi-Pi2 shared memory / RAG architecture

For multiple Raspberry Pi 2 nodes, use the Pi2 devices as lightweight Hermes clients and put shared memory/RAG on a stronger central node. Do not run a full local embedding/vector stack on every Pi2.

Recommended layout:

```text
Pi2 kitchen ┐
Pi2 lab     ├── HTTP/LAN API ──> shared memory/RAG server
Pi2 garage  ┘                    ├── SQLite/Postgres memory store
                                  ├── embedding provider or LAN embedding model
                                  └── Qdrant/Chroma/pgvector vector index
```

Pi2 nodes should:

- run Hermes Agent with the core/native/rag Pi2 profile
- keep local short-term/session state lightweight
- send memory writes, document ingests, and RAG queries to the central API
- avoid local `torch`, `sentence-transformers`, and `chromadb`

The central node can be an x86 mini PC, NAS, Pi 4/5, VM, or cloud host. It should own embedding, vector indexing, deduplication, backups, and cross-device memory policy.

Use metadata on every shared memory/RAG item so nodes do not pollute one another's context:

```json
{
  "device_id": "pi2-kitchen",
  "scope": "global|device|room|user",
  "source": "conversation|note|sensor|manual",
  "created_at": "2026-06-25T00:00:00Z"
}
```

First implementation recommendation:

- start with a tiny central HTTP service backed by SQLite FTS5 for keyword search
- add remote embeddings and Qdrant/pgvector later only if semantic search is needed
- never mount one writable SQLite database over NFS/Samba for many Pi2 nodes; use an API or per-device local DBs that sync into the central server

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
