# Hermes Agent IoT / Pi2 Lite Edition

<p align="center">
  <img src="assets/hermes-icon-white.svg" alt="Hermes Agent" width="100">
</p>

This branch is the Raspberry Pi 2 / IoT usage profile for Hermes Agent. It keeps the native Hermes Agent code layout while making the default install path practical for ARMv7 / 1GB RAM devices.

This Pi2 Lite branch is primarily designed for embedded-system development: always-on controllers, sensor/automation nodes, lab devices, robotics gateways, home/industrial IoT boxes, and other constrained Linux deployments where low memory use, predictable dependencies, and remote-first AI services matter more than desktop-heavy local stacks.

Raspberry Pi 2 is the baseline target, not the upper limit. The same Pi2 Lite profile is intended for Pi2-class or better Linux systems, including Raspberry Pi 3/4/5, Pi Zero 2 W, ARM64 SBCs, x86 mini PCs, and VMs. Higher-spec devices can keep the conservative Pi2 defaults or enable heavier Hermes features later.

This is not a separate mini-agent. The Python package path, CLI entrypoint, tools, plugins, gateway, memory, cron, MCP, ACP, and provider architecture stay intact. The Pi2 Lite profile changes default installation choices and default enabled tool surface.

## Repository

Current repository:

```text
https://github.com/matttest0080-prog/hermes-agent-iot
```

Use this branch for the Pi2 Lite profile:

```text
pi2-lite
```

Upstream project:

```text
https://github.com/NousResearch/hermes-agent
```

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

The Pi2 Lite profile avoids eager install/use of heavy features:

- browser automation runtime
- local Chromium/Playwright-style stacks
- uvloop / `uvicorn[standard]`; dashboard/API use uvicorn's native Python asyncio mode
- image/video generation backends
- voice/STT dependencies such as faster-whisper
- TTS premium providers
- torch / sentence-transformers / chromadb by default
- dashboard/web UI extras unless explicitly requested
- messaging platform extras unless explicitly requested

The code remains present. Heavy features can be re-enabled later with `hermes tools` and the relevant Python extras.

## Recommended Pi2 Lite install

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git

git clone --depth 1 --branch pi2-lite https://github.com/matttest0080-prog/hermes-agent-iot.git
cd hermes-agent-iot
bash setup-pi2-minimal.sh --profile core

source ~/.hermes-venv/bin/activate
hermes setup model
hermes
```

Hermes requires Python `>=3.11,<3.14`. If your Raspberry Pi OS image ships older Python, install Python 3.11+ first.

## Install profiles

```bash
bash setup-pi2-minimal.sh --profile core
bash setup-pi2-minimal.sh --profile native
bash setup-pi2-minimal.sh --profile rag
```

Profiles:

- `core`: smallest practical Hermes CLI profile. Installs package through `pip install -e .[cli,pty]`, writes a config that disables heavy toolsets by default.
- `native`: core plus MCP/ACP/Home Assistant/MQTT/SMS extras. Still disables browser/media/messaging tool surfaces by default.
- `rag`: native plus lightweight document helpers and Honcho optional dependency. Remote embeddings are recommended; local torch stacks are not installed by default.

## What changed for Pi2

The Pi2 installer uses Hermes' native Python package metadata:

```bash
python -m pip install -e ".[cli,pty]"
```

or a broader extras set depending on profile. It does not hand-maintain a separate dependency list and does not patch source files during install.

Pi2 Lite intentionally uses uvicorn's native Python asyncio mode. Do not install `uvicorn[standard]` on Raspberry Pi 2; it pulls `uvloop`, which is not reliable on ARMv7/Pi2. If you need the web API/dashboard dependencies manually, use:

```bash
pip install fastapi uvicorn python-multipart
```

The installer writes a Pi2 config template only when `~/.hermes/config.yaml` does not already exist:

```text
templates/config.pi2-core.yaml
templates/config.pi2-native.yaml
templates/config.pi2-rag.yaml
```

Existing configs are not overwritten.

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

For multiple Pi2 devices, use each Pi2 as a lightweight Hermes client and share memory/RAG through a central LAN/cloud service:

```text
Pi2 nodes -> HTTP API -> shared memory/RAG server -> SQLite/Postgres + optional Qdrant/pgvector
```

The central server should handle embeddings, vector indexing, backups, deduplication, and cross-device scope metadata such as `device_id`, `room`, and `global|device|user`. Avoid having multiple Pi2 nodes write directly to one SQLite database over NFS/Samba.

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

## License

Hermes Agent is MIT licensed. See `LICENSE`.
