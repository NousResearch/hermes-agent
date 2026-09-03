# Hermes Adapter for TencentDB Agent Memory v2

[简体中文](./README_CN.md) · English

This directory is a reference implementation for integrating Hermes with TencentDB Agent Memory v2 API. It implements a Hermes `MemoryProvider` that talks to an already-running Memory Gateway through the Python SDK. It does not start or manage a Gateway subprocess.

For standalone local usage, the recommended Gateway endpoint is `http://127.0.0.1:8420`. The default local convention is `api_key = "local"` and `service_id = "default"`. If your Gateway enables `TDAI_GATEWAY_API_KEY`, use the same value as `TDAI_MEMORY_API_KEY`.

## Architecture

```text
Hermes Agent
  └─ MemoryManager
       └─ memory_tencentdb_v2 provider
            ├─ sync_turn()        completed turn -> add_conversation (L0)
            ├─ prefetch()         search memories/core/scenarios before prompt
            ├─ tdai_memory_search tool
            ├─ tdai_conversation_search tool
            └─ tdai_read_scene tool
                 │
                 ▼
            tencentdb_agent_memory.MemoryClient
                 │ HTTP v2 API
                 ▼
            TencentDB Agent Memory Gateway (:8420 standalone, or remote service)
```

## v1 vs v2

| | v1 `memory_tencentdb` | v2 `memory_tencentdb_v2` |
|---|---|---|
| API | legacy `/recall`, `/capture`, `/search/*` | v2 `/v2/*` |
| HTTP client | raw `urllib.request` | `tencentdb_agent_memory` Python SDK (`httpx`) |
| Gateway lifecycle | may start a local Gateway subprocess | expects an external or container-managed Gateway |
| Standalone convention | localhost only | `endpoint=http://127.0.0.1:8420`, `api_key=local`, `service_id=default` |
| Tools | memory search, conversation search | plus `tdai_read_scene` |

## Quick Start

Recommended: run the installer from the repository root:

```bash
bash scripts/install-hermes-plugin-v2.sh
```

The script downloads and installs the Python SDK wheel, symlinks the `memory_tencentdb_v2` provider into Hermes' memory plugin directory, checks whether `~/.hermes/config.yaml` enables the provider, and writes `TDAI_MEMORY_ENDPOINT`, `TDAI_MEMORY_API_KEY`, and `TDAI_MEMORY_SERVICE_ID` to `~/.hermes/.env`. The SDK is installed into the Python environment Hermes actually uses: explicit `PYTHON_BIN` first, then `HERMES_VENV_DIR/bin/python` (default `~/.hermes/hermes-agent/venv/bin/python`), then system `python3` as a fallback. Override paths with `HERMES_HOME`, `HERMES_AGENT_DIR`, `HERMES_VENV_DIR`, `HERMES_MEMORY_PLUGIN_DIR`, `HERMES_ENV`, or `PYTHON_BIN` if needed.

For manual installation, follow the steps below.

### 1. Install the SDK

The Python SDK has not been published to PyPI yet. Download the wheel first, then install it:

```bash
curl -L -o tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl \
  "https://cnb.cool/tencent/cloud/nosql/nosql-utilities/-/commit-assets/download/cc74bd6dbc931727da9ab6907b5ab1a07d7afd9d/tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl"

# Use the Python interpreter Hermes actually runs with; common path:
~/.hermes/hermes-agent/venv/bin/python -m pip install ./tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl

# If your Hermes uses another Python environment, use that interpreter instead:
# PYTHON_BIN=/path/to/hermes/python
# "$PYTHON_BIN" -m pip install ./tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl
```

The Python import path is `tencentdb_agent_memory`.

### 2. Configure environment

For standalone local Gateway:

```bash
export TDAI_MEMORY_ENDPOINT="http://127.0.0.1:8420"
export TDAI_MEMORY_API_KEY="local"
export TDAI_MEMORY_SERVICE_ID="default"
```

If the Gateway enables `TDAI_GATEWAY_API_KEY`, set `TDAI_MEMORY_API_KEY` to the same value.

### 3. Activate in Hermes (`~/.hermes/config.yaml`)

```yaml
memory:
  provider: memory_tencentdb_v2
```

### 4. Install the provider into Hermes

Development symlink:

```bash
ln -s "$(pwd)/hermes-plugin/memory/memory_tencentdb_v2" \
  <hermes-agent>/plugins/memory/memory_tencentdb_v2
```

Deployment copy:

```bash
cp -r hermes-plugin/memory/memory_tencentdb_v2 \
  <hermes-agent>/plugins/memory/memory_tencentdb_v2
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TDAI_MEMORY_ENDPOINT` | `http://127.0.0.1:8420` | Memory Gateway URL |
| `TDAI_MEMORY_API_KEY` | `local` in standalone examples | Bearer token sent by the SDK |
| `TDAI_MEMORY_SERVICE_ID` | `default` in standalone examples | Memory space ID, sent as `x-tdai-service-id` |

## Provider Responsibilities

| Method / Tool | Purpose |
|---|---|
| `initialize(session_id)` | Create the SDK client and bind Hermes session ID |
| `sync_turn(user_content, assistant_content)` | Write completed turns to L0 through `add_conversation()` |
| `prefetch(query)` | Search L1 memories and read L3/L2 context before the next prompt |
| `tdai_memory_search` | Agent-callable L1 memory search |
| `tdai_conversation_search` | Agent-callable L0 conversation search |
| `tdai_read_scene` | Agent-callable L2 scene read |

## Using This as an Adapter Template

When adapting another Python Agent framework, copy the same pattern:

1. Initialize a `MemoryClient(endpoint, api_key, service_id)`.
2. After each completed turn, call `add_conversation()` with user and assistant messages.
3. Before the next prompt, call `search_atomic()`, `read_core()`, and optionally `list_scenarios()`.
4. Format the recalled memory as a clearly labeled context block.
5. Expose tools for active memory search and scene reading.
6. Keep the adapter best-effort: Memory failures should not block the Agent's main response path.

## Reliability

- Circuit breaker: 5 consecutive failures trigger a 60-second cooldown.
- Thread-safe state: internal mutations are protected by a lock.
- Graceful degradation: failed prefetch/tool calls return empty or user-friendly results instead of crashing the Agent.
