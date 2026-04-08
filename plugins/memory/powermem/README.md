# PowerMem memory provider

[PowerMem](https://github.com/oceanbase/powermem) adds long-term memory to Hermes Agent: LLM-based extraction, deduplication, hybrid retrieval (vector + full-text + optional graph), Ebbinghaus-style decay, and multi-agent scoping.

## How Hermes uses PowerMem

The plugin runs PowerMem **inside the Hermes process** as a Python library. You do **not** need to start `powermem-server` for Hermes to store and search memories, as long as `powermem` is installed and `$HERMES_HOME/.env` contains a valid PowerMem configuration (vector store + embedder; LLM strongly recommended for intelligent extraction on each turn).

Optional: run the [HTTP API server](https://github.com/oceanbase/powermem/blob/main/docs/api/0005-api_server.md) or [Docker image](https://github.com/oceanbase/powermem/blob/main/README.md) if you want a standalone Dashboard (`/dashboard/`), REST API, or to share the same backend with non-Python clients. Hermes still reads the same `.env` semantics when using the SDK path.

## Prerequisites

- Python **3.11+**
- Hermes Agent with this plugin available
- A working PowerMem stack: at minimum **vector store** + **embedding** settings in `.env`; **LLM** settings for automatic fact extraction on conversation turns (see upstream [configuration guide](https://github.com/oceanbase/powermem/blob/main/docs/guides/0003-configuration.md))

## 1. Install the `powermem` package

Install into the same environment as Hermes:

```bash
uv pip install powermem
# or
pip install powermem
```

Install Hermes with the PowerMem extra (pulls in `powermem`):

```bash
pip install "hermes-agent[powermem]"
```

From a Hermes source checkout:

```bash
uv pip install -e ".[powermem]"
```

## 2. Configure PowerMem (`.env`)

PowerMem loads settings from an environment file. For Hermes, put **PowerMem variables in `$HERMES_HOME/.env`** (the same file as your API keys). On startup the plugin sets `POWERMEM_ENV_FILE` to that path when the file exists, so the SDK picks it up automatically.

**Suggested workflow:**

1. Open the upstream [`.env.example`](https://github.com/oceanbase/powermem/blob/main/.env.example) and copy the variables you need into `$HERMES_HOME/.env`.
2. Fill in at least:
   - **Vector store** — e.g. `VECTOR_STORE_*` (OceanBase, PostgreSQL, SQLite, embedded seekdb / `ob_path`, etc.; see [configuration guide](https://github.com/oceanbase/powermem/blob/main/docs/guides/0003-configuration.md)).
   - **Embeddings** — e.g. `EMBEDDING_*` (provider, model, API keys).
   - **LLM** (recommended) — e.g. `LLM_*` for intelligent `add` / extraction during turns.

Alternatively, after `pip install powermem`, you can run **`pmem config init`** in a scratch directory to generate a template `.env`, then merge the PowerMem-related lines into `$HERMES_HOME/.env`. See the [CLI guide](https://github.com/oceanbase/powermem/blob/main/docs/guides/0012-cli_usage.md).

## 3. Optional: deploy PowerMem as a standalone service

These are **not required** for Hermes, but useful for ops, debugging, or other apps using the same memory:

**API server + Dashboard** (same config as SDK — `.env`):

```bash
powermem-server --host 0.0.0.0 --port 8000
```

**Docker** (mount your `.env`):

```bash
docker run -d \
  --name powermem-server \
  -p 8000:8000 \
  -v "$(pwd)/.env:/app/.env:ro" \
  --env-file .env \
  oceanbase/powermem-server:latest
```

## 4. Enable PowerMem in Hermes

```bash
hermes memory setup
# choose powermem; set user_id / agent_id if prompted

hermes config set memory.provider powermem
```

Verify:

```bash
hermes doctor
```

## Hermes-specific configuration

1. **`$HERMES_HOME/.env`** — PowerMem backend settings (`VECTOR_STORE_*`, `LLM_*`, `EMBEDDING_*`, etc.). The plugin wires `POWERMEM_ENV_FILE` to this file when present.

2. **Optional `$HERMES_HOME/powermem.json`** — Hermes-side overrides (often written by `hermes memory setup`):

   ```json
   {
     "enabled": true,
     "user_id": "hermes-user",
     "agent_id": "hermes"
   }
   ```

3. **Gateway users** — When using a messaging platform integration, the platform `user_id` is passed through; the JSON defaults above mainly apply to CLI-only sessions.

4. **Environment shortcuts** (optional): `POWERMEM_USER_ID`, `POWERMEM_AGENT_ID` (see plugin docstring in `__init__.py`).

Full key reference: upstream [configuration guide](https://github.com/oceanbase/powermem/blob/main/docs/guides/0003-configuration.md).

## Tools exposed to the agent

| Tool | Purpose |
|------|--------|
| `powermem_search` | Semantic / hybrid search over memories |
| `powermem_profile` | List recent memories for this user (and agent scope) |
| `powermem_add` | Add an explicit memory (`infer=false`, verbatim; aligns with PowerMem `Memory.add`) |

Each completed turn is also sent to PowerMem with **intelligent extraction** (`infer=true`) in the background when LLM settings are configured.

## Requirements summary

- Python 3.11+
- `powermem` installed in the Hermes environment
- Valid PowerMem configuration in `$HERMES_HOME/.env` (vector store + embedder; LLM recommended)
