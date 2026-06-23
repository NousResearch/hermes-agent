# memory-tencentdb Memory Provider (Hermes)

Hermes-side [`MemoryProvider`](../../../../../hermes-agent/agent/memory_provider.py)
adapter for the **memory-tencentdb** four-layer memory system
(L0 conversation capture → L1 episodic extraction → L2 scene blocks → L3 persona synthesis).

The heavy lifting — capture, extraction, storage, recall, pipeline scheduling —
runs in a Node.js **Gateway** sidecar (shipped by the same package as the
OpenClaw plugin). This Python provider is a thin HTTP client + process
supervisor that plugs the Gateway into Hermes's lifecycle.

## Architecture

```
Hermes Agent (Python)
  └─ MemoryManager
       └─ MemoryTencentdbProvider        (this directory)
            ├─ GatewaySupervisor          — starts / health-checks the sidecar
            └─ MemoryTencentdbSdkClient   — POST /recall, /capture, /search/*, /session/end
                    │
                    ▼  HTTP (127.0.0.1:8420 by default)
            memory-tencentdb Gateway (Node.js)
               └─ memory-tencentdb Core
                    ├─ L0  Conversation store      (SQLite / TCVDB + JSONL)
                    ├─ L1  Episodic extraction     (LLM + vector dedup)
                    ├─ L2  Scene blocks            (Markdown under data dir)
                    ├─ L3  Persona synthesis       (persona.md)
                    └─ Storage backends: SQLite + sqlite-vec  OR  Tencent VectorDB
```

Hermes lifecycle → Gateway mapping:

| Hermes hook / call          | Gateway endpoint | Behavior                                                   |
|-----------------------------|------------------|------------------------------------------------------------|
| `prefetch(query)`           | `POST /recall`   | Synchronous. Returns `<memory-context>` text for injection |
| `sync_turn(user, assistant)`| `POST /capture`  | Fire-and-forget on a background daemon thread (max 4 in-flight) |
| `shutdown()` / `on_session_end` | `POST /session/end` | Flush pending pipeline work                             |
| `get_tool_schemas()`        | —                | Advertises two LLM tools (see below)                       |

Reliability features baked into the provider:

- **Circuit breaker** — 5 consecutive Gateway failures → pause all calls for 60 s.
- **Back-pressure on capture** — at most 4 in-flight `sync_turn` threads; a 5th
  waits up to 5 s for the oldest one before starting (Gateway hangs can't grow
  threads unboundedly).
- **Supervised startup** — if `MEMORY_TENCENTDB_GATEWAY_CMD` is set (or the
  provider auto-discovers `src/gateway/server.ts`, see below), it starts the
  sidecar, polls `/health` for up to 30 s, and tails `gateway.stderr.log` on
  crash for diagnostics.
- **Zero-config auto-discovery** — when `MEMORY_TENCENTDB_GATEWAY_CMD` is
  unset, the provider looks for `src/gateway/server.ts` next to the plugin
  checkout (in-tree) and, as a last resort, under
  `~/.memory-tencentdb/tdai-memory-openclaw-plugin/` (preferred),
  `~/tdai-memory-openclaw-plugin/` (legacy), and
  `~/.hermes/plugins/tdai-memory-openclaw-plugin/`.
  A fresh `git clone` therefore usually works without any extra env wiring —
  override with the env var when you need a non-standard layout.

## Installation Location

This directory (`hermes-plugin/memory/memory_tencentdb/`) is the **source of
truth** for the provider; Hermes does **not** load it from here. At startup
Hermes scans two locations for memory providers, in precedence order (see
`hermes-agent/plugins/memory/__init__.py`):

1. **Bundled** — `<hermes-agent-checkout>/plugins/memory/<name>/`
   **This is the path memory_tencentdb ships under.** It sits alongside the
   other in-tree providers (`byterover/`, `honcho/`, `mem0/`, `hindsight/`,
   …). Bundled entries take precedence over user-installed ones on name
   collision.
2. **User-installed** — `$HERMES_HOME/plugins/<name>/`, where
   `$HERMES_HOME` defaults to `~/.hermes` (see
   `hermes_constants.get_hermes_home()`). This path is for third-party
   providers; we don't use it for memory_tencentdb.

**The trailing directory name must be exactly `memory_tencentdb`** — Hermes
uses that directory name as the provider key; it must match
`plugin.yaml::name` and the value of `memory.provider` in `config.yaml`.
(The hyphenated form `memory-tencentdb` is a *config-side alias*, not a
valid directory name.)

Pick one of the two installation styles:

**Install A — symlink (recommended for developers working on both repos
simultaneously):** keeps this repo as the single source of truth so
`git pull` in the plugin repo is immediately visible to Hermes.

```bash
# from the tdai-memory-openclaw-plugin checkout:
ln -s "$(pwd)/hermes-plugin/memory/memory_tencentdb" \
      <hermes-agent-checkout>/plugins/memory/memory_tencentdb
```

**Install B — copy (shipped alongside hermes-agent):** freezes a specific
version of the provider inside the hermes-agent tree. This is how
memory_tencentdb is currently vendored in this repo pair — the two copies
under `tdai-memory-openclaw-plugin/hermes-plugin/memory/memory_tencentdb/`
and `hermes-agent/plugins/memory/memory_tencentdb/` are kept in sync
manually.

```bash
cp -r tdai-memory-openclaw-plugin/hermes-plugin/memory/memory_tencentdb \
      hermes-agent/plugins/memory/memory_tencentdb
```

Verify Hermes sees the provider:

```bash
$ cd <hermes-agent-checkout>
$ python -c 'from plugins.memory import discover_memory_providers; \
             [print(n, a) for n, _, a in discover_memory_providers()]'
memory_tencentdb True
...
```

If the provider does not appear:
- confirm the target path is `hermes-agent/plugins/memory/memory_tencentdb/`
  (underscore, not hyphen);
- confirm `__init__.py` and `plugin.yaml` sit directly inside that dir;
- the discovery scan requires `__init__.py` to contain the literal string
  `MemoryProvider` or `register_memory_provider` — both are present in
  this provider, so this is a non-issue as long as the file is the one
  from this repo.

> The **Gateway source code** (Node.js sidecar under `src/gateway/`) stays
> in the `tdai-memory-openclaw-plugin` checkout and does NOT need to be
> copied into hermes-agent — the Python provider auto-discovers it via
> the paths listed in Option A below, or via `MEMORY_TENCENTDB_GATEWAY_CMD`.

## Setup

### 1. Activate in Hermes (`~/.hermes/config.yaml`)

```yaml
memory:
  provider: memory_tencentdb   # canonical name
  # Aliases accepted for backward compatibility: `memory-tencentdb`, `tdai`
```

### 2. Provide Gateway runtime + LLM credentials

At minimum the Gateway needs an OpenAI-compatible endpoint for L1/L2/L3
extraction. Set these in the Hermes process environment:

```bash
export MEMORY_TENCENTDB_LLM_API_KEY="sk-..."
export MEMORY_TENCENTDB_LLM_BASE_URL="https://api.openai.com/v1"   # optional
export MEMORY_TENCENTDB_LLM_MODEL="gpt-4o"                         # optional
```

### 3. Start the Gateway

You have three options; pick whichever fits your deployment.

**Option A — Auto-discovery (zero-config).** If the plugin checkout sits at
one of the well-known paths, the provider will find `src/gateway/server.ts`
on its own and `Popen()` it as `node --import tsx <path>`. Searched paths, in
order:

1. In-tree: `<plugin-root>/src/gateway/server.ts` (when Hermes loads this
   provider from a checkout of this repo).
2. `~/.memory-tencentdb/tdai-memory-openclaw-plugin/src/gateway/server.ts` (preferred install location)
3. `~/tdai-memory-openclaw-plugin/src/gateway/server.ts` (legacy)
4. `~/.hermes/plugins/tdai-memory-openclaw-plugin/src/gateway/server.ts`

No environment variables required beyond the LLM credentials above. A line
like

```
INFO plugins.memory.memory_tencentdb: memory-tencentdb Gateway command auto-discovered: /…/src/gateway/server.ts
```

will appear in `~/.hermes/logs/agent.log` on startup.

**Option B — Explicit auto-start.** Override or disable discovery by setting
the command yourself:

```bash
export MEMORY_TENCENTDB_GATEWAY_CMD="node --import tsx /abs/path/to/tdai-memory-openclaw-plugin/src/gateway/server.ts"
```

The provider will `Popen()` this command on `initialize()`, wait for
`GET /health` to report `ok`/`degraded`, and tail stderr on crash.

**Option C — Run it yourself.** Start the Gateway separately on the default
port (`127.0.0.1:8420`) before launching Hermes; the provider will detect it
via `/health` and skip the subprocess-launch path.

```bash
cd tdai-memory-openclaw-plugin
node --import tsx src/gateway/server.ts
```

> Storage backend (SQLite vs Tencent VectorDB), embedding config, pipeline
> cadence, recall strategy, etc. are all **Gateway-side** settings. For
> OpenClaw installs they live in `~/.openclaw/openclaw.json`; for standalone
> Hermes deployments configure the Gateway via its own config file or env.
> See the plugin's top-level [README](../../../README.md) for the full
> configuration schema.

## Environment Variables

### Gateway location

| Variable                          | Default             | Description                                               |
|-----------------------------------|---------------------|-----------------------------------------------------------|
| `MEMORY_TENCENTDB_GATEWAY_HOST`   | `127.0.0.1`         | Gateway host                                              |
| `MEMORY_TENCENTDB_GATEWAY_PORT`   | `8420`              | Gateway port (must be 1..65535; invalid values fall back) |
| `MEMORY_TENCENTDB_GATEWAY_CMD`    | —                   | If set, the provider auto-starts the Gateway with this command. If unset, the provider auto-discovers `src/gateway/server.ts` next to the checkout or under `$HOME` (see Option A above) |
| `MEMORY_TENCENTDB_LOG_DIR`        | `~/.hermes/logs/memory_tencentdb` | Where the supervisor writes `gateway.stdout.log` / `gateway.stderr.log` |

### Gateway data directory (owned by the Gateway, not this provider)

The L0~L3 data directory is resolved **inside the Gateway** (`src/gateway/config.ts`),
not here. Priority:

1. `TDAI_DATA_DIR` env var
2. `data.baseDir` from a `tdai-gateway.yaml` / `tdai-gateway.json` config file
3. Default: `~/.memory-tencentdb/memory-tdai`
   (Override the parent dir with `MEMORY_TENCENTDB_ROOT` if needed.)
4. Legacy fallback: if `~/.memory-tencentdb/memory-tdai` does not exist but the
   pre-0.4 location `~/memory-tdai` does, the Gateway keeps using the legacy
   dir and prints a one-line deprecation warning to stderr. Run
   `install_hermes_memory_tencentdb.sh` to migrate it automatically.

Hermes forwards the inherited environment to the Gateway subprocess, so
setting `TDAI_DATA_DIR` before launching Hermes is enough to override it.
The old `MEMORY_TENCENTDB_DATA_DIR` env var is no longer read — it was never
consumed by the Gateway anyway (names did not match), so removing it just
eliminates a silent no-op.

### Gateway LLM (consumed by the Node sidecar, not by this provider)

| Variable                          | Default                      | Description                         |
|-----------------------------------|------------------------------|-------------------------------------|
| `MEMORY_TENCENTDB_LLM_API_KEY`    | —                            | LLM API key (required for L1/L2/L3) |
| `MEMORY_TENCENTDB_LLM_BASE_URL`   | `https://api.openai.com/v1`  | OpenAI-compatible API base URL      |
| `MEMORY_TENCENTDB_LLM_MODEL`      | `gpt-4o`                     | Model name                          |

> ⚠️ Only `MEMORY_TENCENTDB_*` env vars are honored by this provider for the
> Gateway location and LLM credentials. Data-directory resolution is
> deliberately delegated to the Gateway via `TDAI_DATA_DIR` (see above) so
> the provider and the Gateway can never disagree about where L0~L3 live.

## LLM Tools

This provider exposes two tools to the model via `get_tool_schemas()`:

| Tool                                   | Purpose                                           | Args                                            |
|----------------------------------------|---------------------------------------------------|-------------------------------------------------|
| `memory_tencentdb_memory_search`       | Search L1 structured long-term memories           | `query` (required), `limit` (1..20, default 5), `type` (`persona`/`episodic`/`instruction`) |
| `memory_tencentdb_conversation_search` | Search L0 raw conversation history                | `query` (required), `limit` (1..20, default 5)  |

Tool-call arguments are defensively coerced: `limit` accepts ints, numeric
strings, and floats, rejects bools, and is clamped to `[1, 20]` with a
warning on garbage input.

> These are the **only** tool names registered with the LLM. The old
> `tdai_memory_search` / `tdai_conversation_search` names are not served by
> this provider — if older transcripts reference them, `handle_tool_call`
> will return an "Unknown tool" error.

## Plugin Metadata (`plugin.yaml`)

```yaml
name: memory_tencentdb            # canonical provider name
display_name: memory-tencentdb
hooks:
  - on_memory_write               # reserved; not yet mirrored to the Gateway
  - on_session_end                # triggers POST /session/end
aliases:
  - tdai                          # legacy config value still resolves here
  - memory-tencentdb              # hyphenated form resolves here too
```

## Troubleshooting

- **"memory-tencentdb Gateway not available"** on startup: either
  `MEMORY_TENCENTDB_GATEWAY_CMD` is unset *and* auto-discovery did not find
  `src/gateway/server.ts` *and* nothing is listening on `8420`, or the
  sidecar crashed. Check `~/.hermes/logs/memory_tencentdb/gateway.stderr.log`
  (override with `MEMORY_TENCENTDB_LOG_DIR`). To confirm auto-discovery was
  attempted, enable `DEBUG` logging and look for
  `memory-tencentdb Gateway auto-discovery found no server.ts under: …`;
  that log line enumerates every path that was searched.
- **Gateway starts from the wrong checkout**: auto-discovery walks a fixed
  preference list (in-tree first, then `$HOME`). If you want to pin a
  specific path, set `MEMORY_TENCENTDB_GATEWAY_CMD` explicitly — it always
  wins over discovery.
- **Search tools silently missing from the LLM**: `get_tool_schemas()`
  returns `[]` until either the Gateway is reachable or one of
  `MEMORY_TENCENTDB_GATEWAY_CMD` / `MEMORY_TENCENTDB_GATEWAY_PORT` is set
  in the environment. Set the env var so the tools are advertised
  optimistically at registration time.
- **"circuit breaker tripped"** warnings: five consecutive Gateway errors
  were observed. Calls are paused for 60 s; check Gateway health and logs.
- **Capture backlog warnings**: Gateway is slow or hung — `sync_turn` is
  tracking ≥ 4 in-flight threads. Inspect Gateway logs for stuck L1
  extractions or LLM timeouts.
