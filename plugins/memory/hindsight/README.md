# Hindsight Memory Provider

Long-term memory with knowledge graph, entity resolution, and multi-strategy retrieval. Supports cloud, local embedded, and local external modes.

## Requirements

- **Cloud:** API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)
- **Local Embedded:** API key for a supported LLM provider (OpenAI, Anthropic, Gemini, Groq, OpenRouter, MiniMax, Ollama, or any OpenAI-compatible endpoint). Embeddings and reranking run locally — no additional API keys needed.
- **Local External:** A running Hindsight instance (Docker or self-hosted) reachable over HTTP.

## Setup

```bash
hermes memory setup    # select "hindsight"
```

The setup wizard installs dependencies automatically via `uv`, walks you through configuration, and offers to seed the bank with a **starter memory template** (a curated set of dispositions/instructions for common agent roles) — you can skip it, and it warns before overwriting an already-configured bank.

Or manually (cloud mode with defaults):
```bash
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

### Cloud

Connects to the Hindsight Cloud API. Requires an API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io).

### Local Embedded

Hermes spins up a local Hindsight daemon with built-in PostgreSQL. Requires an LLM API key for memory extraction and synthesis. The daemon starts automatically in the background on first use and stops after 5 minutes of inactivity.

Supports any OpenAI-compatible LLM endpoint (llama.cpp, vLLM, LM Studio, etc.) — pick `openai_compatible` as the provider and enter the base URL.

Daemon startup logs: `~/.hermes/logs/hindsight-embed.log`
Daemon runtime logs: `~/.hindsight/profiles/<profile>.log`

To open the Hindsight web UI (local embedded mode only):
```bash
hindsight-embed -p hermes ui start
```

### Local External

Points the plugin at an existing Hindsight instance you're already running (Docker, self-hosted, etc.). No daemon management — just a URL and an optional API key.

## Config

Config file: `~/.hermes/hindsight/config.json`

### Connection

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud`, `local_embedded`, or `local_external` |
| `api_url` | `https://api.hindsight.vectorize.io` | API URL (cloud and local_external modes) |

### Memory Bank

| Key | Default | Description |
|-----|---------|-------------|
| `bank_id` | `hermes` | Memory bank name (static fallback used when `bank_id_template` is unset or resolves empty) |
| `bank_id_template` | — | Optional template to derive the bank name dynamically. Placeholders: `{profile}`, `{workspace}`, `{platform}`, `{user}`, `{session}`. Example: `hermes-{profile}` isolates memory per active Hermes profile. Empty placeholders collapse cleanly (e.g. `hermes-{user}` with no user becomes `hermes`). |
| `bank_mission` | — | Reflect mission (identity/framing for reflect reasoning). Applied via Banks API. |
| `bank_retain_mission` | — | Retain mission (steers what gets extracted). Applied via Banks API. |

### Recall

| Key | Default | Description |
|-----|---------|-------------|
| `recall_budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |
| `recall_prefetch_method` | `recall` | Auto-recall method: `recall` (raw facts) or `reflect` (LLM synthesis) |
| `recall_max_tokens` | `4096` | Maximum tokens for recall results |
| `recall_max_input_chars` | `800` | Maximum input query length for auto-recall |
| `recall_prompt_preamble` | — | Custom preamble for recalled memories in context |
| `recall_tags` | — | Tags to filter when searching memories |
| `recall_tags_match` | `any` | Tag matching mode: `any` / `all` / `any_strict` / `all_strict` |
| `recall_types` | `observation` | Fact types surfaced by recall (both auto-recall and the `hindsight_recall` tool). Comma-separated string or JSON list. **Default narrowed to `observation` only** (see "Behavior change" below). Set to `observation,world,experience` to also include raw facts. |
| `auto_recall` | `true` | Automatically recall memories before each turn |
| `recall_sync` | `false` | Recall synchronously against the *current* message each turn (higher relevance, adds recall latency). Default off: recall runs in the background and is injected on the next turn. |
| `recall_indicator` | `true` | Show a `👁️ Hindsight — recalled N memories` status line when auto-recall injects memory. Turn off for customer-facing agents. |
| `desktop_context_root` | — | Root directory for Desktop workstream recall (see "Desktop workstream recall" below). A session whose working directory resolves under `<root>/<domain>/` gets that domain key's `extra_tags` from `thread_routing.json` as an `any_strict` recall filter. Absent/empty = layer inert. |

> **Behavior change — `recall_types` defaults to `observation` only.**
>
> Previously recall returned all three fact types. It now returns only observations.
>
> Per [Hindsight's docs](https://hindsight.vectorize.io/developer/observations), observations are the **consolidated** knowledge layer Hindsight builds on top of raw facts: deduplicated beliefs grounded in evidence, refined as new facts arrive, with proof counts and freshness signals. Raw `world` / `experience` facts are the individual supporting evidence that feeds them. For per-turn context injection, observations are denser per token and avoid feeding the model multiple raw facts that one observation already summarizes.
>
> Restore the broad recall with `"recall_types": "observation,world,experience"` (string or JSON list) in `~/.hermes/hindsight/config.json`. This applies to **both** auto-recall and the `hindsight_recall` tool — both read the same `recall_types` setting (the tool schema has no per-call `types` argument), so narrowing the default narrows both paths.

### Desktop workstream recall

Per-session, workstream-scoped recall for Desktop sessions. Desktop sessions do not carry a messaging `thread_id`, so recall normally falls back to the global `recall_tags` filter for every project. This opt-in layer derives a domain-specific tag filter from the session's working directory instead.

**Prerequisites**

- `desktop_context_root` (config key, see the Recall table) AND `"recall_sync": true` must both be set. With asynchronous prefetch the layer trips closed and the configured `recall_tags` baseline stays in force — feature disablement, not a confidentiality guarantee.
- `thread_routing.json` must contain a bare domain key — e.g. `{"myproject": {"extra_tags": ["myproject"]}}` — for each domain being routed. A key without the `<platform>:<id>` colon form is a domain key; it uses the same entry shape and validation as thread keys.
- The routed domain's memories must already carry those tags. This filters memories that already carry the selected tags; it does **not** infer tags from folders or context files, backfill existing memories, or add domain tags to retains. Retain and channel-tag generation are unchanged.

Both files are profile-scoped and live under the **selected profile's** Hermes home: `$HERMES_HOME/hindsight/config.json` and `$HERMES_HOME/hindsight/thread_routing.json`. `$HERMES_HOME` resolves per `get_hermes_home()` — a context-local override first, then the `HERMES_HOME` environment variable, then the default `~/.hermes` — so in a named profile the files belong in that profile's home, not in the default profile's `~/.hermes`. Add them as additions to the existing connection config:

**Settings keys** — `$HERMES_HOME/hindsight/config.json` (add to your existing connection config):

```json
{
  "desktop_context_root": "/srv/workspaces",
  "recall_sync": true
}
```

**Routing table** — `$HERMES_HOME/hindsight/thread_routing.json` (a bare domain-key table):

```json
{
  "myproject": { "extra_tags": ["myproject"] }
}
```

Profiles do not read each other's files, with one legacy exception: `thread_routing.json` has no legacy path, but `config.json` does — `_load_config()` still falls back to the shared `~/.hindsight/config.json` when the profile's own `config.json` does not exist, so a profile that has never created one can pick up Hindsight settings from that legacy shared file. Both files are loaded at provider initialize(); changing either file requires provider reinitialization (another recall call is not enough). The session working directory is resolved per recall call.

**Matching**

- The session cwd is resolved per recall call through the same resolver that selects the project context pack (session override first, terminal-scope fallback), and compared against `desktop_context_root` after canonical `Path.resolve()` on both sides — symlinks are followed, so a link that escapes the root does not match.
- The first path component below the root is the route key: `/workspaces/myproject/src` routes as `myproject`.
- A cwd equal to the root or outside the root does not match and keeps the global recall filter.

**Semantics**

- Relevance filtering, not authorization or isolation: sessions that share a bank or a filter still see whatever the effective filter admits.
- A matching domain route REPLACES `recall_tags` (it is not intersected with it) and uses `tags_match: any_strict`, so memories carrying at least one selected tag are eligible — including multi-tag memories shared across domains — while untagged memories are excluded.
- The Desktop domain layer activates only when the provider's platform is exactly `desktop` and no `platform:thread` override applied. The separate `platform:thread` branch activates on its own and does not require `desktop_context_root`.
- No match, missing/malformed routing table, missing root key, blank domain tags, or an unresolvable cwd keeps the configured baseline. One deliberate exception: a terminal-policy refusal propagates instead of falling back.

### Retain

| Key | Default | Description |
|-----|---------|-------------|
| `auto_retain` | `true` | Automatically retain conversation turns |
| `retain_async` | `true` | Process retain asynchronously on the Hindsight server |
| `retain_every_n_turns` | `1` | Retain every N turns (1 = every turn) |
| `retain_context` | `conversation between Hermes Agent and the User` | Context label for retained memories |
| `retain_tags` | — | Default tags applied to retained memories; merged with per-call tool tags |
| `retain_source` | — | Opt-in `metadata.source` attached to retained memories (identifies the storing client, e.g. `hermes`). Empty by default — no attribution tag ships unless you set it. |
| `retain_indicator` | `true` | Show a `👁️ Hindsight — saving to memory…` status line when a turn is saved. Turn off for customer-facing agents. |
| `retain_user_prefix` | `User` | Label used before user turns in auto-retained transcripts |
| `retain_assistant_prefix` | `Assistant` | Label used before assistant turns in auto-retained transcripts |

### Integration

| Key | Default | Description |
|-----|---------|-------------|
| `memory_mode` | `hybrid` | How memories are integrated into the agent |

**memory_mode:**
- `hybrid` — automatic context injection + tools available to the LLM
- `context` — automatic injection only, no tools exposed
- `tools` — tools only, no automatic injection

### Local Embedded LLM

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `openai` | `openai`, `anthropic`, `gemini`, `groq`, `openrouter`, `minimax`, `ollama`, `lmstudio`, `openai_compatible` |
| `llm_model` | per-provider | Model name (e.g. `gpt-4o-mini`, `qwen/qwen3.5-9b`) |
| `llm_base_url` | — | Endpoint URL for `openai_compatible` (e.g. `http://192.168.1.10:8080/v1`) |

The LLM API key is stored in `~/.hermes/.env` as `HINDSIGHT_LLM_API_KEY`.

## Tools

Available in `hybrid` and `tools` memory modes:

| Tool | Description |
|------|-------------|
| `hindsight_retain` | Store information with auto entity extraction; supports optional per-call `tags` |
| `hindsight_recall` | Multi-strategy search (semantic + entity graph) |
| `hindsight_reflect` | Cross-memory synthesis (LLM-powered) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HINDSIGHT_API_KEY` | API key for Hindsight Cloud |
| `HINDSIGHT_LLM_API_KEY` | LLM API key for local mode |
| `HINDSIGHT_API_LLM_BASE_URL` | LLM Base URL for local mode (e.g. OpenRouter) |
| `HINDSIGHT_API_URL` | Override API endpoint |
| `HINDSIGHT_BANK_ID` | Override bank name |
| `HINDSIGHT_BUDGET` | Override recall budget |
| `HINDSIGHT_MODE` | Override mode (`cloud`, `local_embedded`, `local_external`) |

## Client Version

Requires `hindsight-client >= 0.6.1`. The plugin auto-upgrades on session start if an older version is detected.
