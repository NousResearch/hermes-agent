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

The setup wizard will install dependencies automatically via `uv` and walk you through configuration.

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
| `auto_recall` | `true` | Automatically recall memories before each turn |

### Retain

| Key | Default | Description |
|-----|---------|-------------|
| `auto_retain` | `true` | Automatically retain conversation turns |
| `retain_async` | `true` | Process retain asynchronously on the Hindsight server |
| `retain_every_n_turns` | `1` | Retain every N turns (1 = every turn) |
| `retain_context` | `conversation between Hermes Agent and the User` | Context label for retained memories |
| `retain_tags` | — | Default tags applied to retained memories; merged with per-call tool tags |
| `retain_source` | — | Optional `metadata.source` attached to retained memories |
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

Requires `hindsight-client >= 0.4.22`. The plugin auto-upgrades on session start if an older version is detected.
