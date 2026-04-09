# Hindsight Memory Provider

Long-term memory with knowledge graph, entity resolution, and multi-strategy retrieval. Supports cloud and local (embedded) modes.

## Requirements

- **Cloud:** API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)
- **Local:** API key for a supported LLM provider (OpenAI, Anthropic, Gemini, Groq, MiniMax, or Ollama). Embeddings and reranking run locally — no additional API keys needed.

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

### Cloud Mode

Connects to the Hindsight Cloud API. Requires an API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io).

### Local Mode

Runs an embedded Hindsight server with built-in PostgreSQL. Requires an LLM API key (e.g. Groq, OpenAI, Anthropic) for memory extraction and synthesis. The daemon starts automatically in the background on first use and stops after 5 minutes of inactivity.

Daemon startup logs: `~/.hermes/logs/hindsight-embed.log`
Daemon runtime logs: `~/.hindsight/profiles/<profile>.log`

## Config

Config file: `~/.hermes/hindsight/config.json`

### Connection

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` or `local` |
| `api_url` | `https://api.hindsight.vectorize.io` | API URL (cloud mode) |
| `api_url` | `http://localhost:8888` | API URL (local mode, unused — daemon manages its own port) |

### Memory

| Key | Default | Description |
|-----|---------|-------------|
| `bank_id` | `hermes` | Memory bank name |
| `budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |

### Integration

| Key | Default | Description |
|-----|---------|-------------|
| `memory_mode` | `hybrid` | How memories are integrated into the agent |
| `prefetch_method` | `recall` | Method for automatic context injection |

**memory_mode:**
- `hybrid` — automatic context injection + tools available to the LLM
- `context` — automatic injection only, no tools exposed
- `tools` — tools only, no automatic injection

**prefetch_method:**
- `recall` — injects raw memory facts (fast)
- `reflect` — injects LLM-synthesized summary (slower, more coherent)

### Local Mode LLM

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `openai` | LLM provider: `openai`, `anthropic`, `gemini`, `groq`, `minimax`, `ollama` |
| `llm_model` | per-provider | Model name (e.g. `gpt-4o-mini`, `openai/gpt-oss-120b`) |
| `llm_base_url` | — | LLM Base URL override (e.g. `https://openrouter.ai/api/v1`) |

The LLM API key is stored in `~/.hermes/.env` as `HINDSIGHT_LLM_API_KEY`.

## Tools

Available in `hybrid` and `tools` memory modes:

| Tool | Description |
|------|-------------|
| `hindsight_retain` | Store information with auto entity extraction |
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
| `HINDSIGHT_MODE` | Override mode (`cloud` / `local`) |
