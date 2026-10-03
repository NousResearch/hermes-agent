# skill-rag

Semantic skill retrieval plugin for [Hermes Agent](https://hermes-agent.nousresearch.com).

Automatically recommends relevant skills from your `~/.hermes/skills/` library based on the user's message, using vector embeddings (OpenAI-compatible API) with BM25/FTS5 fallback.

## How It Works

```
User message
    ↓
build_query()          — extract intent from conversation history
    ↓
retrieve()             — cosine similarity via embeddings (LM Studio / Ollama / vLLM)
    ↓
check_fresh()          — detect stale skills → reindex if needed
    ↓
retrieve_bm25()        — fallback if vector results still stale
    ↓
build_context()        — inject into user message as <available_skills> block
```

The plugin hooks into `pre_llm_call` — before each LLM call, it searches for relevant skills and injects recommendations into the user message context.

### Skill Visibility Filtering

The plugin uses the **same visibility rules** as the system prompt builder:

| Filter | Source | Description |
|---|---|---|
| Disabled skills | `config.yaml → skills.disabled` | Permanently disabled via config |
| Platform mismatch | frontmatter `platforms:` | OS-specific skills on wrong platform |
| Environment mismatch | frontmatter `environments:` | Runtime-specific skills |
| Missing apps | frontmatter `requires_apps:` | Skills requiring uninstalled apps |
| Toolset conditions | `metadata.hermes.requires_toolsets` | Skills requiring unavailable toolsets |
| Fallback conditions | `metadata.hermes.fallback_for_toolsets` | Hidden when primary skill is available |

When the skill-rag plugin is active, the system prompt **skips** the static `<available_skills>` block (via a small patch to `prompt_builder.py`). The plugin dynamically injects relevant skills into the user message instead.

## Installation

Copy the `skill-rag/` directory to `~/.hermes/plugins/`:

```bash
cp -r skill-rag/ ~/.hermes/plugins/
```

### Dependencies

```bash
pip install requests numpy
```

Optional (for local embeddings without API):
```bash
pip install sentence-transformers  # ~2GB with PyTorch
```

## Configuration

### Behavioral Settings (config.yaml)

Behavioral settings are read from `~/.hermes/config.yaml` via the standard Hermes plugin config mechanism:

```yaml
plugins:
  entries:
    skill-rag:
      settings:
        provider: openai_compatible    # openai_compatible | local
        api_base: http://localhost:1234/v1  # LM Studio endpoint
        api_model: text-embedding-bge-m3    # Embedding model name
        top_k: 5                       # Number of recommendations
        threshold: 0.3                 # Min cosine similarity (0.75 too high for bge-m3)
        history_window: 4              # Recent messages for query
        assistant_truncate: 500        # Assistant response truncation
```

| Setting | Default | Description |
|---|---|---|
| `provider` | `openai_compatible` | Embedding provider: `openai_compatible` or `local` |
| `api_base` | `http://localhost:1234/v1` | Embedding API endpoint (OpenAI-compatible) |
| `api_model` | `text-embedding-bge-m3` | Model name for embedding |
| `top_k` | `5` | Number of skill recommendations |
| `threshold` | `0.3` | Min cosine similarity for vector search |
| `history_window` | `4` | Number of recent messages for query |
| `assistant_truncate` | `500` | Max assistant response length in query |

### Secret (Environment Variable)

`API_KEY` is a secret and stays in `~/.hermes/.env`:

```bash
SKILL_RAG_API_KEY=your-api-key-here  # Usually not needed for local APIs
```

### Supported Embedding Providers

| Provider | Setup | Notes |
|---|---|---|
| **LM Studio** | Load any embedding model, set `api_base: http://localhost:1234/v1` | Recommended for local use |
| **Ollama** | `ollama pull nomic-embed-text`, set `api_base: http://localhost:11434/v1` | Default endpoint (port 11434) |
| **vLLM** | Run with `--embedding-model`, set `api_base` accordingly | OpenAI-compatible API |
| **llama.cpp** | Server mode with embedding model, set `api_base` | OpenAI-compatible API |
| **local** | Set `provider: local`, `pip install sentence-transformers` | Offline, uses `intfloat/multilingual-e5-small` |

## Architecture

### Files

```
skill-rag/
├── plugin.yaml     # Plugin manifest (hooks, metadata)
├── config.py       # All configuration constants
├── indexer.py      # SQLite + FTS5 index, embedding, scanning
├── retrieval.py    # Query building, vector search, context assembly
├── __init__.py     # Plugin entry point, DI container, hooks
├── requirements.txt
└── README.md
```

### Data Flow

1. **Scanning**: `indexer.scan_skills()` walks `~/.hermes/skills/` and parses SKILL.md frontmatter
2. **Embedding**: Text is embedded via OpenAI-compatible API or local model
3. **Indexing**: Skills stored in SQLite with embeddings as BLOBs
4. **Search**: Vector cosine similarity (primary) or BM25/FTS5 (fallback)
5. **Injection**: Results formatted as `<available_skills>` block in user message

### Thread Safety

- `_init_lock`: Double-checked locking for plugin initialization (H2)
- `Indexer._lock`: Protects SQLite connection creation and closure (H3)
- `requests.Session`: Thread-local connection pooling
- SQLite WAL mode for concurrent read safety

### Security

- FTS5 query words stripped of `"` characters (H1)
- FTS5 table name hardcoded in SQL — no injection via f-string (M6)
- `_MAX_SKILL_FILE_SIZE = 1MB` — oversized files skipped (M3)
- All SQL uses parameterized queries
- Graceful degradation on API failures

## Usage

The plugin works automatically once installed. No manual activation needed.

When you send a message, the plugin:

1. Extracts intent from your message (strips injected blocks, tool outputs, etc.)
2. Searches for skills relevant to your query using vector embeddings
3. Applies the same visibility filtering as the system prompt (disabled, platform, toolsets)
4. Injects up to 5 relevant skills into the context as an `<available_skills>` block
5. You can then load a recommended skill with `skill_view(name="skill-name")`

### System Prompt Opt-Out

To avoid duplicating skills (system prompt + plugin injection), set this in `~/.hermes/config.yaml`:

```yaml
skills:
  prompt_index: false
```

This makes the system prompt skip the static `<available_skills>` block. The plugin becomes the single source of skill recommendations.

**This flag is generic** — it works regardless of which plugin manages skill discovery. It follows the project's plugin policy (`plugins/AGENTS.md`: never hardcode plugin-specific logic in core).

### Example

```
User: "Help me debug a Python async function"
→ Plugin injects: python-async-threading-gui (score: 0.85)
→ You can then: skill_view(name="python-async-threading-gui")
```

## Testing

```bash
# Unit tests (79 tests, no API required)
cd ~/.hermes/plugins/skill-rag
python -m pytest tests/test_skill_rag.py -v

# Integration tests (requires LM Studio or Ollama running)
python tests/test_integration.py
```

## Troubleshooting

### No recommendations appearing

- Check that `~/.hermes/skills/` contains skills with valid SKILL.md frontmatter
- Verify embedding API is running: `curl http://localhost:1234/v1/models`
- Check logs for `[skill-rag]` prefix

### Slow first call

First call initializes the index (scans all skills, embeds changed ones). Subsequent calls are fast.

### FTS5 unavailable warning

SQLite compiled without FTS5 support. Vector search still works; BM25 fallback is disabled.

## License

MIT
