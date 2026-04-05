# Supermemory Memory Provider

Semantic long-term memory with profile recall, semantic search, explicit memory tools, and session-end conversation ingest.

## Requirements

- `pip install supermemory`
- Supermemory API key from [supermemory.ai](https://supermemory.ai)

## Setup

```bash
hermes memory setup    # select "supermemory"
```

Or manually:

```bash
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/supermemory.json`

| Key | Default | Description |
|-----|---------|-------------|
| `container_tag` | `hermes` | Container tag used for search and writes |
| `auto_recall` | `true` | Inject relevant memory context before turns |
| `auto_capture` | `true` | Store cleaned user-assistant turns after each response |
| `max_recall_results` | `10` | Max recalled items to format into context |
| `profile_frequency` | `50` | Include profile facts on first turn and every N turns |
| `capture_mode` | `all` | Skip tiny or trivial turns by default |
| `entity_context` | built-in default | Extraction guidance passed to Supermemory |
| `api_timeout` | `5.0` | Timeout for SDK and ingest requests |

## Tools

| Tool | Description |
|------|-------------|
| `supermemory_store` | Store an explicit memory |
| `supermemory_search` | Search memories by semantic similarity |
| `supermemory_forget` | Forget a memory by ID or best-match query |
| `supermemory_profile` | Retrieve persistent profile and recent context |

## Behavior

When enabled, Hermes can:

- prefetch relevant memory context before each turn
- store cleaned conversation turns after each completed response
- ingest the full session on session end for richer graph updates
- expose explicit tools for search, store, forget, and profile access
