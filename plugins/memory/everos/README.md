# EverOS Memory Provider

Structured memory extraction with episodic, profile, and foresight memory for Hermes Agent. Powered by [EverOS](https://github.com/EverMind-AI/EverOS) — 93% accuracy on the LoCoMo benchmark.

## Features

- **Auto-extraction**: Conversations are automatically analyzed to extract episodic memories, user profile facts, event logs, and foresight predictions
- **Multi-method retrieval**: keyword (BM25), vector semantic, hybrid, RRF fusion, and agentic (LLM-guided multi-round) search
- **Progressive profiles**: Builds a user profile over time from conversation patterns
- **Circuit breaker**: Gracefully handles EverOS server downtime without blocking the agent

## Prerequisites

1. **EverOS server** running via Docker:
   ```bash
   git clone https://github.com/EverMind-AI/EverOS.git
   cd EverOS
   docker compose up -d
   cp env.template .env
   # Edit .env: set LLM_API_KEY and VECTORIZE_API_KEY
   uv sync
   uv run python src/run.py
   ```

2. **Required services** (via Docker Compose):
   - MongoDB
   - Milvus (vector DB)
   - Elasticsearch
   - Redis

   See the [EverOS setup guide](https://github.com/EverMind-AI/EverOS#quick-start) for details.

## Setup

```bash
# Activate the provider
hermes memory setup
# Select "everos" from the list
# Enter your EverOS URL (default: http://localhost:1995)

# Or manually set in config.yaml:
hermes config set memory.provider everos
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVEROS_URL` | `http://localhost:1995` | EverOS API base URL |
| `EVEROS_USER_ID` | `hermes-user` | User identifier in EverOS |

### Config File

Non-secret settings can be stored in `$HERMES_HOME/everos.json`:

```json
{
  "url": "http://localhost:1995",
  "user_id": "hermes-user"
}
```

## Agent Tools

Once active, two tools are available to the agent:

### `everos_search`
Search memories by meaning across all types.

```json
{
  "query": "What did the user say about their trading preferences?",
  "method": "hybrid",
  "memory_types": ["episodic_memory", "profile"],
  "top_k": 10
}
```

**Retrieval methods:**
- `keyword` — BM25 text matching (fast, good for exact terms)
- `vector` — Semantic similarity search
- `hybrid` — Keyword + vector combined (recommended default)
- `rrf` — Reciprocal Rank Fusion of multiple methods
- `agentic` — LLM-guided multi-round retrieval (slowest, most thorough)

### `everos_recall`
Fetch memories by type (no semantic search, just retrieval).

```json
{
  "memory_type": "profile",
  "limit": 10
}
```

**Memory types:**
- `episodic_memory` — Narrative memories of events and conversations
- `profile` — Extracted user profile facts and preferences
- `foresight` — Predicted future associations and trends
- `event_log` — Atomic facts extracted from conversations

## CLI Commands

```bash
hermes everos status    # Connection status and memory stats
hermes everos config    # Show current configuration
hermes everos reset     # Delete all memories for current user
```

## How It Works

1. **sync_turn()** — After each conversation turn, both user and assistant messages are sent to EverOS via `POST /api/v1/memories`. EverOS automatically performs boundary detection and structured extraction.

2. **queue_prefetch()** — After each turn, a background search is queued using the turn content as the query. Results are cached for the next turn's `prefetch()`.

3. **on_session_end()** — Final flush of the last turn pair to ensure nothing is lost.

4. **on_memory_write()** — Built-in memory writes (MEMORY.md, USER.md) are mirrored to EverOS as assistant messages.

## Threading

All API calls run in daemon threads to avoid blocking the agent loop. A circuit breaker pauses calls after 5 consecutive failures for 120 seconds.

## License

This plugin is part of Hermes Agent (MIT License). EverOS is separately licensed under the Apache License 2.0.
