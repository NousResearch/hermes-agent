# Clude Memory Provider

Generative memory with semantic search, importance decay, emotional valence scoring, and graph-linked recall. Memories accumulate importance over time and trigger reflective consolidation cycles — the agent's understanding deepens automatically between sessions.

## Requirements

- A Clude API key from [clude.io](https://clude.io)

## Setup

```bash
hermes memory setup    # select "clude"
```

Or manually:

```bash
hermes config set memory.provider clude
echo "CLUDE_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/clude.json`

| Key | Default | Description |
|-----|---------|-------------|
| `api_url` | `https://clude.io` | Clude API base URL |

## Tools

| Tool | Description |
|------|-------------|
| `clude_recall` | Semantic search over memory — ranked by relevance, recency, and importance |
| `clude_store` | Store a fact, insight, or observation for cross-session recall |

## How it works

Each turn is stored as an episodic memory. `clude_recall` runs hybrid search (vector + BM25 + graph) and returns memories ranked by a weighted score combining semantic relevance, recency, importance, and retrieval frequency. Memories decay slowly over time but are boosted each time they're recalled, keeping frequently-used knowledge fresh.

In the background, Clude periodically runs dream cycles that consolidate memories into higher-level insights — without any action required from the agent.
