# LocalMem Memory Provider

Local LLM fact extraction (qwen2.5:7b) with semantic search (nomic-embed-text)
and ChromaDB vector store. Zero cloud, zero API tokens. Runs entirely on your
machine.

## Requirements

- Ollama running with `qwen2.5:7b` and `nomic-embed-text` pulled
- `pip install chromadb ollama`

## Setup

```bash
hermes memory setup    # select "localmem"
```

Or manually:
```bash
hermes config set memory.provider localmem
```

## Config

Config file: `$HERMES_HOME/localmem.json`

| Key | Default | Description |
|-----|---------|-------------|
| `user_id` | `hermes-user` | User identifier for memory scoping |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | LLM re-ranking for recall precision |
| `llm_model` | `qwen2.5:7b` | Ollama model for fact extraction |
| `embed_model` | `nomic-embed-text` | Ollama model for embeddings |
| `chroma_path` | `$HERMES_HOME/localmem_chroma` | ChromaDB storage path |

## Tools

| Tool | Description |
|------|-------------|
| `localmem_profile` | All stored memories about the user |
| `localmem_search` | Semantic search with optional reranking |
| `localmem_conclude` | Store a fact verbatim (no LLM extraction) |

## Architecture

```
Conversation → sync_turn() → qwen2.5:7b fact extraction → nomic-embed-text
embedding → ChromaDB upsert → localmem_search() → semantic query + optional
LLM re-rank
```

No external APIs, no telemetry, no cloud dependency.
