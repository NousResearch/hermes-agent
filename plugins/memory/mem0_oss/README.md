# Mem0 OSS Memory Provider

Self-hosted memory with full control over LLM, embedder, and vector store.

## Requirements

- `pip install mem0ai`
- An LLM provider (OpenAI API key, or Ollama running locally)
- If using Ollama: `pip install ollama`
- If using PGVector: `pip install psycopg2-binary`
- If using Milvus: `pip install pymilvus`

## Setup

```bash
hermes memory setup    # select "Mem0" → "Self-hosted (OSS)"
```

Or manually create `$HERMES_HOME/mem0_oss.json`:

```json
{
  "llm": {"provider": "openai", "config": {"model": "gpt-5.4", "temperature": 0.1}},
  "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
  "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/qdrant", "collection_name": "mem0"}},
  "user_id": "hermes-user",
  "agent_id": "hermes"
}
```

Then: `hermes config set memory.provider mem0_oss`

## Supported Providers

**LLMs:** OpenAI, Ollama
**Embedders:** OpenAI, Ollama
**Vector Stores:** Qdrant (local), PGVector, Milvus

## Memory Extraction

A custom fact extraction prompt filters out conversational noise (greetings,
small talk, filler) so only durable facts are stored. This runs automatically
on every conversation turn via `sync_turn`.

Explicit facts stored via `mem0_oss_conclude` bypass extraction entirely and
are saved verbatim.

## Tools

| Tool | Description |
|------|-------------|
| `mem0_oss_profile` | All stored memories about the user |
| `mem0_oss_search` | Semantic search with optional reranking |
| `mem0_oss_conclude` | Store a fact verbatim (no LLM extraction) |
