# Cognee Memory Provider

Open-source memory control plane with vector + knowledge-graph storage and semantic recall.

Built on [Cognee](https://github.com/topoteretes/cognee) — extracts entities, builds associations, and enables graph-based associative memory that traditional vector-only stores cannot provide.

## How It Works

Cognee stores facts as a **knowledge graph**: each `cognee_remember` call creates nodes and edges, not just vectors. Recall uses semantic search AND graph traversal, finding related memories that a pure embedding lookup would miss.

```
User says: "I prefer Rust over Go for systems programming"
    ↓
cognee_remember → entity extraction → graph nodes:
  - [User] --[prefers]--> [Rust]
  - [User] --[prefers less]--> [Go]
  - [Rust] --[used for]--> [systems programming]

Later: "What languages does the user like?"
    ↓
cognee_recall → semantic search + graph completion
    → "Prefers Rust for systems programming"
```

## Requirements

- `pip install cognee>=1.0.9`
- Any OpenAI-compatible LLM (DeepSeek, Gemini, OpenAI, etc.)
- An embedding model (default: Gemini `gemini-embedding-001`)

## Setup

```bash
hermes memory setup         # select "cognee"
```

Or manually:
```bash
hermes config set memory.provider cognee
echo "LLM_API_KEY=sk-..." >> ~/.hermes/.env          # DeepSeek or OpenAI
echo "GEMINI_API_KEY=..." >> ~/.hermes/.env           # for embeddings
```

## Configuration

Env vars (set automatically by `apply_to_environment()`):

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_API_KEY` | Yes | API key for the LLM backend |
| `LLM_ENDPOINT` | No | Custom API endpoint (default: DeepSeek) |
| `LLM_MODEL` | No | Model name (default: `deepseek/deepseek-chat`) |
| `GEMINI_API_KEY` | Yes | API key for Gemini embeddings |
| `COGNEE_EMBEDDING_MODEL` | No | Embedding model (default: `gemini/gemini-embedding-001`) |
| `COGNEE_EMBEDDING_DIMENSION` | No | Embedding dimension (default: `768`) |
| `COGNEE_GRAPH_DATABASE_PROVIDER` | No | Graph DB (default: `networkx`) |
| `COGNEE_SKIP_CONNECTION_TEST` | No | Skip live test on init (`true`/`false`) |

## Tools

Three explicit tools exposed to the agent:

| Tool | Description |
|------|-------------|
| `cognee_remember` | Store a durable fact — builds knowledge graph |
| `cognee_recall` | Semantic + graph-completion search |
| `cognee_forget` | Delete memories (requires `confirm=true`) |

## Background Flows

In addition to explicit tools, Cognee runs automatic flows:

| Flow | When | What |
|------|------|------|
| `queue_prefetch` | Before each turn | Async recall (top 5, 8s timeout) → injected as `<memory-context>` |
| `sync_turn` | After each response | Async conversation save with self-improvement |
| `on_session_end` | Session end | Persist last 40 messages |

## Tips

- Cognee works **alongside** any other memory provider — disable it via `memory.enabled: false` in provider config.
- For production: set `COGNEE_GRAPH_DATABASE_PROVIDER=neo4j` for persistent graph storage beyond NetworkX.
- The first call is slower (graph initialization); subsequent calls are fast.
- Memory context is injected into the user message (not system prompt) to preserve prompt caching.
