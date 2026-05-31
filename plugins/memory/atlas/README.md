# Atlas Memory Provider

RDF-grounded, ontology-aware long-term memory for Hermes, backed by
[Atlas](https://github.com/blakeaber/army-of-one) — Blake's unified personal
knowledge substrate.

## What it does

- **Recall** (`atlas_recall`, `prefetch`): pulls active facts about Blake from
  Atlas's `/v1/memory/hermes/read` — preferences, people, projects, decisions
  drawn from Gmail, Calendar, Pipedrive, GitHub, and Claude Code transcripts.
- **Remember** (`atlas_remember`): stores durable facts verbatim with
  provenance + confidence via `/v1/memory/hermes/write`.
- **Ask** (`atlas_ask`): routes a recall question through Atlas's full
  `/v1/ask` retrieval pipeline (BM25 + vector + Cohere rerank + Sonnet
  synthesis) and returns a cited answer. Use for "what's my last activity for
  Apex?", "what did I commit to Greg?", "when did I last email the Lambridge
  team?". The Atlas response is returned **verbatim** so `[cite:<chunk_id>]`
  markers survive intact for audit. Steered by tool description toward recall
  questions — not arbitrary "what is X" world-knowledge questions (D4 §Risk 5).
- **Mirror** (`on_memory_write` hook): echoes Hermes's built-in memory writes
  into Atlas so the RDF store stays in sync with the flat `memory.md`.

### `atlas_ask` example

```jsonc
// tool call from the model
{
  "tool": "atlas_ask",
  "args": {
    "question": "what's my last Pipedrive activity for Apex Capital?",
    "intent_hint": "lookup",
    "life_context": "work",
    "max_chunks": 5
  }
}

// returned verbatim from Atlas /v1/ask
{
  "question": "what's my last Pipedrive activity for Apex Capital?",
  "intent": "lookup",
  "answer": "Last contact was a call with Greg on 2026-05-28 [cite:chunk-abc123].",
  "citations": [
    {"chunk_id": "chunk-abc123", "source_iri": "urn:pipedrive:activity:42",
     "snippet": "Call with Greg re: pipeline review."}
  ],
  "anchors": ["urn:atlas:contact:greg"],
  "temporal": null,
  "confidence": 0.84,
  "latency_ms": 412.0,
  "usd": 0.0021
}
```

Atlas's `/v1/ask` AskRequest accepts `{question, intent_hint?}` with
`extra="forbid"` (see `army-of-one/backend/src/atlas/api/ask_routes.py`); the
plugin folds optional `life_context` and `max_chunks` hints into the
`intent_hint` string so the strict server-side Pydantic model still accepts
the payload.

## Augments, does not replace

Hermes runs the built-in memory provider **plus** one external provider. Atlas
is the external one — it layers RDF-grounded cross-session recall on top of the
built-in memory, which keeps working unchanged. Atlas-side failures degrade
gracefully (circuit breaker + swallowed errors); they never block the agent.

## Configuration

```yaml
# config.yaml
memory:
  provider: atlas
```

Environment (or `$HERMES_HOME/atlas.json`):

| Key | Env var | Default | Notes |
|-----|---------|---------|-------|
| base_url | `ATLAS_BASE_URL` | `http://localhost:8000` | Cloud: `http://atlas.agentic-stack.internal:8000` |
| token | `ATLAS_BEARER_TOKEN` | — | Required for non-localhost (LAN/VPC) |
| agent_name | `ATLAS_AGENT_NAME` | `hermes` | Fact attribution |
| max_age_days | `ATLAS_MAX_AGE_DAYS` | `90` | Recall window |

## Architecture

Hermes-side adapter for army-of-one **Plan 011-C.2**. The REST contract
(`/v1/memory/hermes/{read,write}`) was defined by **Plan 012**. No new pip
deps — uses `httpx` (already a Hermes dependency).
