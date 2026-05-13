# YantrikDB Memory Provider

> **Self-maintaining memory for Hermes.** Canonicalizes duplicates, surfaces contradictions, ranks with recency awareness, and explains recall — instead of behaving like an append-only vector index.

> Installing standalone (until upstream review lands)? See **[../README.md](../README.md)** for the one-liner drop-in install.

## Why YantrikDB?

| Plugin                     | Best at                                               |
|----------------------------|--------------------------------------------------------|
| `mem0`                     | Managed cloud fact extraction with reranking          |
| `honcho`                   | Cross-session user modeling with dialectic Q&A        |
| `byterover` / `supermemory`| Standard semantic vector recall                        |
| `hindsight` / `holographic`| Specialized retrieval schemes                          |
| **`yantrikdb`**            | **Memory maintenance: canonicalization, contradiction tracking, recency ranking, explainable recall, knowledge-graph boosting** |

What the other memory plugins have in common: once a fact is stored, it stays as-is. Duplicate stores pile up, superseded facts silently outrank their replacements, and the agent has no window into *why* a memory came back.

YantrikDB maintains its store. `think()` runs a bounded maintenance pass that canonicalizes near-duplicates, flags contradictions, and re-scores for recency. `recall()` returns a `why_retrieved` reason list per result. `conflicts()` surfaces superseded claims explicitly so the agent resolves them via `resolve_conflict()` rather than hoping last-write-wins.

## Requirements

- A running `yantrikdb-server` (Docker, systemd, or bare binary). The plugin is a thin HTTP client; it does not start or manage the backend.
- A bearer token minted against that server.

## Setup

```bash
hermes memory setup   # select "yantrikdb"
```

Or manually:

```bash
hermes config set memory.provider yantrikdb
cat >> ~/.hermes/.env <<'EOF'
YANTRIKDB_URL=http://localhost:7438
YANTRIKDB_TOKEN=ydb_...
EOF
```

### Running the server

Docker, single node:

```bash
docker run -d -p 7438:7438 \
  -v yantrikdb-data:/var/lib/yantrikdb \
  --name yantrikdb \
  ghcr.io/yantrikos/yantrikdb:latest
```

Mint a token:

```bash
docker exec yantrikdb yantrikdb token \
  --data-dir /var/lib/yantrikdb \
  create --db default --label hermes
# → ydb_abc123...
```

See <https://yantrikdb.com/server/quickstart/> for systemd, HA cluster, and advanced deployments.

## Config

Optional config file: `$HERMES_HOME/yantrikdb.json`. Env vars are the primary source.

| Key               | Env                         | Default                 | Description                                                                        |
|-------------------|-----------------------------|-------------------------|------------------------------------------------------------------------------------|
| `url`             | `YANTRIKDB_URL`             | `http://localhost:7438` | HTTP endpoint.                                                                     |
| `token`           | `YANTRIKDB_TOKEN`           | *required*              | Bearer token from `yantrikdb token create`.                                        |
| `namespace`       | `YANTRIKDB_NAMESPACE`       | `hermes`                | Tenant prefix. Combined with `agent_workspace:agent_identity` at init time.        |
| `top_k`           | `YANTRIKDB_TOP_K`           | `10`                    | Default recall result count (capped at 50 via tool param).                         |
| `connect_timeout` | `YANTRIKDB_CONNECT_TIMEOUT` | `5.0`                   | TCP connect timeout (seconds).                                                     |
| `read_timeout`    | `YANTRIKDB_READ_TIMEOUT`    | `15.0`                  | Per-request read timeout.                                                          |
| `retry_total`     | `YANTRIKDB_RETRY_TOTAL`     | `3`                     | Bounded retries on transient 5xx / connection blips (exponential backoff).         |
| `max_text_len`    | `YANTRIKDB_MAX_TEXT_LEN`    | `25000`                 | Hard cap on memory body length; longer text is truncated client-side with a marker.|

## Tools

| Tool                         | Purpose                                                                                  |
|------------------------------|------------------------------------------------------------------------------------------|
| `yantrikdb_remember`         | Store a memory with importance and optional domain.                                      |
| `yantrikdb_recall`           | Explainable recall — each result includes a `why_retrieved` reason list.                 |
| `yantrikdb_forget`           | Tombstone a memory by rid.                                                               |
| `yantrikdb_think`            | Run the bounded maintenance pass. **The differentiator.**                                |
| `yantrikdb_conflicts`        | List open contradictions detected by `think()`.                                          |
| `yantrikdb_resolve_conflict` | Close a contradiction via `keep_winner` / `merge` / `keep_both` / `dismiss`.             |
| `yantrikdb_relate`           | Record a knowledge-graph edge (e.g. *Alice works_at Acme*).                              |
| `yantrikdb_stats`            | Operational snapshot: memory counts, open conflicts, pending triggers.                   |

### Explainable recall

`yantrikdb_recall` returns each result with a reason list so the agent can audit ranking:

```json
{
  "rid": "019d…",
  "text": "User lives in Seattle since 2024",
  "score": 1.1576,
  "importance": 0.9,
  "why_retrieved": [
    "semantic_match",
    "important (decay=0.91)",
    "graph-connected via User"
  ]
}
```

This lets the agent decide whether to trust a match (semantic + graph + importance) versus a weak match (keyword only), and lets you debug why stale memories are outranking fresh ones.

### The `think()` maintenance pass

Call this at natural break points (end of a phase, long user pause, before a `/compact`). One call:

1. **Canonicalizes** near-duplicates — 20 variations of *"user prefers dark mode"* are linked to 1–2 canonical memories. Originals are kept (not deleted) and recall favors canonical forms.
2. **Flags contradictions** — *"CEO is Alice"* stored earlier and *"CEO is Bob"* stored later surface via `yantrikdb_conflicts`. Nothing is silently overwritten.
3. **Optionally mines patterns** — temporal clusters, entity co-occurrences. Off by default (expensive).

The response is structured:

```json
{
  "consolidated": 3,
  "conflicts_found": 1,
  "patterns_new": 0,
  "patterns_updated": 0,
  "duration_ms": 42,
  "triggers": [
    {"trigger_type": "review_conflicts", "urgency": "medium", "source_rids": ["019d…"]}
  ]
}
```

### Conflict-aware memory

`yantrikdb_conflicts` surfaces what `think()` detected:

```json
{"count": 1, "conflicts": [
  {"conflict_id": "cf_1",
   "memory_a": {"rid": "r1", "text": "CEO is Alice"},
   "memory_b": {"rid": "r2", "text": "CEO is Bob"},
   "detection_reason": "entity_collision: CEO",
   "priority": "high"}
]}
```

The agent resolves explicitly:

```python
yantrikdb_resolve_conflict(
    conflict_id="cf_1",
    strategy="keep_winner",
    winner_rid="r2",
    resolution_note="Bob succeeded Alice in March 2026",
)
```

Strategies:

- `keep_winner` — preserve `winner_rid`, tombstone the other.
- `merge` — emit a consolidated `new_text`, tombstone both.
- `keep_both` — both remain (context-dependent truth).
- `dismiss` — close without action (false positive).

## Hooks

| Hook              | Effect                                                                                          |
|-------------------|-------------------------------------------------------------------------------------------------|
| `on_session_end`  | Calls `/v1/think` to consolidate the session before exit.                                       |
| `on_pre_compress` | Seeds recall with the about-to-be-compressed tail so the Hermes compressor preserves insights. |
| `on_memory_write` | Mirrors built-in `MEMORY.md` / `USER.md` additions into YantrikDB.                              |

Hooks are best-effort: they never block the main Hermes loop and failures are logged at DEBUG without propagating.

Assistant-message extraction is intentionally out of v1 scope — storing LLM output as fact is a hallucination-amplification risk. Only user messages are persisted via `sync_turn`.

## Use cases

- **Long-running research agents** where noise accumulates — `think()` periodically canonicalizes.
- **Debate / planning agents** that encounter shifting facts — contradictions get surfaced, not papered over.
- **User-modeling agents** where preferences evolve — recency ranking plus explicit `resolve_conflict` keeps the picture current.
- **Multi-session projects** where the same entities recur — the knowledge graph boosts cross-session recall.

## Resilience

- Bounded HTTP retries (default 3, configurable) with exponential backoff on transient 5xx and connection errors.
- Circuit breaker: after 5 consecutive transient/server/auth failures the plugin short-circuits for 120 s so a flapping server cannot hammer Hermes' event loop. 4xx errors (deterministic caller mistakes) do not trip the breaker.
- All background writes (sync_turn, prefetch, on_memory_write mirror) run on daemon threads with bounded join timeouts; shutdown flushes pending work.
- Every HTTP call is tagged with a short `req_id` and logged at DEBUG with operation and latency for post-hoc correlation in the Hermes log stream.

## Troubleshooting

**`Connection refused`** — server isn't running. Verify with `curl http://localhost:7438/v1/health`.

**`401 / invalid token`** — mint a fresh token against the *running* server, not a previously-stopped instance. See the [quickstart](https://yantrikdb.com/server/quickstart/).

**Recall returns nothing across sessions** — the effective namespace depends on `agent_workspace` + `agent_identity`. Changing profile names changes the namespace. Check startup logs for `YantrikDB connected: ... (namespace: ...)`.

**Circuit breaker open** — the plugin stopped trying after repeated failures. Check the server; the breaker auto-resets after 120 s.

**Text unexpectedly truncated** — memory bodies over `YANTRIKDB_MAX_TEXT_LEN` (default 25000 chars) are clipped client-side with a visible `…[truncated]` marker. Raise the env var if your use case needs longer bodies, but note that the server may enforce its own limits.

## Architecture notes

The plugin is ~700 lines split into `__init__.py` (provider + 8 tool schemas + 3 optional hooks) and `client.py` (HTTP wrapper, typed errors, config). The only pip dependency is `requests`. The Rust backend is an external dependency managed by the user — same model as the `honcho` plugin.

See [ARCHITECTURE.md](ARCHITECTURE.md) for control flow, error taxonomy, and threading model. [CHANGELOG.md](CHANGELOG.md) tracks versioned changes; [SECURITY.md](SECURITY.md) documents the token/secret handling model.

### License

YantrikDB-server is AGPL-3.0. This plugin ships under Hermes' MIT license. The plugin connects to the server over HTTP and does not statically link or embed any YantrikDB code, so the model is the same as any MIT client that talks to an AGPL server — legally clean.

## Links

- Server repo: <https://github.com/yantrikos/yantrikdb-server>
- Docs: <https://yantrikdb.com>
- HTTP API reference: <https://yantrikdb.com/server/http-api/>
- Issues: <https://github.com/yantrikos/yantrikdb-server/issues>
