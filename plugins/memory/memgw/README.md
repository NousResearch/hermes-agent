# memgw — Memory Gateway provider

Connects Hermes to the self-hosted **Memory Gateway** (Neo4j + Qdrant + Notion)
over its Streamable-HTTP MCP endpoint. Unlike sealed memory backends, the
gateway fuses **semantic + keyword + graph** retrieval and grounds answers in a
knowledge graph and the Obsidian vault.

## Why this over a generic memory backend

| Capability | Generic vector memory | memgw (Memory Gateway) |
|---|---|---|
| Semantic recall | ✅ | ✅ |
| Exact-term / proper-noun recall | ⚠️ blurred by embeddings | ✅ Neo4j BM25 full-text |
| Multi-hop graph context | ❌ | ✅ `RELATES_TO`/`ABOUT` walk |
| Reflection / mental models | sometimes | ✅ `reflect` → durable beliefs |
| Failure/experience learning | ❌ | ✅ `on_delegation` → Experience nodes |
| Knowledge-graph + vault grounding | ❌ | ✅ entities, CRM, Notion, Obsidian |

## Tools exposed to the model

- `memgw_recall` — hybrid recall (semantic + keyword + graph fusion via RRF)
- `memgw_retain` — store a durable memory
- `memgw_reflect` — synthesized beliefs (mental models) on a topic

## Auto behaviour

- **prefetch** — background `recall` (or `reflect`) injected before each turn
- **sync_turn** — store completed turns (non-blocking, single-writer)
- **on_delegation** — record a subagent task+result as an `experience`
- **on_session_end** — store a lightweight session summary

## Setup

```bash
hermes memory setup        # pick "memgw"
hermes memory status       # verify active
```

Or manually:

```bash
export MEMGW_API_URL="https://mcp.danizhaky.com/mcp"   # or http://localhost:8081/mcp
export MEMGW_API_KEY="<gateway bearer token>"          # required for cloud mode
```

Config can also live in `$HERMES_HOME/memgw.json`.

### Modes

- **Cloud** (default): hosted gateway at `mcp.danizhaky.com`, Bearer-authenticated.
- **Local**: point `MEMGW_API_URL` at a `localhost` gateway — no key required.

A circuit breaker pauses calls for 120s after 5 consecutive failures so a
gateway outage never blocks the turn loop; recall degrades gracefully to empty.

## Why memgw is bundled (not under ~/.hermes/plugins/)

AGENTS.md says new memory backends should generally ship as standalone user
plugins. memgw is a **deliberate exception**: it is the owner's *default* memory
provider for this fork (dizhaky/hermes-agent), and the memory plugin system
(`plugins/memory/__init__.py`) explicitly supports bundled providers. Bundling
ships it to every machine via git with no per-machine install — the desired
behaviour for a default backend. It degrades to built-in memory when the `mcp`
dependency or `MEMGW_API_KEY` is absent, so it is inert on installs that don't
configure it. If this fork is ever upstreamed, memgw should move to
`~/.hermes/plugins/` or a pip entry-point package per the standard policy.
