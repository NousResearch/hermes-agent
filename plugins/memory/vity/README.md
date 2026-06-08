# Vity Memory Provider

**Vity by Maximem AI** — cross-session semantic memory for Hermes. Vity gives
the agent a persistent memory graph (facts, preferences, emotions, episodes,
knowledge, profile) with profile-based recall and low-latency context injection
over the Maximem REST API.

> Powered by the [`maximem-vity-sdk`](../../../maximem-vity-sdk) Python client.

## Requirements

- `pip install maximem-vity-sdk` (installed automatically via `plugin.yaml`)
- A Maximem API key from [maximem.ai/dashboard](https://maximem.ai/dashboard) (starts with `mx_`)

## Setup

```bash
hermes memory setup vity   # configure Vity directly (works on a fresh install)
hermes memory setup        # generic picker, choose Vity from the list
```

Or manually:

```bash
hermes config set memory.provider vity
echo "MAXIMEM_API_KEY=mx_..." >> ~/.hermes/.env
```

The provider also reads non-secret overrides from `$HERMES_HOME/vity.json`.

## Configuration

| Key | Env var | Required | Description |
|---|---|---|---|
| `api_key` | `MAXIMEM_API_KEY` | ✅ | Maximem API key (`mx_...`). `VITY_API_KEY` is accepted for backward compatibility. |

**The API key owns the memory space.** Vity does *not* derive memory identity
from gateway users, sessions, or channels — use a separate API key for each
user or account that needs isolated memories.

## How it works

Vity injects memory at two points and writes back after every turn:

- **Warm-up recall** — on session start, `initialize()` kicks off a background
  recall with a broad profile query so the first user message already has
  context. The first turn waits briefly (cold start) for this to land, with a
  blocking fallback if it's slow.
- **Per-turn prefetch** — before each turn, recalled context is injected as a
  `## Vity Memory` block via the purpose-built `recall` endpoint.
- **Per-turn capture** — after each turn, the user/assistant exchange is
  captured into long-term memory in the background (non-blocking).

All recall/capture work happens on daemon threads so the agent loop never
blocks on the network.

## Tools exposed to the agent

| Tool | Parameters | Purpose |
|---|---|---|
| `vity_recall` | `query` (required), `top_k` (default 10, max 50) | Semantic search of the memory graph for relevant context. |
| `vity_profile` | — | Retrieve the user's full stored memory profile. |
| `vity_store` | `content` (required), `memory_type` (`fact` \| `preference` \| `emotion` \| `episode` \| `knowledge` \| `profile`) | Save a new memory fact. |
| `vity_forget` | `query`, `dry_run` (default `true`) | Delete matching memories. Always previews first unless `dry_run=false`. |

The agent is instructed to call `vity_recall` before claiming ignorance about
anything personal, to use `vity_store` for new facts the user shares, and to use
`vity_forget` only when the user explicitly asks to delete a memory.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | `VityMemoryProvider` implementation + plugin `register()` entry point. |
| `plugin.yaml` | Plugin manifest (name, version, pip dependency). |
| `README.md` | This document. |

## Testing

```bash
pytest tests/plugins/memory/test_vity_provider.py
```
