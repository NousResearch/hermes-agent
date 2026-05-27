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
- **Mirror** (`on_memory_write` hook): echoes Hermes's built-in memory writes
  into Atlas so the RDF store stays in sync with the flat `memory.md`.

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
