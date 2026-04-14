# Mem0 HTTP Memory Provider

Direct HTTP integration with the Mem0 API using Hermes' memory-provider interface.

## Why this provider exists

- avoids the heavy `mem0ai` SDK dependency
- keeps the Docker image smaller
- uses the Mem0 REST API directly

## Setup

```bash
hermes config set memory.provider mem0_http
```

Then set:

```bash
MEM0_API_KEY=your-key
MEM0_USER_ID=hermes-agent
MEM0_AGENT_ID=hermes-agent
```

Optional:

```bash
MEM0_BASE_URL=https://api.mem0.ai
```

## Config file

Config file: `$HERMES_HOME/mem0_http.json`

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | Retrieve all stored memories for the configured scope |
| `mem0_search` | Semantic search with scope filters |
| `mem0_conclude` | Store a fact verbatim (no inference) |
