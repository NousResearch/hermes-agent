# ByteRover Memory Provider

Persistent memory via the `brv` CLI — hierarchical knowledge tree with tiered retrieval (fuzzy text → LLM-driven search).

## Requirements

Install the ByteRover CLI:
```bash
curl -fsSL https://byterover.dev/install.sh | sh
# or
npm install -g byterover-cli
```

## Setup

```bash
hermes memory setup    # select "byterover"
```

Or manually:
```bash
hermes config set memory.provider byterover
# Optional cloud sync:
echo "BRV_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

`BRV_API_KEY` is optional and only needed for cloud sync. ByteRover is local-first by default.

Automatic ByteRover prefetch uses an effective deadline of `min(memory.prefetch_timeout, memory.byterover.timeout_query)`. The defaults are 8 seconds for `memory.prefetch_timeout` and 10 seconds for `memory.byterover.timeout_query`, so automatic prefetch waits for at most 8 seconds. Direct `brv_query` calls use `memory.byterover.timeout_query` only, which defaults to 10 seconds.

```bash
hermes config set memory.byterover.timeout_query 30
hermes config set memory.prefetch_timeout 31
```

Both values are in seconds and accept numbers from 0.01 to 3600.

Working directory: `$HERMES_HOME/byterover/` (profile-scoped).

## Tools

| Tool | Description |
|------|-------------|
| `brv_query` | Search the knowledge tree |
| `brv_curate` | Store facts, decisions, patterns |
| `brv_status` | CLI version, tree stats, sync state |
