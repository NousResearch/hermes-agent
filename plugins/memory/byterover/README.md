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

| Env Var | Required | Description |
|---------|----------|-------------|
| `BRV_API_KEY` | No | Cloud sync key (optional, local-first by default) |

Working directory: `$HERMES_HOME/byterover/` (profile-scoped).

## Tools

| Tool | Description |
|------|-------------|
| `brv_query` | Search the knowledge tree |
| `brv_curate` | Store facts, decisions, patterns |
| `brv_status` | CLI version, tree stats, sync state |
