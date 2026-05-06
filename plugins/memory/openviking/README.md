# OpenViking Memory Provider

Context database by Volcengine (ByteDance) with filesystem-style knowledge hierarchy, tiered retrieval, and automatic memory extraction.

## Requirements

- `pip install openviking`
- OpenViking server running (`openviking-server`)
- Embedding + VLM model configured in `~/.openviking/ov.conf`

## Setup

```bash
hermes memory setup    # select "openviking"
```

Or manually:
```bash
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

## Config

All config via environment variables in `.env`:

| Env Var | Default | Description |
|---------|---------|-------------|
| `OPENVIKING_ENDPOINT` | `http://127.0.0.1:1933` | Server URL |
| `OPENVIKING_API_KEY` | (none) | API key (optional) |
| `OPENVIKING_ACCOUNT` | `default` | Tenant account for API requests |
| `OPENVIKING_USER` | `default` | Tenant user for API requests |
| `OPENVIKING_AGENT` | `hermes` | Tenant agent for API requests |

## Behavior notes

- File reads via `viking_read` fall back to full-content reads for file URIs because OpenViking overview/abstract endpoints are directory-oriented and can return 500 on file paths.
- Explicit Hermes memory writes also store a deterministic fallback resource under `viking://resources/hermes_explicit_memories/...` so important notes remain durable even if session extraction is degraded.

## Tools

| Tool | Description |
|------|-------------|
| `viking_search` | Semantic search with fast/deep/auto modes |
| `viking_read` | Read content at a viking:// URI (abstract/overview/full) |
| `viking_browse` | Filesystem-style navigation (list/tree/stat) |
| `viking_remember` | Store a fact for extraction on session commit |
| `viking_add_resource` | Ingest URLs/docs into the knowledge base |
