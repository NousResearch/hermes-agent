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

The setup can link to an existing `~/.openviking/ovcli.conf`, copy its current
connection values into Hermes, or create a minimal `ovcli.conf` when one does
not exist.

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
| `OPENVIKING_API_KEY` | (none) | User/admin API key for authenticated servers |
| `OPENVIKING_ACCOUNT` | `default` | Tenant account for local/trusted mode |
| `OPENVIKING_USER` | `default` | Tenant user for local/trusted mode |
| `OPENVIKING_AGENT` | `hermes` | Hermes peer ID in OpenViking, used for peer-scoped memories |

When `OPENVIKING_API_KEY` is set, Hermes lets OpenViking derive account/user
identity from the key. In local or trusted deployments without an API key,
Hermes sends `OPENVIKING_ACCOUNT` and `OPENVIKING_USER` as identity headers.

## Tools

| Tool | Description |
|------|-------------|
| `viking_search` | Semantic search with fast/deep/auto modes |
| `viking_read` | Read content at a viking:// URI (abstract/overview/full) |
| `viking_browse` | Filesystem-style navigation (list/tree/stat) |
| `viking_remember` | Store a fact for extraction on session commit |
| `viking_add_resource` | Ingest URLs/docs into the knowledge base |
