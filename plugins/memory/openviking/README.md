# OpenViking Memory Provider

Context database by Volcengine (ByteDance) with filesystem-style knowledge hierarchy, tiered retrieval, and automatic memory extraction.

## Requirements

- `httpx` installed in the Hermes environment
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
| `OPENVIKING_API_KEY` | (none) | API key (optional) |
| `OPENVIKING_ACCOUNT` | (none) | Tenant account override |
| `OPENVIKING_USER` | (none) | Tenant user override |
| `OPENVIKING_AGENT` | `hermes` | Tenant agent namespace |

## Recall

Before each model turn, Hermes asks OpenViking for relevant memory context.
The plugin overfetches candidates, deduplicates repeated memories, reranks for
query overlap, and injects a bounded evidence block. This keeps prompt context
focused while still leaving OpenViking tools available for targeted follow-up.

## Writes

After each completed turn, Hermes mirrors the user message and assistant reply
into the OpenViking session. Hermes native memory `add` and `replace` writes are
also mirrored as explicit memory notes, then OpenViking extracts durable memory
when the session commits.

## Tools

| Tool | Description |
|------|-------------|
| `viking_search` | Search durable memory and indexed resources |
| `viking_read` | Read one URI or up to three URIs (abstract/overview/full) |
| `viking_browse` | Diagnostic URI navigation (list/tree/stat), with capped output |
| `viking_remember` | Store a fact for extraction on session commit |
| `viking_add_resource` | Ingest URLs/docs into the knowledge base |
