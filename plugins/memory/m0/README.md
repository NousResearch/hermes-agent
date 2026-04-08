# seekdb M0 (Cloud Memory)

Long-term memory via [seekdb M0](https://m0.seekdb.ai): hosted service with semantic search, capture pipeline, and CRUD. Uses the same REST API as the OpenClaw m0 plugin (`X-API-Key` Access Key).

## Requirements

- An M0 **Access Key** (`ak_...`) from [m0.seekdb.ai](https://m0.seekdb.ai) (see below or use the service UI).
- No extra pip packages (stdlib HTTP).

## Obtaining an Access Key

Create a cloud instance (no prior key required). The JSON response includes `ak`:

```bash
curl -sS -X POST "https://m0.seekdb.ai/api/instances/" \
  -H "Content-Type: application/json" \
  -d '{"name":"my-hermes-memory"}'
```

Save the key to `$HERMES_HOME/.env` as `M0_API_KEY=...` (never commit it).

## Setup

```bash
hermes memory setup    # select "m0"
```

Or manually:

```bash
hermes config set memory.provider m0
echo 'M0_API_KEY=ak_...' >> "$HERMES_HOME/.env"
```

Optional: set `M0_BASE_URL` for a self-hosted endpoint (default `https://m0.seekdb.ai`).

## Config

| Source                 | Purpose                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| `$HERMES_HOME/.env`    | `M0_API_KEY`; optional `M0_BASE_URL`                                                       |
| `$HERMES_HOME/m0.json` | `auto_recall`, `auto_capture`, `recall_limit`, `api_timeout`, `search_rewrite`, `base_url` |

### m0.json example

```json
{
  "auto_recall": true,
  "auto_capture": true,
  "recall_limit": 10,
  "api_timeout": 5.0,
  "search_rewrite": false
}
```

## Tools

| Tool        | Maps to API                 |
| ----------- | --------------------------- |
| `m0_search` | `POST /api/memories/search` |
| `m0_store`  | `POST /api/memories/`       |
| `m0_list`   | `GET /api/memories/`        |
| `m0_get`    | `GET /api/memories/{id}`    |
| `m0_update` | `PUT /api/memories/{id}`    |
| `m0_delete` | `DELETE /api/memories/{id}` |

Turn sync uses `POST /api/memories/capture` with the user/assistant message pair (server-side extraction), when `auto_capture` is true.

## Local integration tests (real API)

Default `pytest` for this plugin uses mocks only. To hit the live API locally:

```bash
export M0_INTEGRATION_TESTS=1
export M0_API_KEY=ak_...
pytest tests/plugins/memory/test_m0_provider.py::TestM0LiveApiOptional -v -o addopts=
```

CI should **not** set `M0_INTEGRATION_TESTS`, so these cases stay skipped.

## See also

- [M0 SKILL / API overview](https://m0.seekdb.ai/SKILL.md)
- Hermes: [Memory Providers](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers) (user guide)

**Note:** M0 is a **hosted HTTP** backend. For a local [PowerMem](https://github.com/oceanbase/powermem) Python stack (SQLite / SeekDB vector store), use the `powermem` memory provider instead — only one external `memory.provider` can be active at a time.
