# DeepParser SDK + API Server v1.0.0

## What's new

This is the first stable release of two packages:

- **`deepparser`** — Python SDK for the DeepParser document parsing API
- **`deepparser-api`** — Self-hostable FastAPI server that wraps the `dp` CLI

---

## Python SDK (`deepparser`)

Install with pip:

```bash
pip install deepparser
```

### Highlights

- **Parse and ask in one call** — `client.parse_and_ask(file, question)` uploads, waits for READY, then returns the answer with citations.
- **Parse once, ask many** — call `client.parse()` then `client.ask(job_id, ...)` repeatedly against the same parsed document.
- **Structured citations** — every `AskResult` carries a list of `Citation` objects with `filename`, `page`, and `cell` fields for tracing answers back to source.
- **Async-native** — built on `httpx.AsyncClient`; use as an async context manager.
- **DWG / DXF support** — CAD drawings parse natively; no CAD software or OCR required.
- **Excel-embedded PDFs** — reconstructs column/row structure instead of flattening to pixel noise.
- **Exponential back-off polling** — `wait_until_ready()` starts at 2 s, caps at 10 s, gives up after 5 min.

### Quick start

```python
import asyncio
from deepparser import DeepParserClient

async def main():
    async with DeepParserClient(api_key="dp_live_...") as client:
        result = await client.parse_and_ask("contract.pdf", "What are the payment terms?")
        print(result.answer)
        for c in result.citations:
            print(f"  {c.filename} p.{c.page}")

asyncio.run(main())
```

### Example scripts

See [`deepparser/examples/`](deepparser/examples/) for five ready-to-run scripts:

| Script | What it shows |
|---|---|
| `basic_parse.py` | Upload any file and ask a question |
| `excel_embedded.py` | Parse once, ask multiple questions |
| `dwg_query.py` | Query a CAD drawing for rooms and dimensions |
| `batch_upload.py` | Process a folder concurrently with `asyncio.Semaphore` |
| `citations.py` | Display answer with page/cell source references |

### API reference

| Method | Description |
|---|---|
| `parse(file, *, sync=False)` | Submit a parse job; returns `ParseJob` |
| `get_status(job_id)` | Poll job status |
| `wait_until_ready(job_id)` | Block until READY or raise `ParseTimeoutError` |
| `ask(job_id, question)` | Ask a question against a parsed document |
| `parse_and_ask(file, question)` | Convenience: parse + wait + ask in one call |
| `demo()` | Downloads a sample contract and runs a demo question |
| `register_key(email, ...)` | Request an API key |

Exceptions: `AuthError`, `RateLimitError`, `ParseFailedError`, `ParseTimeoutError`, `JobNotFoundError`.

---

## API Server (`deepparser-api`)

Self-host the full parsing backend:

```bash
pip install "deepparser[server]"
deepparser-server
```

Or with Docker:

```bash
docker pull ysh145/deepparser-api:1.0.0
docker run -e DEEPPARSER_API_KEY=dp_live_... -p 8000:8000 ysh145/deepparser-api:1.0.0
```

### Server highlights

- **Key registration** — `POST /keys/register` issues `dp_live_*` API keys tied to an email address
- **Rate limiting** — 5 failed auth attempts per 15 min per IP; 3 key registrations per hour per IP
- **Async parse pipeline** — `asyncio.Semaphore(4)` caps concurrent `dp` CLI calls; `asyncio.shield` prevents task cancellation on sync timeout
- **Admin endpoints** — `GET /admin/keys` and `POST /admin/keys/{key_id}/activate` behind a separate `ADMIN_PASSWORD`
- **SQLite persistence** — jobs and keys stored in `/data/deepparser.db`; Fly.io `max_machines_running=1` preserves single-writer guarantee
- **Structured JSON logging** — all requests logged with job_id, duration_ms, status

### Fly.io deploy

```bash
fly launch --no-deploy
fly secrets set DEEPPARSER_API_KEY=dp_live_... ADMIN_PASSWORD=$(openssl rand -hex 24)
fly deploy
```

---

## Benchmark results

Evaluated against LlamaIndex (default OpenAI settings) on 50 QA pairs across:
- Excel-exported PDFs with table structure
- DWG/DXF CAD drawings
- Scanned PDFs

| Category | Win rate |
|---|---|
| Excel-embedded PDFs | — |
| CAD drawings (DWG/DXF) | — |
| Scanned PDFs | — |
| **Overall** | — |

*Run `python benchmark/runner.py` then `python benchmark/score.py` to reproduce.*

---

## Breaking changes

First stable release — no prior versions to break.

---

## What's next

- Streaming answer support (`text/event-stream`)
- Multi-document ask (ask across a corpus of parsed jobs)
- Webhook notifications when async parse completes
- `dp` CLI version pinning in Docker image

---

## Contributors

Sean Yang (@ysh145)
