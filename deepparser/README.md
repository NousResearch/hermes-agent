# DeepParser Python SDK

Parse PDFs, Excel-embedded tables, scanned documents, DWG drawings, and more — without OCR soup.

```
pip install deepparser
```

---

## Quick start

### Get an API key

```python
import asyncio
from deepparser import DeepParserClient

async def main():
    info = await DeepParserClient.register_key(
        "you@company.com",
        intended_use="parsing invoices",
        base_url="https://your-deepparser-server.fly.dev",
    )
    print(info.api_key)  # dp_live_...

asyncio.run(main())
```

---

## Example 1 — Magic demo (2 lines, ~2 min to first result)

Runs against a hosted sample contract. Needs no file of your own.

```python
import asyncio
from deepparser import DeepParserClient

async def main():
    async with DeepParserClient(
        api_key="dp_live_...",
        base_url="https://your-deepparser-server.fly.dev",
    ) as client:
        result = await client.demo()
        print(result.answer)
        for c in result.citations:
            print(f"  cited: {c.filename}, page {c.page}")

asyncio.run(main())
```

---

## Example 2 — Parse a PDF and ask a question

`parse_and_ask()` handles upload → polling → ask in one call.

```python
import asyncio
from deepparser import DeepParserClient

async def main():
    async with DeepParserClient(
        api_key="dp_live_...",
        base_url="https://your-deepparser-server.fly.dev",
    ) as client:
        result = await client.parse_and_ask(
            "invoice.pdf",
            "What is the total amount due and the payment deadline?",
        )
        print(result.answer)
        for c in result.citations:
            print(f"  → page {c.page}: {c.filename}")

asyncio.run(main())
```

---

## Example 3 — Excel-embedded PDF

Works exactly like any other file — DeepParser reconstructs the table
structure that PDF flattens into pixel noise.

```python
import asyncio
from deepparser import DeepParserClient

async def main():
    async with DeepParserClient(
        api_key="dp_live_...",
        base_url="https://your-deepparser-server.fly.dev",
    ) as client:
        result = await client.parse_and_ask(
            "budget_export.pdf",   # originally an Excel workbook, saved as PDF
            "What was Q3 total expenses broken down by department?",
        )
        print(result.answer)

asyncio.run(main())
```

---

## Example 4 — DWG drawing (structural / MEP engineering)

DeepParser reads `.dwg` and `.dxf` files natively. Ask about dimensions,
schedules, and annotations without any CAD software.

```python
import asyncio
from deepparser import DeepParserClient

async def main():
    async with DeepParserClient(
        api_key="dp_live_...",
        base_url="https://your-deepparser-server.fly.dev",
    ) as client:
        result = await client.parse_and_ask(
            "floor_plan.dwg",
            "List all room names and their floor areas in square meters.",
        )
        print(result.answer)

asyncio.run(main())
```

---

## Example 5 — Raw async: upload → poll → ask

Use this pattern when you need the job ID for your own database, want to
fan out multiple questions, or are integrating into an existing async
pipeline.

```python
import asyncio
from deepparser import DeepParserClient, ParseFailedError, ParseTimeoutError

async def main():
    async with DeepParserClient(
        api_key="dp_live_...",
        base_url="https://your-deepparser-server.fly.dev",
        debug=True,  # logs HTTP calls to stderr
    ) as client:

        # 1. Upload — returns immediately with job_id
        job = await client.parse("contract.pdf")
        print(f"queued: {job.job_id}")

        # 2. Poll until READY (exponential back-off, raises on failure)
        try:
            job = await client.wait_until_ready(job.job_id)
        except ParseFailedError as e:
            print(f"parse failed: {e.detail}")
            return
        except ParseTimeoutError:
            print("timed out — try splitting the document")
            return

        # 3. Ask multiple questions against the same parsed doc
        q1 = await client.ask(job.job_id, "What are the parties to this agreement?")
        q2 = await client.ask(job.job_id, "What is the governing law clause?")

        print("parties:", q1.answer)
        print("governing law:", q2.answer)

asyncio.run(main())
```

---

## API reference

### `DeepParserClient(api_key, base_url, *, timeout=300.0, debug=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | `str` | Your `dp_live_...` key from `register_key()` |
| `base_url` | `str` | URL of your deployed DeepParser server |
| `timeout` | `float` | Total HTTP timeout in seconds |
| `debug` | `bool` | Print `METHOD URL → status (ms)` to stderr |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `demo()` | `AskResult` | Parse a hosted sample PDF and return a citation |
| `parse_and_ask(file, question)` | `AskResult` | One-call convenience: upload → wait → ask |
| `parse(file, *, filename, sync)` | `ParseJob` | Upload only; returns `job_id` immediately |
| `get_status(job_id)` | `ParseJob` | Poll job status |
| `wait_until_ready(job_id)` | `ParseJob` | Block until READY, raise on failure |
| `ask(job_id, question)` | `AskResult` | Ask against a READY job |
| `register_key(email, ...)` *(static)* | `KeyInfo` | Create a new API key |

### Response models

**`AskResult`**
```
answer: str
citations: list[Citation]
```

**`Citation`**
```
filename: str
page: int | None
cell: str | None      # for spreadsheet cells, e.g. "B12"
```

**`ParseJob`**
```
job_id: str
status: QUEUED | PARSING | READY | PARSE_FAILED | TIMEOUT
result: ParseResult | None
error_detail: str | None
```

### Exceptions

| Exception | When |
|-----------|------|
| `AuthError` | Key missing, invalid, or revoked (HTTP 401/403) |
| `RateLimitError` | Too many requests from this IP (HTTP 429) |
| `ParseFailedError` | dp_cli subprocess failed; `.detail` has the reason |
| `ParseTimeoutError` | Parse exceeded 120 s; try splitting the document |
| `JobNotFoundError` | Job ID not found or belongs to another key |

---

## Supported file types

`.pdf` `.docx` `.doc` `.ppt` `.pptx` `.xls` `.xlsx` `.csv` `.txt` `.md`
`.jpg` `.jpeg` `.png` `.dwg` `.dxf`

Maximum file size: **50 MB**

---

## Self-host the server

```bash
pip install "deepparser[server]"
export DPCLI_FOLDER_ID=your_deepparser_folder_id
export ADMIN_PASSWORD=choose_a_strong_password
deepparser-server          # listens on :8000
```

Or with Docker:

```bash
docker run -p 8000:8000 \
  -e DPCLI_FOLDER_ID=... \
  -e ADMIN_PASSWORD=... \
  ghcr.io/ysh145/deepparser:latest
```
