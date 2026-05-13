# HN Post Draft

**Title (use after DWG hard gate passes — see TODOS.md):**
Show HN: DeepParser – parse DWG/CAD drawings and Excel-embedded PDFs without OCR

**Title (use for initial launch if DWG gate not yet confirmed):**
Show HN: DeepParser – parse Excel-embedded PDFs and scanned documents via a REST API + Python SDK

---

**Body:**

I've been building data pipelines that pull numbers out of engineering drawings and budget spreadsheets exported to PDF. Both cases are unsolved by standard PDF parsers:

- **DWG/DXF** — CAD drawings aren't text documents at all. AutoCAD exports vector geometry and annotation entities; PDF renderers flatten everything to pixels and OCR produces garbage.
- **Excel-exported PDFs** — the source data has row/column structure but standard parsers return a stream of tokens with no table boundaries.

DeepParser is a REST API (and Python SDK) that sends documents to a specialized backend that understands these formats natively.

**What's shipping today:**

1. **Python SDK** (`pip install deepparser`) — async, httpx-based, structured citations with page/cell references
2. **Self-hostable API server** (`pip install "deepparser[server]"` or Docker) — FastAPI, SQLite, Fly.io deploy config included
3. **Five example scripts** covering the common patterns: batch upload, citation display, multi-question on a single parse, DWG query

**Quick demo:**

```python
from deepparser import DeepParserClient
import asyncio

async def main():
    async with DeepParserClient(api_key="dp_live_...") as client:
        result = await client.parse_and_ask(
            "floor_plan.dwg",
            "List all rooms with their areas."
        )
        print(result.answer)

asyncio.run(main())
```

**Self-hosting:** The server wraps the `dp` CLI subprocess. You bring your own `DEEPPARSER_API_KEY` (from beta.deepparser.ai) and the server handles job queuing, API key management, rate limiting, and structured logging. One-command Fly.io deploy is in the repo.

**Benchmark:** I'm running a 50-pair QA benchmark comparing against LlamaIndex (default settings) across Excel PDFs, CAD drawings, and scanned PDFs. Results will be posted in the repo once scoring is complete.

**Repo:** https://github.com/ysh145/hermes-agent/tree/main/deepparser

Feedback welcome — especially from anyone doing document extraction on engineering or financial documents.

---

*Tags to consider: python, pdf, cad, document-parsing, api*
