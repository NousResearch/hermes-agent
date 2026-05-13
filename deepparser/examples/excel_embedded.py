"""
excel_embedded.py — Extract table data from a PDF exported from Excel.

Standard PDF parsers flatten spreadsheet PDFs into pixel noise.
DeepParser reconstructs column/row structure from the source data.

Usage:
    export DEEPPARSER_API_KEY=dp_live_...
    python excel_embedded.py budget.pdf
"""
import asyncio
import os

from deepparser import DeepParserClient

QUESTIONS = [
    "List every department and its total budget as a markdown table.",
    "Which line item had the largest year-over-year increase?",
    "What is the grand total across all departments?",
]


async def main(file_path: str) -> None:
    api_key = os.environ["DEEPPARSER_API_KEY"]
    base_url = os.environ.get("DEEPPARSER_BASE_URL", "https://deepparser-api.fly.dev")

    async with DeepParserClient(api_key=api_key, base_url=base_url) as client:
        print(f"Parsing {file_path}…")

        # Parse once, ask multiple questions against the same parsed doc
        job = await client.parse(file_path, sync=True)
        if job.status != "READY":
            job = await client.wait_until_ready(job.job_id)

        print(f"Ready (job {job.job_id[:8]}…)\n")

        for q in QUESTIONS:
            print(f"Q: {q}")
            result = await client.ask(job.job_id, q)
            print(f"A: {result.answer}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python excel_embedded.py <budget.pdf>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
