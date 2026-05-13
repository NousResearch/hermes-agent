"""
citations.py — Ask a question and display citations with page/cell references.

Useful for compliance and audit workflows where you need to trace every
claim back to a specific page or spreadsheet cell.

Usage:
    export DEEPPARSER_API_KEY=dp_live_...
    python citations.py contract.pdf "What are the indemnification obligations?"
"""
import asyncio
import os
import sys

from deepparser import DeepParserClient


async def main(file_path: str, question: str) -> None:
    api_key = os.environ["DEEPPARSER_API_KEY"]
    base_url = os.environ.get("DEEPPARSER_BASE_URL", "https://deepparser-api.fly.dev")

    async with DeepParserClient(api_key=api_key, base_url=base_url) as client:
        print(f"Parsing {file_path}…")
        result = await client.parse_and_ask(file_path, question)

    print(f"\nAnswer:\n{result.answer}\n")

    if not result.citations:
        print("(no citations returned)")
        return

    print(f"Sources ({len(result.citations)} citation{'s' if len(result.citations) != 1 else ''}):")
    for i, c in enumerate(result.citations, 1):
        parts = [c.filename]
        if c.page is not None:
            parts.append(f"page {c.page}")
        if c.cell:
            parts.append(f"cell {c.cell}")
        print(f"  [{i}] {' — '.join(parts)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python citations.py <file> <question>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
