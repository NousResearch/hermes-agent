"""
basic_parse.py — Upload any PDF and ask a question.

Usage:
    export DEEPPARSER_API_KEY=dp_live_...
    export DEEPPARSER_BASE_URL=https://deepparser-api.fly.dev   # or your server
    python basic_parse.py report.pdf "What is the executive summary?"
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

        print(f"\nAnswer:\n{result.answer}")
        if result.citations:
            print("\nCited sources:")
            for c in result.citations:
                loc = f"p.{c.page}" if c.page else c.cell or "?"
                print(f"  {c.filename} — {loc}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python basic_parse.py <file> <question>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
