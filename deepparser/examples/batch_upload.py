"""
batch_upload.py — Parse a folder of documents concurrently, then ask each a question.

Parses up to 4 files in parallel (matches the server's default Semaphore(4)).

Usage:
    export DEEPPARSER_API_KEY=dp_live_...
    python batch_upload.py ./invoices/ "What is the total amount due?"
"""
import asyncio
import os
import sys
from pathlib import Path

from deepparser import DeepParserClient, ParseFailedError, ParseTimeoutError

SUPPORTED = {".pdf", ".docx", ".xlsx", ".dwg", ".dxf", ".png", ".jpg"}
MAX_CONCURRENT = 4


async def process_one(
    client: DeepParserClient,
    sem: asyncio.Semaphore,
    file_path: Path,
    question: str,
) -> tuple[str, str | None, str | None]:
    async with sem:
        try:
            result = await client.parse_and_ask(file_path, question)
            return (file_path.name, result.answer, None)
        except ParseFailedError as e:
            return (file_path.name, None, f"parse failed: {e.detail}")
        except ParseTimeoutError:
            return (file_path.name, None, "timed out")
        except Exception as e:
            return (file_path.name, None, str(e))


async def main(folder: str, question: str) -> None:
    api_key = os.environ["DEEPPARSER_API_KEY"]
    base_url = os.environ.get("DEEPPARSER_BASE_URL", "https://deepparser-api.fly.dev")

    files = [
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    ]
    if not files:
        print(f"No supported files found in {folder}")
        return

    print(f"Processing {len(files)} files with question: {question!r}\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with DeepParserClient(api_key=api_key, base_url=base_url) as client:
        tasks = [process_one(client, sem, f, question) for f in files]
        results = await asyncio.gather(*tasks)

    for filename, answer, error in results:
        if error:
            print(f"✗ {filename}: {error}")
        else:
            print(f"✓ {filename}:")
            print(f"  {answer[:200]}{'…' if len(answer) > 200 else ''}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_upload.py <folder> <question>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
