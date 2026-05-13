"""
dwg_query.py — Query a DWG/DXF drawing for dimensions, schedules, and annotations.

No CAD software required. DeepParser reads .dwg and .dxf files natively.

Usage:
    export DEEPPARSER_API_KEY=dp_live_...
    python dwg_query.py floor_plan.dwg "List all rooms with their areas."
"""
import asyncio
import os
import sys

from deepparser import DeepParserClient


async def main(dwg_path: str, question: str) -> None:
    api_key = os.environ["DEEPPARSER_API_KEY"]
    base_url = os.environ.get("DEEPPARSER_BASE_URL", "https://deepparser-api.fly.dev")

    async with DeepParserClient(api_key=api_key, base_url=base_url) as client:
        print(f"Parsing {dwg_path}…")

        job = await client.parse(dwg_path, sync=True)
        if job.status != "READY":
            # DWG drawings can take 60–90s on first parse
            print("Processing drawing (this may take a minute)…")
            job = await client.wait_until_ready(job.job_id)

        result = await client.ask(job.job_id, question)

        print(f"\nAnswer:\n{result.answer}")
        if result.citations:
            print("\nCited from:")
            for c in result.citations:
                print(f"  {c.filename}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dwg_query.py <drawing.dwg> <question>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
