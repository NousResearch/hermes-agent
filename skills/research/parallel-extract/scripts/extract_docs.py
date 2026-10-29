#!/usr/bin/env python3
"""Parallel v1 Extract — batch URLs to extracted_docs.json. Requires PARALLEL_API_KEY."""
import argparse
import json
import sys
from pathlib import Path

from parallel import Parallel

EXTRACTION_OBJECTIVE = (
    "Extract comprehensive technical documentation from each URL for an AI "
    "software-engineering knowledge base. Preserve hierarchy, API references, "
    "env vars, auth, code examples, CLI/Docker guidance, limitations, and "
    "best practices. Prefer complete technical content over abbreviated excerpts."
)

SEARCH_QUERIES = [
    "LangChain LangGraph LangSmith",
    "Model Context Protocol MCP",
    "OpenRouter LiteLLM configuration",
]

BATCH_SIZE = 20


def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def extract_urls(urls, out_path="extracted_docs.json", full_content=True):
    client = Parallel()
    session_id = None
    all_results, all_errors, all_usage = [], [], []

    for batch_idx, batch in enumerate(chunk(urls, BATCH_SIZE)):
        print(f"[batch {batch_idx}] extracting {len(batch)} URLs...", file=sys.stderr)
        resp = client.extract(
            urls=batch,
            objective=EXTRACTION_OBJECTIVE,
            search_queries=SEARCH_QUERIES,
            session_id=session_id,
            advanced_settings={
                "full_content": full_content,
                "excerpt_settings": {"max_chars_per_result": 100000},
                "fetch_policy": {
                    "disable_cache_fallback": True,
                    "max_age_seconds": 172800,
                    "timeout_seconds": 120,
                },
            },
        )
        session_id = resp.session_id
        all_results.extend(resp.results)
        all_errors.extend(resp.errors)
        all_usage.extend(getattr(resp, "usage", None) or [])
        print(
            f"  ok={len(resp.results)} err={len(resp.errors)}",
            file=sys.stderr,
        )

    out = {
        "session_id": session_id,
        "results": [
            {
                "url": r.url,
                "title": r.title,
                "publish_date": getattr(r, "publish_date", None),
                "excerpts": list(r.excerpts or []),
                "full_content": r.full_content,
            }
            for r in all_results
        ],
        "errors": [
            {
                "url": e.url,
                "error_type": e.error_type,
                "http_status_code": getattr(e, "http_status_code", None),
                "content": e.content,
            }
            for e in all_errors
        ],
        "usage": [
            {"name": getattr(u, "name", None), "count": getattr(u, "count", None)}
            for u in all_usage
        ],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"Wrote {len(all_results)} results, {len(all_errors)} errors to {out_path}",
        file=sys.stderr,
    )
    return out


def _load_urls(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch Parallel Extract to JSON")
    p.add_argument(
        "urls_file",
        nargs="?",
        help="Text file with one URL per line (# comments allowed)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="extracted_docs.json",
        help="Output JSON path (default: extracted_docs.json)",
    )
    p.add_argument(
        "--no-full-content",
        action="store_true",
        help="Omit full page markdown (excerpts only)",
    )
    args = p.parse_args(argv)
    if not args.urls_file:
        p.error("urls_file is required (one URL per line)")
    urls = _load_urls(Path(args.urls_file))
    if not urls:
        p.error(f"No URLs in {args.urls_file}")
    extract_urls(urls, out_path=args.output, full_content=not args.no_full_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())