#!/usr/bin/env python3
"""Semantic search over a FAISS PDF index.

Usage:
    python search_pdfs.py "differential privacy convergence"
    python search_pdfs.py "gradient inversion attack" --top-k 10 --show-context
    python search_pdfs.py "Theorem 3 optimal control" --json

Dependencies: sentence-transformers, faiss-cpu
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pdf_index import PDFIndex


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Semantic search over indexed PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  search_pdfs.py "neural network optimization"
  search_pdfs.py "federated learning privacy" --top-k 10 --show-context
  search_pdfs.py "convergence bound" --json | jq '.[].filename'
        """,
    )
    ap.add_argument(
        "query",
        help="Natural language search query",
    )
    ap.add_argument(
        "--index-dir", default="./pdf_index",
        help="Index directory (default: ./pdf_index)",
    )
    ap.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results (default: 5)",
    )
    ap.add_argument(
        "--show-context", action="store_true",
        help="Print full chunk text instead of first 200 chars",
    )
    ap.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = ap.parse_args()

    idx = PDFIndex(index_dir=args.index_dir)

    try:
        results = idx.search(args.query, top_k=args.top_k)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not results:
        print(f"No results for: {args.query}")
        return

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}\n")

    for i, r in enumerate(results, 1):
        text = r["text"] if args.show_context else r["text"][:200]
        if not args.show_context and len(r["text"]) > 200:
            text += "..."

        print(f"{'─' * 60}")
        print(f"#{i}  [{r['filename']}:p{r['page']}]  score={r['score']:.4f}")
        print(f"{'─' * 60}")
        print(text)
        print()


if __name__ == "__main__":
    main()
