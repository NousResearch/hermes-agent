#!/usr/bin/env python3
"""Build a FAISS vector index from a directory of PDFs.

Usage:
    python index_pdfs.py ~/papers/
    python index_pdfs.py ~/papers/ --index-dir ./my_index --chunk-size 800 --force

Dependencies: pymupdf, sentence-transformers, faiss-cpu
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pdf_index import PDFIndex


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Index PDFs for semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  index_pdfs.py ~/papers/
  index_pdfs.py ~/papers/ --index-dir ./my_index --force
  index_pdfs.py ~/papers/ --model all-mpnet-base-v2
        """,
    )
    ap.add_argument(
        "input_dir",
        help="Directory containing PDFs (scanned recursively)",
    )
    ap.add_argument(
        "--index-dir", default="./pdf_index",
        help="Where to store index files (default: ./pdf_index)",
    )
    ap.add_argument(
        "--chunk-size", type=int, default=500,
        help="Characters per text chunk (default: 500)",
    )
    ap.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    ap.add_argument(
        "--extensions", default=".pdf",
        help="Comma-separated file extensions (default: .pdf)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-index all files even if already indexed",
    )
    ap.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress bars",
    )
    args = ap.parse_args()

    extensions = tuple(e.strip() for e in args.extensions.split(","))

    idx = PDFIndex(index_dir=args.index_dir)
    idx.index_directory(
        input_dir=args.input_dir,
        chunk_size=args.chunk_size,
        extensions=extensions,
        model_name=args.model,
        force=args.force,
        progress=not args.quiet,
    )

    info = idx.info()
    print(f"\nIndex at: {info['index_dir']}")
    print(f"  Files:  {info['indexed_files']}")
    print(f"  Chunks: {info['total_chunks']}")
    print(f"  Model:  {info['model']}")


if __name__ == "__main__":
    main()
