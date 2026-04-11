from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf4llm

from pdf_ingest_config import CHUNKS_FILENAME, PYMUPDF_JSON_FILENAME, PYMUPDF_MARKDOWN_FILENAME
from pdf_ingest_lib import write_json

PARSER_NAME = "pymupdf4llm"


def parser_version() -> str:
    try:
        import pymupdf4llm as mod
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"


def extract_pdf(pdf_path: Path, output_dir: Path, page_count: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    chunks: list[dict[str, Any]] = []
    for idx, chunk in enumerate(page_chunks, start=1):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        chunks.append(
            {
                "ordinal": idx,
                "section_heading": None,
                "page_start": idx,
                "page_end": idx,
                "parser_chunk_id": f"page-{idx}",
                "provenance_refs": [idx],
                "text": text,
            }
        )

    markdown_path = output_dir / PYMUPDF_MARKDOWN_FILENAME
    json_path = output_dir / PYMUPDF_JSON_FILENAME
    chunks_path = output_dir / CHUNKS_FILENAME

    markdown_path.write_text(markdown_text, encoding="utf-8")
    write_json(json_path, {"page_chunks": page_chunks, "page_count": page_count})
    write_json(chunks_path, chunks)

    return {
        "parser_name": PARSER_NAME,
        "parser_version": parser_version(),
        "markdown_text": markdown_text,
        "markdown_path": markdown_path,
        "json_path": json_path,
        "chunk_path": chunks_path,
        "chunks": chunks,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract a PDF with PyMuPDF4LLM")
    parser.add_argument("pdf_path")
    parser.add_argument("output_dir")
    parser.add_argument("--page-count", type=int, default=1)
    args = parser.parse_args()

    payload = extract_pdf(Path(args.pdf_path), Path(args.output_dir), args.page_count)
    print(
        json.dumps(
            {
                "parser_name": payload["parser_name"],
                "parser_version": payload["parser_version"],
                "markdown_path": str(payload["markdown_path"]),
                "json_path": str(payload["json_path"]),
                "chunk_path": str(payload["chunk_path"]),
                "chunk_count": len(payload["chunks"]),
            },
            indent=2,
        )
    )
