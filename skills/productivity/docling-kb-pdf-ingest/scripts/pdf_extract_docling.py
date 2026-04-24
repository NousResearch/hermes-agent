from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz
from docling.document_converter import DocumentConverter
from ocrmac.ocrmac import text_from_image

from pdf_ingest_config import CHUNKS_FILENAME, DOCLING_JSON_FILENAME, DOCLING_MARKDOWN_FILENAME
from pdf_ingest_lib import split_markdown_chunks, write_json

PARSER_NAME = "docling"
OCR_FALLBACK_MIN_TEXT_CHARS = 120


def parser_version() -> str:
    try:
        import docling
        return getattr(docling, "__version__", "unknown")
    except Exception:
        return "unknown"


def _ocr_fallback_markdown(pdf_path: Path, output_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    ocr_dir = output_dir / "ocr-fallback"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    page_blocks: list[str] = []
    ocr_chunks: list[dict[str, Any]] = []
    for page_index, page in enumerate(doc, start=1):
        image_path = ocr_dir / f"page-{page_index}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        pix.save(str(image_path))
        lines = text_from_image(str(image_path), recognition_level="accurate", detail=False)
        line_text = "\n".join(line.strip() for line in lines if line and line.strip())
        if not line_text:
            continue
        page_blocks.append(f"## OCR Fallback — Page {page_index}\n\n{line_text}")
        ocr_chunks.append(
            {
                "ordinal": page_index,
                "section_heading": f"OCR Fallback — Page {page_index}",
                "page_start": page_index,
                "page_end": page_index,
                "parser_chunk_id": f"ocr-fallback-page-{page_index}",
                "provenance_refs": [page_index],
                "text": line_text,
            }
        )
    doc.close()
    return "\n\n".join(page_blocks).strip(), ocr_chunks


def extract_pdf(pdf_path: Path, output_dir: Path, page_count: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    document = result.document
    markdown_text = document.export_to_markdown().strip()
    document_dict = document.export_to_dict()
    chunks = split_markdown_chunks(markdown_text, page_count)

    ocr_fallback_used = False
    if len(markdown_text) < OCR_FALLBACK_MIN_TEXT_CHARS:
        fallback_markdown, fallback_chunks = _ocr_fallback_markdown(pdf_path, output_dir)
        if fallback_markdown and len(fallback_markdown) > len(markdown_text):
            if markdown_text:
                markdown_text = f"{markdown_text}\n\n## OCR Fallback Supplement\n\n{fallback_markdown}"
                chunks.extend(fallback_chunks)
            else:
                markdown_text = fallback_markdown
                chunks = fallback_chunks
            document_dict["ocr_fallback"] = {
                "used": True,
                "pages_with_text": [chunk["page_start"] for chunk in fallback_chunks],
            }
            ocr_fallback_used = True
        else:
            document_dict["ocr_fallback"] = {"used": False, "pages_with_text": []}
    else:
        document_dict["ocr_fallback"] = {"used": False, "pages_with_text": []}

    markdown_path = output_dir / DOCLING_MARKDOWN_FILENAME
    json_path = output_dir / DOCLING_JSON_FILENAME
    chunks_path = output_dir / CHUNKS_FILENAME

    markdown_path.write_text(markdown_text, encoding="utf-8")
    write_json(json_path, document_dict)
    write_json(chunks_path, chunks)

    return {
        "parser_name": PARSER_NAME,
        "parser_version": parser_version(),
        "markdown_text": markdown_text,
        "markdown_path": markdown_path,
        "json_path": json_path,
        "chunk_path": chunks_path,
        "chunks": chunks,
        "ocr_fallback_used": ocr_fallback_used,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract a PDF with Docling")
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
                "ocr_fallback_used": payload["ocr_fallback_used"],
            },
            indent=2,
        )
    )
