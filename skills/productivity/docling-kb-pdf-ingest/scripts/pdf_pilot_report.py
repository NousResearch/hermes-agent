from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pdf_ingest_config import PILOT_BATCH_ID, PILOT_FILES, WIKI_QUERIES_HOME
from pdf_ingest_lib import read_json, slugify
from pdf_ingest_pipeline import ingest_pdf


def _score(markdown_text: str, parser_selected: str, ocr_needed: bool) -> dict[str, int]:
    text = markdown_text.strip()
    has_headings = "#" in text
    reading_order = 2 if len(text) > 200 else 1 if text else 0
    heading_structure = 2 if has_headings else 1 if text else 0
    table_figure_preservation = 1 if parser_selected == "docling" else 0
    if ocr_needed:
        ocr_accuracy = 2 if len(text) > 40 else 1 if text else 0
    else:
        ocr_accuracy = 2
    promotion_readiness = 2 if len(text) > 400 else 1 if text else 0
    return {
        "reading_order": reading_order,
        "heading_structure": heading_structure,
        "table_figure_preservation": table_figure_preservation,
        "ocr_text_accuracy": ocr_accuracy,
        "promotion_readiness": promotion_readiness,
    }


def run_pilot() -> dict[str, Any]:
    rows = []
    for label, file_path in PILOT_FILES.items():
        result = ingest_pdf(
            pdf_path=Path(file_path),
            ingress_channel="manual_pilot",
            source_context={"pilot_label": label},
            dry_run_promotion=True,
            promote_source_page=(label != "scanned-ocr-needed"),
        )
        markdown_path = Path(result["markdown_artifact_path"])
        markdown_text = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
        scores = _score(markdown_text, result["parser_selected"], result["ocr_needed"])
        rows.append(
            {
                "label": label,
                "file_path": file_path,
                "record_id": result["record_id"],
                "parser_selected": result["parser_selected"],
                "route_reason": result["route_reason"],
                "latency_ms_total": result["latency_ms_total"],
                "latency_ms_parse": result["latency_ms_parse"],
                "page_count": result["page_count"],
                "text_layer_present": result["text_layer_present"],
                "ocr_needed": result["ocr_needed"],
                "scores": scores,
                "markdown_artifact_path": result["markdown_artifact_path"],
                "promotion_candidate_path": result.get("promotion_candidate_path"),
                "source_page_path": result.get("source_page_path"),
            }
        )
    return {"pilot_batch_id": PILOT_BATCH_ID, "rows": rows}


def write_query_page(report: dict[str, Any]) -> Path:
    WIKI_QUERIES_HOME.mkdir(parents=True, exist_ok=True)
    target = WIKI_QUERIES_HOME / "pdf-ingest-pilot-april-2026.md"
    lines = [
        "---",
        'title: PDF Ingest Pilot — April 2026',
        'created: 2026-04-11',
        'updated: 2026-04-11',
        'type: query',
        'tags: [pdf, ingest, pilot, docling, kb]',
        'sources: ["raw/pdf pilot corpus"]',
        '---',
        '',
        '# PDF Ingest Pilot — April 2026',
        '',
        '> Pilot report for the Docling-first Hermes KB PDF ingest path.',
        '',
        '## Findings',
        '',
    ]
    for row in report["rows"]:
        lines.extend([
            f"- `{row['label']}` → `{row['parser_selected']}` ({row['route_reason']}); total {row['latency_ms_total']} ms; record `{row['record_id']}`",
        ])
    lines.extend([
        '',
        '## Recommendation',
        '',
        '- Keep Docling as the default parser.',
        '- Keep the PyMuPDF4LLM fast path only if the simple born-digital sample shows a meaningful latency win without fidelity loss.',
    ])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    report = run_pilot()
    query_path = write_query_page(report)
    report["query_page_path"] = str(query_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
