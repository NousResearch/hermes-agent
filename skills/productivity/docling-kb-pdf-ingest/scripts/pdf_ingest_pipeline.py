from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pdf_extract_docling import extract_pdf as extract_docling
from pdf_extract_pymupdf4llm import extract_pdf as extract_pymupdf4llm
from pdf_ingest_config import RESULT_FILENAME, ensure_pdf_ingest_dirs
from pdf_ingest_lib import (
    Timer,
    build_chunk_records,
    build_source_record,
    build_transform_record,
    choose_parser,
    copy_to_raw_if_needed,
    dedup_key,
    derived_dir,
    find_source_record,
    inspect_pdf,
    iso_now,
    persist_records,
    read_ingest_state,
    read_json,
    record_duplicate_transform,
    record_success,
    sanitize_filename,
    sha256_file,
    stable_source_id,
    stable_transform_id,
    write_ingest_state,
    write_json,
)
from pdf_promote_to_wiki import promote_source_summary, write_promotion_candidate


PARSER_MAP = {
    "docling": extract_docling,
    "pymupdf4llm": extract_pymupdf4llm,
}


def ingest_pdf(
    *,
    pdf_path: Path,
    ingress_channel: str,
    source_context: dict[str, Any] | None = None,
    dry_run_promotion: bool = True,
    promote_source_page: bool = False,
) -> dict[str, Any]:
    ensure_pdf_ingest_dirs()
    timer_total = Timer()
    staged_path = pdf_path.resolve()
    original_filename = sanitize_filename(staged_path.name)
    source_sha256 = sha256_file(staged_path)
    state = read_ingest_state()
    key = dedup_key(ingress_channel=ingress_channel, source_sha256=source_sha256)

    existing = state["processed"].get(key)
    if existing:
        source_id = existing["source_id"]
        transform = record_duplicate_transform(
            source_id=source_id,
            ingress_channel=ingress_channel,
            parser_selected=existing.get("parser_selected", "unknown"),
            route_reason=existing.get("route_reason", "duplicate reuse"),
        )
        result_path = derived_dir(source_id) / RESULT_FILENAME
        result = read_json(result_path, {}) if result_path.exists() else {}
        if not result:
            result = {
                "source_id": source_id,
                "record_id": existing.get("record_id", source_id),
                "raw_path": existing["raw_path"],
                "parser_selected": existing.get("parser_selected"),
                "route_reason": existing.get("route_reason"),
            }
        result["status"] = "skipped_duplicate"
        result["transform_id"] = transform["transform_id"]

        if promote_source_page:
            source_record = find_source_record(source_id)
            markdown_path = Path(result.get("markdown_artifact_path", ""))
            if source_record and markdown_path.exists():
                result["source_page_path"] = str(
                    promote_source_summary(
                        source_record=source_record,
                        markdown_text=markdown_path.read_text(encoding="utf-8"),
                    )
                )

        write_json(result_path, result)
        return result

    inspected = inspect_pdf(staged_path)
    parser_selected, route_reason = choose_parser(
        page_count=inspected["page_count"],
        text_layer_present=inspected["text_layer_present"],
        ocr_needed=inspected["ocr_needed"],
    )

    observed_at = iso_now()
    raw_path = copy_to_raw_if_needed(
        source_path=staged_path,
        source_sha256=source_sha256,
        original_filename=original_filename,
        observed_at=observed_at,
    )

    source_id = stable_source_id(source_sha256)
    source_output_dir = derived_dir(source_id)
    parse_started_at = iso_now()
    timer_parse = Timer()
    parser_payload = PARSER_MAP[parser_selected](raw_path, source_output_dir, inspected["page_count"])
    latency_ms_parse = timer_parse.ms()
    parse_completed_at = iso_now()

    chunk_records = build_chunk_records(
        source_id=source_id,
        source_sha256=source_sha256,
        chunks=parser_payload["chunks"],
    )

    source_record = build_source_record(
        source_id=source_id,
        original_filename=original_filename,
        staging_path=staged_path,
        raw_path=raw_path,
        source_sha256=source_sha256,
        ingress_channel=ingress_channel,
        inspected=inspected,
        parser_selected=parser_selected,
        route_reason=route_reason,
        parse_started_at=parse_started_at,
        parse_completed_at=parse_completed_at,
        latency_ms_total=timer_total.ms(),
        latency_ms_parse=latency_ms_parse,
        markdown_artifact_path=parser_payload["markdown_path"],
        json_artifact_path=parser_payload["json_path"],
        chunk_artifact_path=parser_payload["chunk_path"],
        parser_version=parser_payload["parser_version"],
        source_context=source_context,
    )

    transform_id = stable_transform_id(source_id, parse_started_at)
    promotion_candidate_path = None
    source_page_path = None
    if dry_run_promotion:
        promotion_candidate_path = write_promotion_candidate(
            output_dir=source_output_dir,
            source_record=source_record,
            markdown_text=parser_payload["markdown_text"],
        )
    if promote_source_page:
        source_page_path = promote_source_summary(
            source_record=source_record,
            markdown_text=parser_payload["markdown_text"],
        )

    transform_record = build_transform_record(
        transform_id=transform_id,
        source_id=source_id,
        ingress_channel=ingress_channel,
        parser_selected=parser_selected,
        route_reason=route_reason,
        started_at=parse_started_at,
        completed_at=parse_completed_at,
        status="success",
        error_message=None,
        output_counts={
            "chunks_written": len(chunk_records),
            "statements_written": 0,
            "entities_written": 0,
            "links_written": 0,
            "promotion_candidates_written": 1 if promotion_candidate_path else 0,
            "source_pages_written": 1 if source_page_path else 0,
        },
    )

    persist_records(source_record, chunk_records, transform_record)
    record_success(state, key=key, source_record=source_record, transform_record=transform_record)
    write_ingest_state(state)

    result = {
        "status": "success",
        "source_id": source_id,
        "record_id": source_id,
        "raw_path": str(raw_path),
        "staging_path": str(staged_path),
        "parser_selected": parser_selected,
        "route_reason": route_reason,
        "page_count": inspected["page_count"],
        "text_layer_present": inspected["text_layer_present"],
        "ocr_needed": inspected["ocr_needed"],
        "latency_ms_total": timer_total.ms(),
        "latency_ms_parse": latency_ms_parse,
        "transform_id": transform_id,
        "markdown_artifact_path": str(parser_payload["markdown_path"]),
        "json_artifact_path": str(parser_payload["json_path"]),
        "chunk_artifact_path": str(parser_payload["chunk_path"]),
        "promotion_candidate_path": str(promotion_candidate_path) if promotion_candidate_path else None,
        "source_page_path": str(source_page_path) if source_page_path else None,
    }
    write_json(source_output_dir / RESULT_FILENAME, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a staged PDF into the Hermes evidence layer")
    parser.add_argument("pdf_path")
    parser.add_argument("--ingress-channel", default="manual_pilot")
    parser.add_argument("--source-context-json", default="{}")
    parser.add_argument("--no-dry-run-promotion", action="store_true")
    parser.add_argument("--promote-source-page", action="store_true")
    args = parser.parse_args()

    source_context = json.loads(args.source_context_json)
    result = ingest_pdf(
        pdf_path=Path(args.pdf_path),
        ingress_channel=args.ingress_channel,
        source_context=source_context,
        dry_run_promotion=not args.no_dry_run_promotion,
        promote_source_page=args.promote_source_page,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
