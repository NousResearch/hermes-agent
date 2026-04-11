from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

from pdf_ingest_config import (
    CHUNKS_JSONL,
    DERIVED_PDF_HOME,
    EVIDENCE_HOME,
    INGEST_STATE_JSON,
    PILOT_BATCH_ID,
    PYMUPDF_FAST_PATH_MAX_PAGES,
    RAW_PDF_HOME,
    SOURCES_JSONL,
    TELEGRAM_STAGING_HOME,
    TRANSFORMS_JSONL,
    ensure_pdf_ingest_dirs,
)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._ -]+", "-", value).strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text or "document"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_source_id(source_sha256: str) -> str:
    return f"src_pdf_{source_sha256[:12]}"


def stable_transform_id(source_id: str, started_at: str) -> str:
    token = hashlib.sha256(f"{source_id}:{started_at}".encode("utf-8")).hexdigest()[:12]
    return f"xform_pdf_{token}"


def sanitize_filename(name: str) -> str:
    base = Path(name).name.strip() or "document.pdf"
    safe = re.sub(r"[^\w.\- ]+", "_", base)
    safe = re.sub(r"\s+", "-", safe)
    return safe or "document.pdf"


def inspect_pdf(path: Path) -> dict[str, Any]:
    doc = fitz.open(path)
    page_count = doc.page_count
    text_pages = 0
    total_text_chars = 0
    for page in doc:
        text = page.get_text("text") or ""
        stripped = text.strip()
        if stripped:
            text_pages += 1
            total_text_chars += len(stripped)
    doc.close()
    text_layer_present = text_pages > 0
    ocr_needed = not text_layer_present
    return {
        "page_count": page_count,
        "text_layer_present": text_layer_present,
        "ocr_needed": ocr_needed,
        "text_pages": text_pages,
        "text_chars": total_text_chars,
        "file_size_bytes": path.stat().st_size,
    }


def choose_parser(*, page_count: int, text_layer_present: bool, ocr_needed: bool) -> tuple[str, str]:
    if text_layer_present and page_count <= PYMUPDF_FAST_PATH_MAX_PAGES and not ocr_needed:
        return (
            "pymupdf4llm",
            f"born-digital text layer present, {page_count} page(s), OCR not needed",
        )
    reason_bits = [f"default Docling path"]
    if not text_layer_present:
        reason_bits.append("no text layer detected")
    if ocr_needed:
        reason_bits.append("OCR likely needed")
    if page_count > PYMUPDF_FAST_PATH_MAX_PAGES:
        reason_bits.append(f"page count {page_count} exceeds fast-path limit")
    return ("docling", "; ".join(reason_bits))


def immutable_raw_pdf_path(*, source_sha256: str, original_filename: str, observed_at: str) -> Path:
    date_prefix = observed_at[:10]
    return RAW_PDF_HOME / date_prefix / f"{source_sha256}__{sanitize_filename(original_filename)}"


def copy_to_raw_if_needed(*, source_path: Path, source_sha256: str, original_filename: str, observed_at: str) -> Path:
    raw_path = immutable_raw_pdf_path(
        source_sha256=source_sha256,
        original_filename=original_filename,
        observed_at=observed_at,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        shutil.copy2(source_path, raw_path)
    return raw_path


def copy_to_telegram_staging(path: Path, original_filename: str | None = None) -> Path:
    ensure_pdf_ingest_dirs()
    name = sanitize_filename(original_filename or path.name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = TELEGRAM_STAGING_HOME / f"{stamp}-{name}"
    shutil.copy2(path, dest)
    return dest


def derived_dir(source_id: str) -> Path:
    target = DERIVED_PDF_HOME / source_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_ingest_state() -> dict[str, Any]:
    ensure_pdf_ingest_dirs()
    state = read_json(INGEST_STATE_JSON, {"processed": {}})
    if "processed" not in state:
        state = {"processed": {}}
    return state


def write_ingest_state(state: dict[str, Any]) -> None:
    write_json(INGEST_STATE_JSON, state)


def dedup_key(*, ingress_channel: str, source_sha256: str) -> str:
    return f"{ingress_channel}:{source_sha256}"


def ensure_evidence_layout() -> None:
    ensure_pdf_ingest_dirs()


def split_markdown_chunks(markdown_text: str, page_count: int) -> list[dict[str, Any]]:
    text = markdown_text.strip()
    if not text:
        return []
    sections = re.split(r"\n(?=##?\s)", text)
    chunks: list[dict[str, Any]] = []
    for idx, section in enumerate(sections, start=1):
        heading = None
        first_line = section.strip().splitlines()[0] if section.strip().splitlines() else ""
        if first_line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", first_line).strip() or None
        chunks.append(
            {
                "ordinal": idx,
                "section_heading": heading,
                "page_start": 1,
                "page_end": max(1, page_count),
                "text": section.strip(),
            }
        )
    return chunks


def build_source_record(
    *,
    source_id: str,
    original_filename: str,
    staging_path: Path,
    raw_path: Path,
    source_sha256: str,
    ingress_channel: str,
    inspected: dict[str, Any],
    parser_selected: str,
    route_reason: str,
    parse_started_at: str,
    parse_completed_at: str,
    latency_ms_total: int,
    latency_ms_parse: int,
    markdown_artifact_path: Path,
    json_artifact_path: Path,
    chunk_artifact_path: Path,
    parser_version: str,
    source_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "lineage_version": 1,
        "record_type": "source",
        "source_type": "pdf",
        "source_id": source_id,
        "record_id": source_id,
        "ingress_channel": ingress_channel,
        "original_filename": original_filename,
        "staging_path": str(staging_path),
        "raw_path": str(raw_path),
        "source_sha256": source_sha256,
        "page_count": inspected["page_count"],
        "file_size_bytes": inspected["file_size_bytes"],
        "text_layer_present": inspected["text_layer_present"],
        "ocr_needed": inspected["ocr_needed"],
        "parser_selected": parser_selected,
        "route_reason": route_reason,
        "parser_version": parser_version,
        "parse_started_at": parse_started_at,
        "parse_completed_at": parse_completed_at,
        "latency_ms_total": latency_ms_total,
        "latency_ms_parse": latency_ms_parse,
        "markdown_artifact_path": str(markdown_artifact_path),
        "json_artifact_path": str(json_artifact_path),
        "chunk_artifact_path": str(chunk_artifact_path),
        "pilot_batch_id": PILOT_BATCH_ID,
        "source_context": source_context or {},
    }


def build_chunk_records(*, source_id: str, source_sha256: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        ordinal = int(chunk["ordinal"])
        chunk_text = chunk["text"].strip()
        text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        chunk_id = f"chunk_pdf_{source_sha256[:12]}_{ordinal:03d}"
        records.append(
            {
                "lineage_version": 1,
                "record_type": "chunk",
                "source_id": source_id,
                "chunk_id": chunk_id,
                "ordinal": ordinal,
                "page_start": chunk.get("page_start", 1),
                "page_end": chunk.get("page_end", 1),
                "section_heading": chunk.get("section_heading"),
                "parser_chunk_id": chunk.get("parser_chunk_id"),
                "text": chunk_text,
                "text_hash": text_hash,
                "provenance_refs": chunk.get("provenance_refs", []),
            }
        )
    return records


def build_transform_record(
    *,
    transform_id: str,
    source_id: str,
    ingress_channel: str,
    parser_selected: str,
    route_reason: str,
    started_at: str,
    completed_at: str,
    status: str,
    output_counts: dict[str, int],
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "lineage_version": 1,
        "record_type": "transform",
        "transform_id": transform_id,
        "transform_type": "pdf_ingest",
        "source_id": source_id,
        "ingress_channel": ingress_channel,
        "parser_selected": parser_selected,
        "route_reason": route_reason,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "error_message": error_message,
        "output_counts": output_counts,
    }


def record_success(state: dict[str, Any], *, key: str, source_record: dict[str, Any], transform_record: dict[str, Any]) -> None:
    state["processed"][key] = {
        "status": "success",
        "source_id": source_record["source_id"],
        "record_id": source_record["record_id"],
        "raw_path": source_record["raw_path"],
        "parser_selected": source_record["parser_selected"],
        "route_reason": source_record["route_reason"],
        "parse_completed_at": source_record["parse_completed_at"],
        "transform_id": transform_record["transform_id"],
    }


def record_duplicate_transform(*, source_id: str, ingress_channel: str, parser_selected: str, route_reason: str) -> dict[str, Any]:
    started_at = iso_now()
    transform_id = stable_transform_id(source_id, started_at)
    transform = build_transform_record(
        transform_id=transform_id,
        source_id=source_id,
        ingress_channel=ingress_channel,
        parser_selected=parser_selected,
        route_reason=route_reason,
        started_at=started_at,
        completed_at=started_at,
        status="skipped_duplicate",
        error_message=None,
        output_counts={
            "chunks_written": 0,
            "statements_written": 0,
            "entities_written": 0,
            "links_written": 0,
            "promotion_candidates_written": 0,
        },
    )
    append_jsonl(TRANSFORMS_JSONL, transform)
    return transform


def persist_records(source_record: dict[str, Any], chunk_records: list[dict[str, Any]], transform_record: dict[str, Any]) -> None:
    append_jsonl(SOURCES_JSONL, source_record)
    for chunk in chunk_records:
        append_jsonl(CHUNKS_JSONL, chunk)
    append_jsonl(TRANSFORMS_JSONL, transform_record)


def find_source_record(source_id: str) -> dict[str, Any] | None:
    if not SOURCES_JSONL.exists():
        return None
    for line in SOURCES_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("source_id") == source_id:
            return record
    return None


class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)
