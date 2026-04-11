from __future__ import annotations

from pathlib import Path

from hermes_constants import get_hermes_home


HERMES_HOME = get_hermes_home()
KB_HOME = HERMES_HOME / "kb"
STAGING_HOME = KB_HOME / "staging"
RAW_PDF_HOME = KB_HOME / "raw" / "pdf"
EVIDENCE_HOME = KB_HOME / "evidence"
DERIVED_PDF_HOME = EVIDENCE_HOME / "derived" / "pdf"
WIKI_HOME = KB_HOME / "wiki"
WIKI_SOURCES_HOME = WIKI_HOME / "sources"
WIKI_QUERIES_HOME = WIKI_HOME / "queries"

INBOX_STAGING_HOME = STAGING_HOME / "inbox"
TELEGRAM_STAGING_HOME = STAGING_HOME / "telegram"
PILOT_STAGING_HOME = STAGING_HOME / "pilot"

SOURCES_JSONL = EVIDENCE_HOME / "sources.jsonl"
CHUNKS_JSONL = EVIDENCE_HOME / "chunks.jsonl"
STATEMENTS_JSONL = EVIDENCE_HOME / "statements.jsonl"
ENTITIES_JSONL = EVIDENCE_HOME / "entities.jsonl"
LINKS_JSONL = EVIDENCE_HOME / "links.jsonl"
TRANSFORMS_JSONL = EVIDENCE_HOME / "transforms.jsonl"
INGEST_STATE_JSON = EVIDENCE_HOME / "ingest-state.json"

PROMOTION_CANDIDATE_FILENAME = "promotion-candidate.md"
RESULT_FILENAME = "result.json"
DOCLING_MARKDOWN_FILENAME = "docling.md"
DOCLING_JSON_FILENAME = "docling.json"
PYMUPDF_MARKDOWN_FILENAME = "pymupdf4llm.md"
PYMUPDF_JSON_FILENAME = "pymupdf4llm.json"
CHUNKS_FILENAME = "chunks.json"

PYMUPDF_FAST_PATH_MAX_PAGES = 9

INGRESS_MANUAL_PILOT = "manual_pilot"
INGRESS_TELEGRAM = "telegram_document"
INGRESS_GMAIL = "gmail_attachment"

PILOT_BATCH_ID = "pdf-pilot-april-2026"
PILOT_FILES = {
    "simple-born-digital": "/Users/quark/Library/CloudStorage/Dropbox/00_Quark_MAIN/00_SHARED_INBOX/2026-02-27 interactive session log.pdf",
    "complex-layout-figures": "/Users/quark/Library/CloudStorage/Dropbox/00_Quark_MAIN/00_SHARED_INBOX/screencapture-x-gkisokay-article-2041129801516458090-2026-04-06-10_11_28.pdf",
    "scanned-ocr-needed": "/Users/quark/Library/CloudStorage/Dropbox/00_Quark_MAIN/00_SHARED_INBOX/US_SocialSecurity_TBO_250602.pdf",
}


def ensure_pdf_ingest_dirs() -> None:
    for path in [
        KB_HOME,
        STAGING_HOME,
        RAW_PDF_HOME,
        EVIDENCE_HOME,
        DERIVED_PDF_HOME,
        WIKI_HOME,
        WIKI_SOURCES_HOME,
        WIKI_QUERIES_HOME,
        INBOX_STAGING_HOME,
        TELEGRAM_STAGING_HOME,
        PILOT_STAGING_HOME,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    for file_path in [
        SOURCES_JSONL,
        CHUNKS_JSONL,
        STATEMENTS_JSONL,
        ENTITIES_JSONL,
        LINKS_JSONL,
        TRANSFORMS_JSONL,
    ]:
        if not file_path.exists():
            file_path.write_text("", encoding="utf-8")

    if not INGEST_STATE_JSON.exists():
        INGEST_STATE_JSON.write_text('{"processed": {}}', encoding="utf-8")
