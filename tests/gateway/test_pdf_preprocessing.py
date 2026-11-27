from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.pdf_preprocessing import (
    PdfPreprocessingError,
    preprocess_event_pdfs,
    preprocess_pdf_path,
    recover_pending_pdfs,
)
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import SessionSource


def _pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.7\ncontrolled fixture\n")


def _converter(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/python3
import hashlib, json, os, shutil, sys
from pathlib import Path

source = Path(sys.argv[1])
digest = hashlib.sha256(source.read_bytes()).hexdigest()
root = Path(os.environ["MARKITDOWN_OUTPUT_ROOT"]) / "by-sha256" / digest
assets = root / "assets"
assets.mkdir(parents=True, exist_ok=True)
render = assets / "page-0001.png"
render.write_bytes(b"rendered-page")
markdown = root / "fixture.md"
markdown.write_text("# complete\\n\\n## Página 1\\n\\nOCR WITNESS 924\\n", encoding="utf-8")
copied = root / source.name
shutil.copy2(source, copied)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    "status": "complete", "all_pages_covered": True,
    "source_sha256": digest, "page_count": 1,
    "pages": [{"page": 1, "ocr_attempted": True, "ocr_text_chars": 15,
               "render_path": str(render), "render_sha256": sha(render)}],
    "markdown_path": str(markdown), "markdown_sha256": sha(markdown),
    "copied_source": str(copied), "manifest_path": str(root / "manifest.json"),
}
(root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
print(json.dumps({"ok": True, "count": 1, "results": [manifest]}))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _configure(tmp_path: Path, monkeypatch, converter: Path) -> Path:
    archive = tmp_path / "archive"
    monkeypatch.setenv("HERMES_USER_FILES_DIR", str(archive))
    monkeypatch.setenv("MARKITDOWN_OUTPUT_ROOT", str(archive / ".markdown"))
    monkeypatch.setenv("HERMES_PDF_PREPROCESSOR", str(converter))
    return archive


def test_event_pdf_is_replaced_only_by_complete_markdown(tmp_path, monkeypatch):
    archive = _configure(tmp_path, monkeypatch, _converter(tmp_path / "converter"))
    source = tmp_path / "doc_original.pdf"
    _pdf(source)
    event = SimpleNamespace(
        media_urls=[str(source)], media_types=["application/pdf"],
        media_text_inlined=[False], metadata={},
    )

    records = asyncio.run(preprocess_event_pdfs(event))

    assert records == [{
        "original_name": "original.pdf",
        "source_sha256": records[0]["source_sha256"],
        "page_count": 1,
        "markdown_path": event.media_urls[0],
        "manifest_path": records[0]["manifest_path"],
        "all_pages_covered": True,
    }]
    assert event.media_types == ["text/markdown"]
    assert event.media_text_inlined == [False]
    assert "OCR WITNESS 924" in Path(event.media_urls[0]).read_text(encoding="utf-8")
    assert Path(records[0]["manifest_path"]).is_file()
    assert not list((archive / "pdf-inbox" / "pending").glob("*.pdf"))


def test_failed_converter_is_fail_closed_and_original_stays_pending(tmp_path, monkeypatch):
    archive = _configure(tmp_path, monkeypatch, tmp_path / "missing-converter")
    source = tmp_path / "important.pdf"
    _pdf(source)

    with pytest.raises(PdfPreprocessingError, match="unavailable"):
        preprocess_pdf_path(str(source))

    pending = list((archive / "pdf-inbox" / "pending").glob("*.pdf"))
    failures = list((archive / "pdf-inbox" / "pending").glob("*.failure.json"))
    assert len(pending) == len(failures) == 1
    assert pending[0].read_bytes() == source.read_bytes()
    receipt = json.loads(failures[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "pending_recovery"
    assert receipt["attempts"] == 1


def test_pending_conversion_recovers_idempotently(tmp_path, monkeypatch):
    archive = _configure(tmp_path, monkeypatch, tmp_path / "missing-converter")
    source = tmp_path / "recover.pdf"
    _pdf(source)
    with pytest.raises(PdfPreprocessingError):
        preprocess_pdf_path(str(source))

    monkeypatch.setenv("HERMES_PDF_PREPROCESSOR", str(_converter(tmp_path / "converter")))
    assert recover_pending_pdfs() == {"found": 1, "recovered": 1, "failed": 0}
    assert recover_pending_pdfs() == {"found": 0, "recovered": 0, "failed": 0}
    assert not list((archive / "pdf-inbox" / "pending").glob("*"))


def test_non_pdf_document_is_untouched(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("hello", encoding="utf-8")
    event = SimpleNamespace(
        media_urls=[str(source)], media_types=["text/plain"],
        media_text_inlined=[False], metadata={},
    )
    records = asyncio.run(preprocess_event_pdfs(event))
    assert records == []
    assert event.media_urls == [str(source)]
    assert event.media_types == ["text/plain"]


def test_pdf_magic_is_converted_even_without_pdf_metadata(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, _converter(tmp_path / "converter"))
    source = tmp_path / "opaque-upload"
    _pdf(source)
    event = SimpleNamespace(
        media_urls=[str(source)], media_types=["application/octet-stream"],
        media_text_inlined=[False], metadata={},
    )

    records = asyncio.run(preprocess_event_pdfs(event))

    assert len(records) == 1
    assert event.media_types == ["text/markdown"]


def test_declared_pdf_with_invalid_magic_fails_closed(tmp_path):
    source = tmp_path / "declared.pdf"
    source.write_text("not actually a PDF", encoding="utf-8")
    event = SimpleNamespace(
        media_urls=[str(source)], media_types=["application/pdf"],
        media_text_inlined=[False], metadata={},
    )

    with pytest.raises(PdfPreprocessingError, match="invalid magic bytes"):
        asyncio.run(preprocess_event_pdfs(event))


def test_inbound_pipeline_stops_before_model_context_on_pdf_failure(monkeypatch):
    async def fail_closed(_event):
        raise PdfPreprocessingError("forced OCR failure")

    monkeypatch.setattr("gateway.pdf_preprocessing.preprocess_event_pdfs", fail_closed)
    adapter = SimpleNamespace(send=AsyncMock())

    class Runner(GatewayInboundMixin):
        def _session_key_for_source(self, source):
            return "session"

        def _consume_pending_native_image_paths(self, session_key):
            return []

        def _delivery_adapter_for(self, source):
            return adapter

    runner = Runner()
    source = SessionSource(
        platform=Platform.WHATSAPP, chat_id="chat", chat_type="dm", user_id="user",
    )
    event = MessageEvent(
        source=source, text="must not reach context", message_type=MessageType.DOCUMENT,
        media_urls=["document.pdf"], media_types=["application/pdf"],
        media_text_inlined=[False], metadata={},
    )

    result = asyncio.run(runner._prepare_inbound_message_text(event=event, source=source, history=[]))

    assert result is None
    adapter.send.assert_awaited_once()