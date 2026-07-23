import json
import zipfile
from pathlib import Path

from gateway.file_intake import (
    extract_if_supported,
    get_file_record,
    list_sources,
    record_incoming_file,
)


def test_record_incoming_file_creates_manifest_and_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = tmp_path / "report.pdf"
    payload.write_bytes(b"%PDF-1.4 fake")

    rec = record_incoming_file(
        cache_path=payload,
        original_filename="report.pdf",
        mime_type="application/pdf",
        source_platform="telegram",
        chat_id=100,
        thread_id=200,
        message_id=300,
        user_id=400,
        platform_file_id="file-id",
        platform_unique_id="unique-id",
    )

    assert rec.sha256
    assert rec.duplicate is False
    assert rec.status == "received"
    row = get_file_record(rec.sha256)
    assert row["original_filename"] == "report.pdf"
    assert row["mime_type"] == "application/pdf"
    assert row["size_bytes"] == payload.stat().st_size
    assert Path(row["artifact_dir"]).is_dir()
    sources = list_sources(rec.sha256)
    assert len(sources) == 1
    assert sources[0]["source_platform"] == "telegram"
    assert sources[0]["chat_id"] == "100"
    assert sources[0]["message_id"] == "300"


def test_record_incoming_file_dedups_content_but_keeps_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_bytes(b"same bytes")
    p2.write_bytes(b"same bytes")

    first = record_incoming_file(cache_path=p1, original_filename="a.txt", source_platform="telegram", message_id=1)
    second = record_incoming_file(cache_path=p2, original_filename="b.txt", source_platform="telegram", message_id=2)

    assert first.sha256 == second.sha256
    assert first.duplicate is False
    assert second.duplicate is True
    sources = list_sources(first.sha256)
    assert [s["message_id"] for s in sources] == ["1", "2"]


def _write_pptx(path: Path):
    slide_xml = """
    <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Hello</a:t></a:r><a:r><a:t>world</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>
    """
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
        zf.writestr("ppt/slides/slide2.xml", slide_xml.replace("Hello", "Second"))


def test_extract_if_supported_writes_pptx_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    pptx = tmp_path / "deck.pptx"
    _write_pptx(pptx)

    rec = record_incoming_file(
        cache_path=pptx,
        original_filename="deck.pptx",
        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        source_platform="telegram",
    )
    extracted = extract_if_supported(rec, filename="deck.pptx", mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

    assert extracted.status == "extracted"
    text = Path(extracted.extracted_text_path).read_text(encoding="utf-8")
    assert "Slides: 2" in text
    assert "Hello world" in text
    structure = json.loads(Path(extracted.structure_path).read_text(encoding="utf-8"))
    assert structure["slide_count"] == 2
    row = get_file_record(rec.sha256)
    assert row["status"] == "extracted"
    assert row["extracted_text_path"] == extracted.extracted_text_path


def test_cleanup_keeps_extracted_manifest_file(tmp_path, monkeypatch):
    import os
    import time
    from gateway.platforms.base import cleanup_document_cache, get_document_cache_dir
    from gateway.file_intake import update_file_status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    cache_dir = get_document_cache_dir()
    keep = cache_dir / "keep.pdf"
    old = cache_dir / "old.pdf"
    keep.write_bytes(b"keep")
    old.write_bytes(b"old")

    rec = record_incoming_file(cache_path=keep, original_filename="keep.pdf", source_platform="telegram")
    update_file_status(rec.sha256, "extracted")

    old_time = time.time() - 48 * 3600
    os.utime(keep, (old_time, old_time))
    os.utime(old, (old_time, old_time))

    removed = cleanup_document_cache(max_age_hours=24)
    assert removed == 1
    assert keep.exists()
    assert not old.exists()


def _record_and_extract(tmp_path, filename, mime, writer, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    path = tmp_path / filename
    writer(path)
    rec = record_incoming_file(cache_path=path, original_filename=filename, mime_type=mime, source_platform="telegram")
    return extract_if_supported(rec, filename=filename, mime_type=mime)


def test_docx_extractor_writes_text(tmp_path, monkeypatch):
    def writer(path):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("word/document.xml", '<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>Hello docx</w:t></w:r></w:p></w:body></w:document>')

    rec = _record_and_extract(tmp_path, "doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", writer, monkeypatch)
    assert rec.status == "extracted"
    assert "Hello docx" in Path(rec.extracted_text_path).read_text(encoding="utf-8")


def test_xlsx_extractor_writes_sheet_sample(tmp_path, monkeypatch):
    def writer(path):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("xl/sharedStrings.xml", '<sst xmlns="x"><si><t>Hello xlsx</t></si></sst>')
            zf.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="x"><sheetData><row><c t="s"><v>0</v></c></row></sheetData></worksheet>')

    rec = _record_and_extract(tmp_path, "book.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", writer, monkeypatch)
    assert rec.status == "extracted"
    structure = json.loads(Path(rec.structure_path).read_text(encoding="utf-8"))
    assert structure["sheet_count"] == 1
    assert "Hello xlsx" in Path(rec.extracted_text_path).read_text(encoding="utf-8")


def test_pdf_extractor_rejects_vulnerable_pypdf_before_import(tmp_path, monkeypatch):
    from gateway import file_intake

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(file_intake, "_package_version", lambda name: "5.9.0")
    path = tmp_path / "unsafe.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    rec = record_incoming_file(
        cache_path=path,
        original_filename="unsafe.pdf",
        mime_type="application/pdf",
        source_platform="telegram",
    )

    extracted = extract_if_supported(rec, filename="unsafe.pdf", mime_type="application/pdf")

    assert extracted.status == "received"
    row = get_file_record(rec.sha256)
    assert row is not None
    assert row["status"] == "failed_extract"
    assert "pypdf 5.9.0" in row["error"]
    assert ">=6.13.3,<7" in row["error"]


def test_zip_inspector_rejects_zip_slip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("../evil.txt", "bad")
    rec = record_incoming_file(cache_path=path, original_filename="evil.zip", mime_type="application/zip", source_platform="telegram")

    extracted = extract_if_supported(rec, filename="evil.zip", mime_type="application/zip")

    assert extracted.status == "received"
    row = get_file_record(rec.sha256)
    assert row is not None
    assert row["status"] == "failed_extract"
    assert "unsafe archive member" in row["error"]


def test_zip_inspector_writes_safe_manifest(tmp_path, monkeypatch):
    def writer(path):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("folder/file.txt", "hello")

    rec = _record_and_extract(tmp_path, "safe.zip", "application/zip", writer, monkeypatch)
    assert rec.status == "extracted"
    structure = json.loads(Path(rec.structure_path).read_text(encoding="utf-8"))
    assert structure["entry_count"] == 1
    assert structure["entries"][0]["name"] == "folder/file.txt"


def test_recent_files_and_context_note_and_semantic_card(tmp_path, monkeypatch):
    from gateway.file_intake import (
        build_intake_context_note,
        build_semantic_file_card,
        list_recent_files,
        update_file_status,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = tmp_path / "brief.txt"
    payload.write_text("hello", encoding="utf-8")
    rec = record_incoming_file(
        cache_path=payload,
        original_filename="brief.txt",
        mime_type="text/plain",
        source_platform="telegram",
        chat_id="chat-1",
        thread_id="thread-1",
        message_id="msg-1",
    )
    extracted = Path(rec.artifact_dir) / "extracted.md"
    extracted.write_text("# Brief\nhello", encoding="utf-8")
    update_file_status(rec.sha256, "extracted", extracted_text_path=str(extracted))

    recent = list_recent_files(source_platform="telegram", chat_id="chat-1", limit=5)
    assert recent[0]["sha256"] == rec.sha256
    assert recent[0]["message_id"] == "msg-1"

    note = build_intake_context_note(payload)
    assert rec.sha256 in note
    assert "status=extracted" in note
    assert str(extracted) in note

    card = build_semantic_file_card(rec.sha256, summary="Short summary")
    assert "# File intake: brief.txt" in card
    assert "Short summary" in card
    assert "No automatic Gbrain/Notion writeback" in card
