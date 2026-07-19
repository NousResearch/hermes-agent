import asyncio
import zipfile

import gateway.run as gateway_run


_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _write_docx(path, text):
    document = (
        f'<w:document xmlns:w="{_NS_W}"><w:body><w:p><w:r>'
        f"<w:t>{text}</w:t>"
        "</w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)


def test_gateway_inlines_supported_document_without_blocking_loop(tmp_path, monkeypatch):
    path = tmp_path / "report.docx"
    _write_docx(path, "Quarterly result")
    called = False
    real_to_thread = asyncio.to_thread

    async def tracked_to_thread(func, *args, **kwargs):
        nonlocal called
        called = True
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", tracked_to_thread)
    note = asyncio.run(
        gateway_run._build_document_context_note_with_extraction(
            "report.docx",
            str(path),
            "/agent/cache/report.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    assert called is True
    assert "Quarterly result" in note
    assert "/agent/cache/report.docx" in note


def test_gateway_falls_back_to_path_note_for_malformed_document(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip")

    note = asyncio.run(
        gateway_run._build_document_context_note_with_extraction(
            "broken.docx",
            str(path),
            "/agent/cache/broken.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    assert "not inlined" in note
    assert "/agent/cache/broken.docx" in note


def test_gateway_caps_extracted_text_before_prompt_injection(tmp_path, monkeypatch):
    path = tmp_path / "long.docx"
    _write_docx(path, "abcdefghijklmnopqrstuvwxyz")
    monkeypatch.setattr(gateway_run, "_DOCUMENT_TEXT_INLINE_CHAR_LIMIT", 10)

    note = asyncio.run(
        gateway_run._build_document_context_note_with_extraction(
            "long.docx",
            str(path),
            "/agent/cache/long.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    assert "abcdefghij" in note
    assert "klmnopqrstuvwxyz" not in note
    assert "truncated for prompt size" in note
