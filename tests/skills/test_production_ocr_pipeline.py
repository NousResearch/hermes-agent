"""Tests for the bundled Spearhead production OCR pipeline helper.

These tests use monkeypatched runners and temporary files only. They do not invoke
real tesseract/poppler binaries and do not touch private documents.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills" / "productivity" / "ocr-and-documents" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import production_ocr_pipeline as pop  # noqa: E402


def _result(returncode=0, stdout="", stderr="", timed_out=False, seconds=0.01):
    return pop.CommandResult(returncode, stdout, stderr, timed_out, seconds)


def test_check_does_not_require_input(monkeypatch, capsys):
    monkeypatch.setattr(
        pop,
        "check_tools",
        lambda: {"tesseract": "/usr/bin/tesseract", "pdftotext": "/usr/bin/pdftotext", "pdftoppm": "/usr/bin/pdftoppm", "pdfinfo": "/usr/bin/pdfinfo", "missing": []},
    )

    assert pop.main(["--check"]) == 0
    assert '"missing": []' in capsys.readouterr().out


def test_check_missing_tools_returns_dependency_exit_without_input(monkeypatch, capsys):
    monkeypatch.setattr(
        pop,
        "check_tools",
        lambda: {"tesseract": None, "pdftotext": "/usr/bin/pdftotext", "pdftoppm": None, "pdfinfo": "/usr/bin/pdfinfo", "missing": ["tesseract", "pdftoppm"]},
    )

    assert pop.main(["--check"]) == 2
    out = capsys.readouterr().out
    assert "tesseract" in out
    assert "pdftoppm" in out


def test_text_pdf_short_circuits_before_ocr(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF fake")
    out_dir = tmp_path / "out"

    def fake_run(cmd, *, timeout, sandbox=True, cwd=None):
        assert cmd[0] == "pdftotext"
        Path(cmd[-1]).write_text("Selectable Czech/English text " * 3, encoding="utf-8")
        return _result()

    monkeypatch.setattr(pop, "run_command", fake_run)
    record = pop.process_document(pdf, out_dir, lang="ces+eng", psms=pop.DEFAULT_PSMS, dpi=300, timeout=120)

    assert record["method"] == "text_pdf"
    assert record["pages"] == []
    assert record["status"] == "ok"


def test_image_ocr_uses_default_psm_retry_matrix_and_review_gate(monkeypatch, tmp_path):
    img = tmp_path / "scan.png"
    img.write_bytes(b"fake")

    attempts = []

    def fake_attempt(image, out_dir, *, lang, psm, timeout):
        attempts.append(psm)
        metrics = {"word_count": 1, "char_count": 4, "mean_confidence": 50.0, "median_confidence": 50.0, "low_confidence_word_ratio": 1.0}
        if psm == 6:
            metrics = {"word_count": 2, "char_count": 10, "mean_confidence": 91.0, "median_confidence": 91.0, "low_confidence_word_ratio": 0.0}
        return {"psm": psm, "lang": lang, "status": "ok", "metrics": metrics, "outputs": {}, "attempts": []}

    page = pop.ocr_page_with_retries(img, tmp_path / "page", lang="ces+eng", psms=pop.DEFAULT_PSMS, timeout=120, attempt_fn=fake_attempt)

    assert attempts == [3, 6]
    assert page["selected_psm"] == 6
    assert page["needs_review"] is False


def test_plain_text_batch_writes_manifest(monkeypatch, tmp_path):
    doc = tmp_path / "note.txt"
    doc.write_text("already extracted", encoding="utf-8")
    out_dir = tmp_path / "out"

    assert pop.main([str(doc), "--output-dir", str(out_dir)]) == 0
    manifest = out_dir / "batch-manifest.json"
    assert manifest.exists()
    assert '"plain_text"' in (out_dir / "note" / "manifest.json").read_text(encoding="utf-8")


def test_render_pdf_uses_numeric_page_order(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF fake")
    out_dir = tmp_path / "rendered"

    def fake_run(cmd, *, timeout, sandbox=True, cwd=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ["page-1.png", "page-10.png", "page-2.png"]:
            (out_dir / name).write_bytes(b"img")
        return _result()

    monkeypatch.setattr(pop, "run_command", fake_run)

    pages = pop.render_pdf(pdf, out_dir, dpi=300, timeout=120)

    assert [p.name for p in pages] == ["page-1.png", "page-2.png", "page-10.png"]


def test_render_pdf_creates_output_directory_before_running_poppler(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF fake")
    out_dir = tmp_path / "missing" / "rendered"

    def fake_run(cmd, *, timeout, sandbox=True, cwd=None):
        assert cwd == out_dir
        assert out_dir.exists()
        (out_dir / "page-1.png").write_bytes(b"img")
        return _result()

    monkeypatch.setattr(pop, "run_command", fake_run)

    assert [p.name for p in pop.render_pdf(pdf, out_dir, dpi=300, timeout=120)] == ["page-1.png"]


def test_render_pdf_fails_when_poppler_outputs_no_pages(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF fake")

    monkeypatch.setattr(pop, "run_command", lambda *a, **k: _result())

    with pytest.raises(RuntimeError, match="produced no page images"):
        pop.render_pdf(pdf, tmp_path / "rendered", dpi=300, timeout=120)


def test_duplicate_basenames_get_distinct_manifest_dirs(tmp_path):
    first = tmp_path / "a" / "receipt.txt"
    second = tmp_path / "b" / "receipt.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    out_dir = tmp_path / "out"

    assert pop.main([str(tmp_path), "--output-dir", str(out_dir)]) == 0

    manifests = sorted(out_dir.glob("receipt*/manifest.json"))
    assert len(manifests) == 2


def test_empty_input_directory_returns_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    out_dir = tmp_path / "out"

    assert pop.main([str(empty), "--output-dir", str(out_dir)]) == 1
    assert "no supported input files" in capsys.readouterr().err
