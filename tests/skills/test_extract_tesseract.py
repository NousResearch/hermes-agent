"""Tests for skills/productivity/ocr-and-documents/scripts/extract_tesseract.py.

The optional local/offline Tesseract OCR backend. These tests never invoke the real
tesseract/pdftoppm binaries, require no network, and use no credentials -- every
external command is intercepted via the module's single `_run` choke point or via
`find_binary`/`list_languages` monkeypatches.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills" / "productivity" / "ocr-and-documents" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import extract_tesseract as et  # noqa: E402


def _completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


# --------------------------------------------------------------------------- #
# 1. Missing tesseract binary
# --------------------------------------------------------------------------- #
class TestMissingTesseract:
    def test_require_binary_raises_missing_dependency(self, monkeypatch):
        monkeypatch.setattr(et, "find_binary", lambda name: None)
        with pytest.raises(et.OcrError) as exc:
            et.require_binary(et.TESSERACT_BIN, "hint")
        assert exc.value.error_type == "missing_dependency"
        assert exc.value.exit_code == et.EXIT_MISSING_DEP

    def test_ocr_image_missing_tesseract(self, monkeypatch, tmp_path):
        img = tmp_path / "scan.png"
        img.write_bytes(b"\x89PNG fake")
        monkeypatch.setattr(et, "find_binary", lambda name: None)
        with pytest.raises(et.OcrError) as exc:
            et.ocr_image(img)
        assert exc.value.error_type == "missing_dependency"

    def test_main_missing_tesseract_exit_code(self, monkeypatch, tmp_path, capsys):
        img = tmp_path / "scan.png"
        img.write_bytes(b"\x89PNG fake")
        monkeypatch.setattr(et, "find_binary", lambda name: None)
        rc = et.main([str(img), "--json"])
        assert rc == et.EXIT_MISSING_DEP
        assert "missing_dependency" in capsys.readouterr().err

    def test_run_filenotfound_maps_to_missing_dependency(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("no such file")
        monkeypatch.setattr(et.subprocess, "run", boom)
        with pytest.raises(et.OcrError) as exc:
            et._run(["tesseract", "--version"], timeout=5)
        assert exc.value.error_type == "missing_dependency"


# --------------------------------------------------------------------------- #
# 2. Missing / invalid language
# --------------------------------------------------------------------------- #
class TestLanguageValidation:
    def test_missing_language(self):
        with pytest.raises(et.OcrError) as exc:
            et.validate_language("ces", available=["eng", "osd"])
        assert exc.value.error_type == "missing_language"
        assert exc.value.exit_code == et.EXIT_LANGUAGE

    def test_installed_language_passes(self):
        assert et.validate_language("eng", available=["eng", "ces"]) == ["eng"]

    def test_multi_language_partial_missing(self):
        with pytest.raises(et.OcrError) as exc:
            et.validate_language("eng+ces", available=["eng"])
        assert exc.value.error_type == "missing_language"
        assert "ces" in exc.value.message

    @pytest.mark.parametrize("bad", ["", "../etc/passwd", "eng;rm -rf", "en g", "eng+"])
    def test_malformed_spec_rejected(self, bad):
        with pytest.raises(et.OcrError) as exc:
            et.validate_language(bad, available=["eng"])
        assert exc.value.error_type == "invalid_language"

    def test_list_languages_parses_output(self, monkeypatch):
        monkeypatch.setattr(et, "find_binary", lambda name: "/usr/bin/tesseract")
        monkeypatch.setattr(
            et, "_run",
            lambda cmd, timeout: _completed(stdout="List of available languages:\neng\nces\nosd\n"),
        )
        assert et.list_languages() == ["eng", "ces", "osd"]


# --------------------------------------------------------------------------- #
# 3. Image OCR: command construction + error handling
# --------------------------------------------------------------------------- #
class TestImageOcr:
    def test_spearhead_defaults_use_mixed_czech_english_and_final_dpi(self):
        assert et.DEFAULT_LANG == "ces+eng"
        assert et.RASTER_DPI == 300

    def test_build_ocr_command_is_explicit_args(self):
        cmd = et.build_ocr_command("/usr/bin/tesseract", "/tmp/a.png", "eng+ces")
        assert cmd == ["/usr/bin/tesseract", "/tmp/a.png", "stdout", "-l", "eng+ces"]
        # all entries are strings -> safe for shell=False execution
        assert all(isinstance(x, str) for x in cmd)

    def test_build_ocr_command_extra_config(self):
        cmd = et.build_ocr_command("/usr/bin/tesseract", "a.png", "eng", extra_config=["--psm", 6])
        assert cmd[-2:] == ["--psm", "6"]

    def test_ocr_image_success(self, monkeypatch, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(b"fake")
        monkeypatch.setattr(et, "find_binary", lambda name: "/usr/bin/tesseract")
        monkeypatch.setattr(et, "list_languages", lambda *a, **k: ["eng", "ces"])
        captured = {}

        def fake_run(cmd, timeout):
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            return _completed(stdout="hello world\n", returncode=0)

        monkeypatch.setattr(et, "_run", fake_run)
        out = et.ocr_image(img, lang="eng", timeout=33)
        assert out == "hello world\n"
        assert captured["cmd"][0] == "/usr/bin/tesseract"
        assert "stdout" in captured["cmd"]
        assert captured["timeout"] == 33

    def test_ocr_image_nonzero_returncode_raises(self, monkeypatch, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(b"fake")
        monkeypatch.setattr(et, "find_binary", lambda name: "/usr/bin/tesseract")
        monkeypatch.setattr(et, "list_languages", lambda *a, **k: ["eng", "ces"])
        monkeypatch.setattr(et, "_run", lambda cmd, timeout: _completed(stderr="boom", returncode=1))
        with pytest.raises(et.OcrError) as exc:
            et.ocr_image(img, lang="eng")
        assert exc.value.error_type == "ocr_failed"

    def test_ocr_image_missing_file(self, tmp_path):
        with pytest.raises(et.OcrError) as exc:
            et.ocr_image(tmp_path / "nope.png")
        assert exc.value.error_type == "input_not_found"
        assert exc.value.exit_code == et.EXIT_INPUT

    def test_ocr_image_file_too_large(self, monkeypatch, tmp_path):
        img = tmp_path / "big.png"
        img.write_bytes(b"x" * 1000)
        with pytest.raises(et.OcrError) as exc:
            et.ocr_image(img, max_file_size=10)
        assert exc.value.error_type == "file_too_large"
        assert exc.value.exit_code == et.EXIT_LIMIT


# --------------------------------------------------------------------------- #
# 4. PDF rasterization gating + text-first preference
# --------------------------------------------------------------------------- #
class TestPdfRouting:
    def test_text_layer_preferred_no_ocr(self, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(et, "detect_pdf_text", lambda p, max_pages=50: "embedded text")

        def fail_run(*a, **k):
            raise AssertionError("OCR must not run when a text layer exists")

        monkeypatch.setattr(et, "_run", fail_run)
        result = et.ocr_pdf(pdf)
        assert result == {"source": "text-layer", "text": "embedded text", "pages_ocred": 0}

    def test_force_ocr_bypasses_text_layer(self, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(et, "detect_pdf_text", lambda p, max_pages=50: "embedded text")
        monkeypatch.setattr(et, "find_binary", lambda name: "/usr/bin/tesseract" if name == et.TESSERACT_BIN else "/usr/bin/pdftoppm")
        monkeypatch.setattr(et, "list_languages", lambda *a, **k: ["eng", "ces"])

        def fake_raster(pdf_path, out_dir, **k):
            p = Path(out_dir) / "page-1.png"
            p.write_bytes(b"img")
            return [p]

        monkeypatch.setattr(et, "rasterize_pdf", fake_raster)
        monkeypatch.setattr(et, "ocr_image", lambda *a, **k: "ocr page text")
        result = et.ocr_pdf(pdf, force_ocr=True)
        assert result["source"] == "ocr"
        assert result["pages_ocred"] == 1
        assert "ocr page text" in result["text"]

    def test_rasterize_uses_pdftoppm(self, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF fake")
        out = tmp_path / "out"
        monkeypatch.setattr(et, "find_binary", lambda name: "/usr/bin/pdftoppm" if name == et.PDFTOPPM_BIN else None)
        captured = {}

        def fake_run(cmd, timeout):
            captured["cmd"] = cmd
            (out / "page-1.png").write_bytes(b"img")
            return _completed(returncode=0)

        monkeypatch.setattr(et, "_run", fake_run)
        images = et.rasterize_pdf(pdf, out)
        assert [p.name for p in images] == ["page-1.png"]
        assert captured["cmd"][0] == "/usr/bin/pdftoppm"
        assert "-png" in captured["cmd"]

    def test_rasterize_sorts_pdftoppm_pages_numerically(self, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF fake")
        out = tmp_path / "out"
        monkeypatch.setattr(et, "find_binary", lambda name: "/usr/bin/pdftoppm" if name == et.PDFTOPPM_BIN else None)

        def fake_run(cmd, timeout):
            for name in ["page-1.png", "page-10.png", "page-2.png"]:
                (out / name).write_bytes(b"img")
            return _completed(returncode=0)

        monkeypatch.setattr(et, "_run", fake_run)
        images = et.rasterize_pdf(pdf, out)
        assert [p.name for p in images] == ["page-1.png", "page-2.png", "page-10.png"]

    def test_rasterize_no_rasterizer_raises_missing_dependency(self, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF fake")
        monkeypatch.setattr(et, "find_binary", lambda name: None)
        # Force the pymupdf fallback import to fail deterministically.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def no_pymupdf(name, *a, **k):
            if name == "pymupdf":
                raise ImportError("no pymupdf")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", no_pymupdf)
        with pytest.raises(et.OcrError) as exc:
            et.rasterize_pdf(pdf, tmp_path / "out")
        assert exc.value.error_type == "missing_dependency"
        assert exc.value.exit_code == et.EXIT_MISSING_DEP

    def test_pdf_force_ocr_missing_tesseract(self, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF fake")
        monkeypatch.setattr(et, "find_binary", lambda name: None)
        with pytest.raises(et.OcrError) as exc:
            et.ocr_pdf(pdf, force_ocr=True)
        assert exc.value.error_type == "missing_dependency"

    def test_pdf_too_large(self, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"x" * 5000)
        with pytest.raises(et.OcrError) as exc:
            et.ocr_pdf(pdf, max_file_size=10)
        assert exc.value.error_type == "file_too_large"


# --------------------------------------------------------------------------- #
# 5. Timeout / failure path
# --------------------------------------------------------------------------- #
class TestTimeout:
    def test_run_timeout_maps_to_timeout_error(self, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="tesseract", timeout=1)

        monkeypatch.setattr(et.subprocess, "run", boom)
        with pytest.raises(et.OcrError) as exc:
            et._run(["tesseract", "a.png", "stdout"], timeout=1)
        assert exc.value.error_type == "timeout"
        assert exc.value.exit_code == et.EXIT_TIMEOUT

    def test_ocr_image_propagates_timeout(self, monkeypatch, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(b"fake")
        monkeypatch.setattr(et, "find_binary", lambda name: "/usr/bin/tesseract")
        monkeypatch.setattr(et, "list_languages", lambda *a, **k: ["eng", "ces"])

        def timeout_run(cmd, timeout):
            raise et.OcrError("timeout", "timed out", et.EXIT_TIMEOUT)

        monkeypatch.setattr(et, "_run", timeout_run)
        with pytest.raises(et.OcrError) as exc:
            et.ocr_image(img, lang="eng")
        assert exc.value.exit_code == et.EXIT_TIMEOUT


# --------------------------------------------------------------------------- #
# CLI: --check and structured output (no real deps required)
# --------------------------------------------------------------------------- #
class TestCli:
    def test_check_reports_status_without_deps(self, monkeypatch, capsys):
        monkeypatch.setattr(et, "find_binary", lambda name: None)
        monkeypatch.setattr(et, "_pymupdf_available", lambda: False)
        rc = et.main(["--check"])
        assert rc == et.EXIT_OK
        import json
        report = json.loads(capsys.readouterr().out)
        assert report["tesseract"] is None
        assert report["image_ocr_ready"] is False
        assert report["pdf_ocr_ready"] is False

    def test_help_returns_ok(self, capsys):
        assert et.main(["--help"]) == et.EXIT_OK
        assert "Tesseract" in capsys.readouterr().out

    def test_run_rejects_non_list_command(self):
        with pytest.raises(et.OcrError):
            et._run("tesseract a.png", timeout=5)

    def test_help_names_production_pipeline_for_batch_ocr(self, capsys):
        assert et.main(["--help"]) == et.EXIT_OK
        help_text = capsys.readouterr().out
        assert "production_ocr_pipeline.py" in help_text
        assert "Quick/single-file" in help_text
