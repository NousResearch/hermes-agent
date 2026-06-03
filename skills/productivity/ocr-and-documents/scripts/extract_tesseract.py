#!/usr/bin/env python3
"""Optional local/offline OCR backend using the system Tesseract binary.

A lightweight alternative to marker-pdf: no PyTorch, no model downloads, no network.
It shells out to the locally installed `tesseract` binary, which is NEVER installed
automatically by this script. For scanned PDFs it also needs a rasterizer: `pdftoppm`
(from poppler-utils) or, as a fallback, the `pymupdf` package if it is already
importable in the environment.

Where it fits among the sibling extractors:
  * extract_pymupdf.py  -> text-based PDFs (instant, ~25MB, no OCR)
  * extract_tesseract.py -> local/offline OCR for images & scanned PDFs (this file)
  * extract_marker.py   -> high-quality OCR/layout, but ~3-5GB PyTorch + models

Design / security notes (see SKILL.md "Security limits"):
  * Every external command runs via subprocess with shell=False and an explicit arg
    list -- no shell string is ever constructed, so paths/langs cannot inject.
  * Every invocation has a timeout.
  * Languages are validated against an allow-pattern AND against `tesseract
    --list-langs`; unknown/garbage language specs are rejected before any OCR runs.
  * File-size, page-count and DPI limits guard against resource exhaustion. They are
    overridable via env vars (HERMES_OCR_*) for operators who need different bounds.
  * Text-based PDFs are PREFERRED: when a text layer is detected (via pymupdf, when
    available) that text is returned and OCR is skipped unless --force-ocr is given.
  * Missing dependencies and failures produce structured output (dict / JSON), never a
    bare traceback, and map to stable non-zero exit codes.

Usage:
    Quick/single-file helper:
      python extract_tesseract.py image.png                  # OCR an image -> stdout
      python extract_tesseract.py image.png --lang ces+eng   # explicit mixed CZ/EN
      python extract_tesseract.py scan.pdf                   # text layer first, OCR if none
      python extract_tesseract.py scan.pdf --force-ocr       # always OCR every page
      python extract_tesseract.py doc.pdf --json             # structured JSON result
      python extract_tesseract.py --check                    # report dependency status
      python extract_tesseract.py --list-langs               # installed OCR languages

    Spearhead batch/canonical OCR:
      use production_ocr_pipeline.py for directories, manifests, confidence metrics,
      TSV/hOCR/ALTO/PAGE outputs, PSM retry matrix, and review flags.

Exit codes:
    0 ok | 1 generic failure | 2 missing dependency | 3 language error
    4 timeout | 5 resource limit exceeded | 6 bad input (file not found, etc.)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ----- binaries (resolved on PATH; never auto-installed) -----
TESSERACT_BIN = "tesseract"
PDFTOPPM_BIN = "pdftoppm"

# ----- default security limits (override via HERMES_OCR_* env vars) -----
DEFAULT_LANG = "ces+eng"
DEFAULT_TIMEOUT_S = 120          # per external command
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB input cap
MAX_PDF_PAGES = 50               # max pages rasterized/OCR'd from one PDF
MAX_IMAGES = 200                 # hard cap on images OCR'd in one PDF run
RASTER_DPI = 300                 # Spearhead final-ish OCR default; lower only for triage

# Tesseract language tokens look like: eng, ces, deu, chi_sim, osd ...
# A spec is one or more tokens joined by '+'. Anything else is rejected outright
# so a hostile/garbage "language" can never reach the command line.
_LANG_TOKEN = r"[A-Za-z]{2,}(?:_[A-Za-z]+)*"
LANG_ARG_RE = re.compile(rf"^{_LANG_TOKEN}(?:\+{_LANG_TOKEN})*$")

# ----- stable exit codes -----
EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_MISSING_DEP = 2
EXIT_LANGUAGE = 3
EXIT_TIMEOUT = 4
EXIT_LIMIT = 5
EXIT_INPUT = 6

_APT_HINT = "Install with your OS package manager, e.g. `apt-get install tesseract-ocr` (no sudo/auto-install is performed here)."
_RASTER_HINT = "Install poppler-utils (provides pdftoppm) or the `pymupdf` Python package."


class OcrError(Exception):
    """Structured OCR failure carrying a stable error_type and process exit code."""

    def __init__(self, error_type: str, message: str, exit_code: int = EXIT_GENERIC):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.exit_code = exit_code


# --------------------------------------------------------------------------- #
# subprocess wrapper (single choke point -- tests monkeypatch this)
# --------------------------------------------------------------------------- #
def _run(cmd, timeout):
    """Run an explicit-arg command with shell=False. Never accepts a shell string.

    Translates the two failure modes we care about into OcrError so callers never
    see a raw FileNotFoundError / TimeoutExpired.
    """
    if not isinstance(cmd, (list, tuple)) or not cmd:
        raise OcrError("internal", "command must be a non-empty arg list", EXIT_GENERIC)
    try:
        return subprocess.run(
            list(cmd),
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise OcrError(
            "missing_dependency", f"Executable not found: {cmd[0]}. {_APT_HINT}", EXIT_MISSING_DEP
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise OcrError(
            "timeout", f"{cmd[0]} timed out after {timeout}s", EXIT_TIMEOUT
        ) from exc


def _env_int(name, default):
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError:
        return default
    return val if val > 0 else default


def _pymupdf_available():
    try:
        import pymupdf  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# dependency detection
# --------------------------------------------------------------------------- #
def find_binary(name):
    """Return the resolved path of a binary on PATH, or None. Never installs."""
    return shutil.which(name)


def require_binary(name, hint):
    path = find_binary(name)
    if not path:
        raise OcrError(
            "missing_dependency",
            f"Required binary '{name}' not found on PATH. {hint}",
            EXIT_MISSING_DEP,
        )
    return path


# --------------------------------------------------------------------------- #
# languages
# --------------------------------------------------------------------------- #
def list_languages(tesseract_bin=None, timeout=30):
    """Return the OCR languages installed for the local tesseract."""
    tesseract_bin = tesseract_bin or require_binary(TESSERACT_BIN, _APT_HINT)
    proc = _run([tesseract_bin, "--list-langs"], timeout=timeout)
    langs = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("list of"):
            continue
        langs.append(line)
    return langs


def validate_language(lang, available=None, tesseract_bin=None):
    """Validate a language spec against the allow-pattern and installed languages.

    Returns the list of requested language tokens. Raises OcrError on a malformed
    spec ('invalid_language') or a not-installed language ('missing_language').
    """
    if not lang or not LANG_ARG_RE.match(lang):
        raise OcrError(
            "invalid_language",
            f"Invalid language spec: {lang!r}. Use tokens like 'eng' or 'eng+ces'.",
            EXIT_LANGUAGE,
        )
    requested = lang.split("+")
    if available is None:
        available = list_languages(tesseract_bin)
    missing = [token for token in requested if token not in available]
    if missing:
        raise OcrError(
            "missing_language",
            f"Language(s) not installed: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(available)) or '(none)'}.",
            EXIT_LANGUAGE,
        )
    return requested


# --------------------------------------------------------------------------- #
# OCR an image
# --------------------------------------------------------------------------- #
def build_ocr_command(tesseract_bin, image_path, lang, extra_config=None):
    """Build the explicit tesseract arg list (shell=False). Output goes to stdout."""
    cmd = [str(tesseract_bin), str(image_path), "stdout", "-l", str(lang)]
    if extra_config:
        cmd.extend(str(arg) for arg in extra_config)
    return cmd


def ocr_image(
    image_path,
    lang=DEFAULT_LANG,
    timeout=DEFAULT_TIMEOUT_S,
    max_file_size=MAX_FILE_SIZE_BYTES,
    tesseract_bin=None,
    available_langs=None,
    _check_size=True,
):
    """OCR a single image file and return the recognized text.

    `_check_size` is disabled for internally-generated raster pages whose size we
    already control.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise OcrError("input_not_found", f"File not found: {image_path}", EXIT_INPUT)
    if _check_size:
        size = image_path.stat().st_size
        if size > max_file_size:
            raise OcrError(
                "file_too_large",
                f"{image_path} is {size} bytes (> limit {max_file_size}).",
                EXIT_LIMIT,
            )
    tesseract_bin = tesseract_bin or require_binary(TESSERACT_BIN, _APT_HINT)
    if available_langs is None:
        available_langs = list_languages(tesseract_bin)
    validate_language(lang, available=available_langs)
    proc = _run(build_ocr_command(tesseract_bin, image_path, lang), timeout=timeout)
    if proc.returncode != 0:
        raise OcrError(
            "ocr_failed",
            f"tesseract exited {proc.returncode}: {(proc.stderr or '').strip()[:500]}",
            EXIT_GENERIC,
        )
    return proc.stdout


# --------------------------------------------------------------------------- #
# PDF handling (text-first, OCR fallback)
# --------------------------------------------------------------------------- #
def _page_sort_key(path: Path):
    try:
        return (int(path.stem.rsplit("-", 1)[-1]), path.name)
    except ValueError:
        return (10**9, path.name)


def detect_pdf_text(pdf_path, max_pages=MAX_PDF_PAGES):
    """Return the embedded text layer if the PDF has a usable one, else None.

    Returns None when pymupdf is not importable -- in that case the caller cannot
    cheaply know whether a text layer exists and must decide whether to OCR.
    """
    try:
        import pymupdf  # type: ignore
    except Exception:
        return None
    parts = []
    doc = pymupdf.open(str(pdf_path))
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            parts.append(page.get_text())
    finally:
        doc.close()
    joined = "".join(parts).strip()
    return joined or None


def rasterize_pdf(pdf_path, out_dir, dpi=RASTER_DPI, max_pages=MAX_PDF_PAGES, timeout=DEFAULT_TIMEOUT_S):
    """Rasterize PDF pages to PNG files in out_dir.

    Prefers pdftoppm (poppler); falls back to pymupdf if it is importable. Raises
    OcrError('missing_dependency') when neither rasterizer is available.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdftoppm = find_binary(PDFTOPPM_BIN)
    if pdftoppm:
        prefix = str(out_dir / "page")
        cmd = [
            pdftoppm, "-png",
            "-r", str(dpi),
            "-f", "1",
            "-l", str(max_pages),
            str(pdf_path), prefix,
        ]
        proc = _run(cmd, timeout=timeout)
        if proc.returncode != 0:
            raise OcrError(
                "rasterize_failed",
                f"pdftoppm exited {proc.returncode}: {(proc.stderr or '').strip()[:500]}",
                EXIT_GENERIC,
            )
        images = sorted(out_dir.glob("page*.png"), key=_page_sort_key)
        if not images:
            raise OcrError("rasterize_failed", "pdftoppm produced no images.", EXIT_GENERIC)
        return images[:max_pages]

    # Fallback: existing project rasterization path via pymupdf, if present.
    try:
        import pymupdf  # type: ignore
    except Exception as exc:
        raise OcrError(
            "missing_dependency",
            f"No PDF rasterizer available. {_RASTER_HINT}",
            EXIT_MISSING_DEP,
        ) from exc
    images = []
    doc = pymupdf.open(str(pdf_path))
    try:
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix)
            out = out_dir / f"page-{i + 1:04d}.png"
            pix.save(str(out))
            images.append(out)
    finally:
        doc.close()
    if not images:
        raise OcrError("rasterize_failed", "pymupdf produced no images.", EXIT_GENERIC)
    return images


def ocr_pdf(
    pdf_path,
    lang=DEFAULT_LANG,
    force_ocr=False,
    timeout=DEFAULT_TIMEOUT_S,
    max_file_size=MAX_FILE_SIZE_BYTES,
    max_pages=MAX_PDF_PAGES,
    dpi=RASTER_DPI,
    tesseract_bin=None,
    available_langs=None,
):
    """Extract text from a PDF, preferring the embedded text layer over OCR.

    Returns a dict: {"source": "text-layer"|"ocr", "text": str, "pages_ocred": int}.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise OcrError("input_not_found", f"File not found: {pdf_path}", EXIT_INPUT)
    size = pdf_path.stat().st_size
    if size > max_file_size:
        raise OcrError(
            "file_too_large",
            f"{pdf_path} is {size} bytes (> limit {max_file_size}).",
            EXIT_LIMIT,
        )

    # Prefer the existing text layer; only OCR when forced or when none is found.
    if not force_ocr:
        text = detect_pdf_text(pdf_path, max_pages=max_pages)
        if text:
            return {"source": "text-layer", "text": text, "pages_ocred": 0}

    # OCR path: resolve + validate dependencies and language up front (fail fast).
    tesseract_bin = tesseract_bin or require_binary(TESSERACT_BIN, _APT_HINT)
    if available_langs is None:
        available_langs = list_languages(tesseract_bin)
    validate_language(lang, available=available_langs)

    with tempfile.TemporaryDirectory(prefix="hermes_ocr_") as tmp:
        images = rasterize_pdf(pdf_path, tmp, dpi=dpi, max_pages=max_pages, timeout=timeout)
        if len(images) > MAX_IMAGES:
            raise OcrError(
                "too_many_pages",
                f"{len(images)} pages exceeds limit {MAX_IMAGES}.",
                EXIT_LIMIT,
            )
        parts = []
        for img in images:
            parts.append(
                ocr_image(
                    img,
                    lang=lang,
                    timeout=timeout,
                    max_file_size=max_file_size,
                    tesseract_bin=tesseract_bin,
                    available_langs=available_langs,
                    _check_size=False,
                )
            )
        return {"source": "ocr", "text": "\n".join(parts), "pages_ocred": len(images)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def check():
    """Report dependency status without running OCR. Safe to call when deps missing."""
    tesseract = find_binary(TESSERACT_BIN)
    pdftoppm = find_binary(PDFTOPPM_BIN)
    pymupdf = _pymupdf_available()
    report = {
        "tesseract": tesseract,
        "pdftoppm": pdftoppm,
        "pymupdf": pymupdf,
        "languages": list_languages(tesseract) if tesseract else [],
        "image_ocr_ready": bool(tesseract),
        "pdf_ocr_ready": bool(tesseract) and (bool(pdftoppm) or pymupdf),
    }
    return report


def _emit_error(err, as_json):
    if as_json:
        print(json.dumps({"ok": False, "error_type": err.error_type, "error": err.message}, ensure_ascii=False), file=sys.stderr)
    else:
        print(f"ERROR [{err.error_type}]: {err.message}", file=sys.stderr)
    return err.exit_code


def _arg_value(argv, flag, default=None):
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return EXIT_OK

    as_json = "--json" in argv

    try:
        if argv[0] == "--check":
            print(json.dumps(check(), indent=2, ensure_ascii=False))
            return EXIT_OK
        if argv[0] == "--list-langs":
            print(json.dumps(list_languages(), ensure_ascii=False))
            return EXIT_OK

        path = argv[0]
        lang = _arg_value(argv, "--lang", DEFAULT_LANG)
        force_ocr = "--force-ocr" in argv
        timeout = _env_int("HERMES_OCR_TIMEOUT_S", DEFAULT_TIMEOUT_S)
        max_file_size = _env_int("HERMES_OCR_MAX_FILE_BYTES", MAX_FILE_SIZE_BYTES)
        max_pages = _env_int("HERMES_OCR_MAX_PAGES", MAX_PDF_PAGES)
        dpi = _env_int("HERMES_OCR_DPI", RASTER_DPI)

        if Path(path).suffix.lower() == ".pdf":
            result = ocr_pdf(
                path, lang=lang, force_ocr=force_ocr, timeout=timeout,
                max_file_size=max_file_size, max_pages=max_pages, dpi=dpi,
            )
        else:
            text = ocr_image(path, lang=lang, timeout=timeout, max_file_size=max_file_size)
            result = {"source": "ocr", "text": text, "pages_ocred": 1}
    except OcrError as err:
        return _emit_error(err, as_json)

    if as_json:
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    else:
        print(result["text"])
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
