"""Shared page rasterizer with a fallback chain: pypdfium2 -> pdftoppm.

Returns PIL Images so callers can annotate/save. Not a CLI.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def available_backends() -> list[str]:
    """Names of usable rasterizer backends, in preference order."""
    backends = []
    try:
        import pypdfium2  # noqa: F401
        backends.append("pypdfium2")
    except ImportError:
        pass
    if shutil.which("pdftoppm"):
        backends.append("pdftoppm")
    return backends


def missing_hints() -> list[str]:
    """Install hints for when no backend is available."""
    return [
        "python3 -m pip install pypdfium2",
        "poppler-utils (provides pdftoppm), e.g. apt-get install poppler-utils",
    ]


def rasterize_page(pdf_path: str, page: int, dpi: int = 150, password: str | None = None):
    """Render one 1-based page to a PIL Image, or None if no backend works.

    Raises ValueError for an out-of-range page when a backend is present.
    """
    for backend in available_backends():
        if backend == "pypdfium2":
            return _via_pdfium(pdf_path, page, dpi, password)
        if backend == "pdftoppm":
            img = _via_pdftoppm(pdf_path, page, dpi, password)
            if img is not None:
                return img
    return None


def _via_pdfium(pdf_path: str, page: int, dpi: int, password: str | None):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path, password=password)
    try:
        if not 1 <= page <= len(doc):
            raise ValueError(f"page {page} out of range 1-{len(doc)}")
        bitmap = doc[page - 1].render(scale=dpi / 72.0)
        return bitmap.to_pil().convert("RGB")
    finally:
        doc.close()


def _via_pdftoppm(pdf_path: str, page: int, dpi: int, password: str | None):
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "page")
        cmd = ["pdftoppm", "-png", "-r", str(dpi), "-f", str(page), "-l", str(page)]
        if password:
            cmd += ["-upw", password]
        cmd += [pdf_path, prefix]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            raise ValueError(f"pdftoppm failed: {proc.stderr.strip()}")
        produced = sorted(Path(tmp).glob("page*.png"))
        if not produced:
            raise ValueError(f"page {page} out of range (pdftoppm produced no image)")
        with Image.open(produced[0]) as img:
            return img.convert("RGB")
