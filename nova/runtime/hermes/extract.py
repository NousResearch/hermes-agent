"""Document extraction, delegated to the runtime's own extractor.

The runtime ships a mature extractor covering PDF, legacy Office, OpenDocument, RTF and
EPUB, with typed errors for malformed input. Re-implementing that would be weeks of work
and a permanent maintenance liability, so NOVA uses it rather than competing with it.

**The import is deliberately lazy.** ``nova`` must load and test with only the standard
library and PyYAML present; importing the runtime at module scope would break that. Inside
this function the runtime is, by definition, installed — this code only ever runs against a
live one. ``tests/platform/test_boundaries.py`` enforces exactly that distinction: no
module-level runtime import anywhere, lazy ones permitted only inside an adapter package.
"""

from __future__ import annotations

from pathlib import Path

from nova.runtime.base import ExtractedDocument

#: Read directly as text without invoking the extractor. Cheaper, and it keeps plain
#: documents working on an installation whose optional extraction deps are missing.
NATIVE_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".log"}
)


def extract(path: Path) -> ExtractedDocument:
    """Text from ``path``, using the runtime's extractor for formats that need it."""
    suffix = path.suffix.lower()

    if suffix in NATIVE_SUFFIXES:
        return _native(path)

    try:
        # Lazy by design — see the module docstring.
        from tools.read_extract import ExtractionError, extract_document_bytes
    except ImportError:
        result = _native(path)
        if result.extracted:
            return result
        return ExtractedDocument(
            path=str(path),
            text="",
            extracted=False,
            method="unavailable",
            detail=(
                f"{suffix or 'this file'} needs the runtime's document extractor, which is "
                "not importable here"
            ),
        )

    try:
        text = extract_document_bytes(path.read_bytes(), path.name)
    except ExtractionError as exc:
        return ExtractedDocument(
            path=str(path), text="", extracted=False, method="runtime",
            detail=f"document could not be extracted: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 — one bad file must not stop an ingest
        return ExtractedDocument(
            path=str(path), text="", extracted=False, method="runtime",
            detail=f"extractor failed: {type(exc).__name__}: {exc}",
        )

    if not isinstance(text, str) or not text.strip():
        return ExtractedDocument(
            path=str(path), text="", extracted=False, method="runtime",
            detail="extractor returned no text (an image-only PDF needs OCR)",
        )
    return ExtractedDocument(path=str(path), text=text, extracted=True, method="runtime")


def _native(path: Path) -> ExtractedDocument:
    try:
        return ExtractedDocument(
            path=str(path), text=path.read_text(encoding="utf-8"), extracted=True, method="native"
        )
    except (OSError, UnicodeDecodeError) as exc:
        return ExtractedDocument(
            path=str(path), text="", extracted=False, method="native",
            detail=f"not readable as UTF-8 text: {exc}",
        )
