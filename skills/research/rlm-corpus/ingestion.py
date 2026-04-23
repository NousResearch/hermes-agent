#!/usr/bin/env python3
"""RLM-Corpus ingestion pipeline.

Converts a directory of PDF/Markdown/LaTeX/plain-text documents into a
structured JSON cache that can be loaded into an RLM REPL session.

CLI:
    python ingestion.py ingest \\
        --source ~/physics/svt-corpus/ \\
        --cache ~/.hermes/rlm-cache/svt-corpus/ \\
        --backend auto \\
        --workers 4

The command is idempotent: files whose SHA-256 hash has not changed since
the last run are skipped. Failures are recorded in `_ingest_errors.json`
inside the cache directory; a single bad file never aborts the batch.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger("rlm_corpus.ingest")

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".tex", ".txt"}
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Data shapes (plain dicts are written to disk; dataclasses are internal)
# ---------------------------------------------------------------------------


@dataclass
class Section:
    heading: str
    level: int
    text: str
    start_char: int
    end_char: int


@dataclass
class Reference:
    raw: str
    key: str | None = None


@dataclass
class DocumentMetadata:
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    source_type: str = "txt"


@dataclass
class IngestedDocument:
    file_path: str
    file_hash: str
    ingested_at: str
    schema_version: int
    metadata: DocumentMetadata
    full_text: str
    sections: list[Section]
    references: list[Reference]
    stats: dict[str, int]


# ---------------------------------------------------------------------------
# Hashing / cache layout
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def cache_filename(source_path: Path, source_root: Path) -> str:
    """Produce a stable, unique filename for a source document's JSON record.

    Uses the relative path under `source_root` with separators mashed into
    underscores so multiple files named `paper.pdf` in different subdirs
    don't collide.
    """
    try:
        rel = source_path.relative_to(source_root)
    except ValueError:
        rel = Path(source_path.name)
    flat = str(rel).replace(os.sep, "__").replace("/", "__")
    return f"{flat}.json"


# ---------------------------------------------------------------------------
# PDF extraction backends
# ---------------------------------------------------------------------------


def _extract_with_pymupdf(path: Path) -> str:
    import pymupdf  # type: ignore

    doc = pymupdf.open(path)
    try:
        return "\n\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _extract_with_marker(path: Path) -> str:
    # marker-pdf's API has shifted a few times; wrap the current public entry.
    from marker.converters.pdf import PdfConverter  # type: ignore
    from marker.models import create_model_dict  # type: ignore
    from marker.output import text_from_rendered  # type: ignore

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(path))
    text, _, _ = text_from_rendered(rendered)
    return text


def _extract_with_pypdf(path: Path) -> str:
    import pypdf  # type: ignore

    reader = pypdf.PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts)


PDF_BACKENDS: dict[str, Callable[[Path], str]] = {
    "marker": _extract_with_marker,
    "pymupdf": _extract_with_pymupdf,
    "pypdf": _extract_with_pypdf,
}


def _backend_chain(preferred: str) -> list[str]:
    """Return an ordered list of backend names to try.

    `auto` uses pymupdf first (lightweight, reliable, preserves most layout)
    then marker (heavier, better for math-heavy docs) then pypdf as a last
    resort.
    """
    preferred = preferred.lower()
    if preferred == "auto":
        return ["pymupdf", "marker", "pypdf"]
    # Put the user's preference first, then the others as fallbacks.
    rest = [b for b in ("pymupdf", "marker", "pypdf") if b != preferred]
    return [preferred, *rest]


def extract_pdf_text(path: Path, backend: str = "auto") -> tuple[str, str]:
    """Extract text from a PDF.

    Returns (text, backend_used). Raises RuntimeError if every backend fails.
    """
    errors: list[str] = []
    for name in _backend_chain(backend):
        fn = PDF_BACKENDS[name]
        try:
            text = fn(path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except ImportError as exc:
            errors.append(f"{name}: not installed ({exc})")
            continue
        except BaseException as exc:  # noqa: BLE001 -- includes pyo3 PanicException
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if text and text.strip():
            return text, name
        errors.append(f"{name}: produced empty text")
    raise RuntimeError(
        "All PDF backends failed for "
        f"{path}:\n  - " + "\n  - ".join(errors)
    )


# ---------------------------------------------------------------------------
# Structure / metadata parsing
# ---------------------------------------------------------------------------


YEAR_RE = re.compile(r"(19|20)\d{2}")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TEX_HEADING_RE = re.compile(
    r"\\(chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]*)\}"
)

# A generous PDF-heading detector. We require the line to be short, mostly
# printable, and either numbered (`2.1 Foo`) or title-cased. Long paragraphs
# of body text are excluded by the length and word-count gate.
PDF_HEADING_RE = re.compile(
    r"""
    ^                              # start of line
    (?:                            # optional numeric prefix
        (?:\d+(?:\.\d+){0,3})      #   e.g. "2.3.1"
        \s*[.)]?\s*
    )?
    (                              # the heading text itself (captured)
        [A-Z][^\n]{0,120}
    )
    $
    """,
    re.MULTILINE | re.VERBOSE,
)

REFERENCES_HEADERS = (
    "references",
    "bibliography",
    "works cited",
)


def _parse_markdown_sections(text: str) -> list[Section]:
    matches = list(MD_HEADING_RE.finditer(text))
    if not matches:
        return []
    sections: list[Section] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(2).strip()
        level = len(m.group(1))
        sections.append(Section(heading, level, text[start:end].strip(), start, end))
    return sections


def _parse_latex_sections(text: str) -> list[Section]:
    matches = list(TEX_HEADING_RE.finditer(text))
    if not matches:
        return []
    level_map = {
        "chapter": 1,
        "section": 1,
        "subsection": 2,
        "subsubsection": 3,
        "paragraph": 4,
        "subparagraph": 5,
    }
    sections: list[Section] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(2).strip()
        level = level_map.get(m.group(1).lower(), 1)
        sections.append(Section(heading, level, text[start:end].strip(), start, end))
    return sections


def _looks_like_heading(line: str) -> bool:
    """Heuristic: is this line plausibly a section heading in a PDF?"""
    s = line.strip()
    if not s or len(s) > 120:
        return False
    words = s.split()
    if len(words) > 15:
        return False
    # Ends with sentence punctuation → probably a sentence, not a heading.
    if s[-1] in ".?!;:,":
        return False
    # Numeric-prefixed headings ("2.1 Methods", "3 Results")
    if re.match(r"^\d+(\.\d+){0,3}\s+\S", s):
        return True
    # All-caps or title-cased short lines
    if s.isupper():
        return True
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha_words:
        return False
    titled = sum(1 for w in alpha_words if w[0].isupper())
    return titled / max(1, len(alpha_words)) >= 0.6 and len(words) <= 10


def _parse_pdf_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    candidates: list[tuple[int, str, int]] = []
    char_offset = 0
    for line in lines:
        line_len = len(line) + 1
        if _looks_like_heading(line):
            candidates.append((char_offset, line.strip(), 1))
        char_offset += line_len

    if not candidates:
        return []

    sections: list[Section] = []
    for i, (start, heading, level) in enumerate(candidates):
        end = candidates[i + 1][0] if i + 1 < len(candidates) else len(text)
        sections.append(Section(heading, level, text[start:end].strip(), start, end))
    return sections


def parse_sections(text: str, source_type: str) -> list[Section]:
    parser = {
        "md": _parse_markdown_sections,
        "tex": _parse_latex_sections,
        "pdf": _parse_pdf_sections,
        "txt": lambda _t: [],
    }.get(source_type, lambda _t: [])
    sections = parser(text)
    if sections:
        return sections
    # Graceful fallback: the whole document is one "Body" section.
    return [Section("Body", 1, text.strip(), 0, len(text))]


def extract_references(sections: list[Section], text: str) -> list[Reference]:
    """Pull a references block if one is identifiable.

    Primary strategy: split on blank lines (typical preprint formatting).
    If that produces a single blob but multiple year-bearing lines exist,
    fall back to one-line-per-reference. This is intentionally shallow —
    the root LM can do better reasoning over the raw text than we can
    via regex.
    """
    for s in sections:
        if s.heading.lower().strip(" .:0123456789") in REFERENCES_HEADERS:
            body = s.text
            first_newline = body.find("\n")
            if first_newline >= 0:
                body = body[first_newline + 1 :]
            blocks = [e.strip() for e in re.split(r"\n\s*\n", body) if e.strip()]
            if len(blocks) <= 1:
                lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
                year_lines = [ln for ln in lines if YEAR_RE.search(ln)]
                if len(year_lines) >= 2:
                    return [Reference(raw=ln) for ln in lines]
            return [Reference(raw=e) for e in blocks]
    return []


def extract_metadata(
    path: Path,
    text: str,
    source_type: str,
) -> DocumentMetadata:
    md = DocumentMetadata(source_type=source_type)

    # Title: first markdown H1, first \title{...}, or filename stem.
    if source_type == "md":
        m = re.search(r"^\#\s+(.+)$", text, re.MULTILINE)
        if m:
            md.title = m.group(1).strip()
    elif source_type == "tex":
        m = re.search(r"\\title\{([^}]+)\}", text)
        if m:
            md.title = m.group(1).strip()
    elif source_type == "pdf":
        # First non-empty line of the document, trimmed.
        for line in text.splitlines():
            if line.strip():
                md.title = line.strip()[:250]
                break
    if not md.title:
        md.title = path.stem.replace("_", " ").replace("-", " ").strip()

    # Authors heuristic for PDFs: line containing commas and "and" near the top.
    if source_type == "pdf":
        head = "\n".join(text.splitlines()[:40])
        for line in head.splitlines():
            s = line.strip()
            if 5 < len(s) < 300 and (" and " in s or "," in s):
                # crude: "Name1, Name2, and Name3"
                names = re.split(r",| and ", s)
                names = [n.strip() for n in names if n.strip() and len(n.strip()) < 80]
                # Must look like names — at least one uppercase letter, no digits, no punctuation tail
                if names and all(
                    re.match(r"^[A-Z][\w'.\-]+(?:\s+[A-Z][\w'.\-]+)+$", n) for n in names
                ):
                    md.authors = names
                    break

    # Year
    year_source = path.stem + " " + text[:2000]
    m = YEAR_RE.search(year_source)
    if m:
        try:
            md.year = int(m.group(0))
        except ValueError:
            md.year = None

    # DOI
    m = DOI_RE.search(text[:8000])
    if m:
        md.doi = m.group(0)

    return md


def estimate_tokens(text: str) -> int:
    # Rough 4 chars/token heuristic. Good enough for UI display.
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def _source_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".md", ".markdown"):
        return "md"
    if ext == ".tex":
        return "tex"
    return "txt"


def ingest_file(
    source_path: Path,
    cache_dir: Path,
    source_root: Path,
    backend: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Process a single document. Returns a status dict for reporting."""
    source_path = source_path.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / cache_filename(source_path, source_root)

    digest = sha256_file(source_path)

    # Incremental skip
    if not force and out_file.exists():
        try:
            with out_file.open() as fh:
                existing = json.load(fh)
            if existing.get("file_hash") == digest and existing.get("schema_version") == SCHEMA_VERSION:
                return {
                    "status": "skipped",
                    "source": str(source_path),
                    "cache_file": str(out_file),
                    "reason": "hash unchanged",
                }
        except (json.JSONDecodeError, OSError):
            pass  # fall through and re-ingest

    source_type = _source_type_for(source_path)
    backend_used = source_type

    if source_type == "pdf":
        text, backend_used = extract_pdf_text(source_path, backend=backend)
    else:
        text = source_path.read_text(encoding="utf-8", errors="replace")

    sections = parse_sections(text, source_type)
    metadata = extract_metadata(source_path, text, source_type)
    references = extract_references(sections, text)

    doc = IngestedDocument(
        file_path=str(source_path),
        file_hash=digest,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        schema_version=SCHEMA_VERSION,
        metadata=metadata,
        full_text=text,
        sections=sections,
        references=references,
        stats={
            "char_count": len(text),
            "token_count_estimate": estimate_tokens(text),
            "section_count": len(sections),
            "reference_count": len(references),
        },
    )

    payload = asdict(doc)
    payload["extraction_backend"] = backend_used

    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp_file.replace(out_file)

    return {
        "status": "ingested",
        "source": str(source_path),
        "cache_file": str(out_file),
        "backend": backend_used,
        "char_count": len(text),
        "section_count": len(sections),
    }


def _ingest_file_safe(args: tuple[Path, Path, Path, str, bool]) -> dict[str, Any]:
    source_path, cache_dir, source_root, backend, force = args
    try:
        return ingest_file(source_path, cache_dir, source_root, backend, force)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 -- keep batch alive, log & continue
        return {
            "status": "error",
            "source": str(source_path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Directory-level orchestration
# ---------------------------------------------------------------------------


def iter_source_files(source_root: Path) -> Iterable[Path]:
    for p in sorted(source_root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part.startswith(".") for part in p.relative_to(source_root).parts):
            continue  # skip dotfiles / hidden dirs
        yield p


def ingest_directory(
    source_root: Path,
    cache_dir: Path,
    backend: str = "auto",
    workers: int = 1,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_source_files(source_root))
    if dry_run:
        return {
            "dry_run": True,
            "source_root": str(source_root),
            "cache_dir": str(cache_dir),
            "candidate_files": [str(p) for p in files],
            "count": len(files),
        }

    results: list[dict[str, Any]] = []
    if workers <= 1 or len(files) <= 1:
        for p in files:
            results.append(_ingest_file_safe((p, cache_dir, source_root, backend, force)))
    else:
        ctx_args = [(p, cache_dir, source_root, backend, force) for p in files]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_ingest_file_safe, ctx_args):
                results.append(r)

    errors = [r for r in results if r.get("status") == "error"]
    if errors:
        (cache_dir / "_ingest_errors.json").write_text(
            json.dumps(
                {
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "errors": errors,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "source_root": str(source_root),
        "cache_dir": str(cache_dir),
        "schema_version": SCHEMA_VERSION,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "total": len(results),
            "ingested": sum(1 for r in results if r["status"] == "ingested"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "errors": len(errors),
        },
    }
    (cache_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return {"manifest": manifest, "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RLM-Corpus ingestion pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a directory into the cache")
    ingest.add_argument("--source", required=True, type=Path)
    ingest.add_argument("--cache", required=True, type=Path)
    ingest.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "pymupdf", "marker", "pypdf"],
        help="Preferred PDF extraction backend (others used as fallbacks)",
    )
    ingest.add_argument("--workers", type=int, default=1)
    ingest.add_argument("--force", action="store_true", help="Re-ingest even if hash matches")
    ingest.add_argument("--dry-run", action="store_true")
    ingest.add_argument("--verbose", "-v", action="store_true")

    sub.add_parser("show-schema", help="Print the JSON schema version")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "show-schema":
        print(f"schema_version={SCHEMA_VERSION}")
        return 0

    if args.command == "ingest":
        if not args.source.exists():
            print(f"error: source does not exist: {args.source}", file=sys.stderr)
            return 2
        result = ingest_directory(
            source_root=args.source,
            cache_dir=args.cache,
            backend=args.backend,
            workers=args.workers,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(json.dumps(result.get("manifest", result), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
