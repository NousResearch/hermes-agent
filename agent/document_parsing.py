"""Document parsing foundation for local ingestion workflows.

This module provides a normalized parsing interface that Hermes can build on
for future workspace and RAG features. LiteParse is treated as an optional
backend for richer local extraction from PDFs, Office files, and images.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


TEXT_FILE_SUFFIXES = {
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".xml",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".sql",
    ".sh",
}

LITEPARSE_FILE_SUFFIXES = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".svg",
}


class DocumentParsingError(RuntimeError):
    """Raised when a document cannot be parsed with the requested backend."""


class DocumentParserUnavailable(DocumentParsingError):
    """Raised when an optional parser backend is requested but unavailable."""


@dataclass
class ParsedTextItem:
    text: str
    bbox: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None


@dataclass
class ParsedPage:
    page_number: int
    text: str = ""
    items: list[ParsedTextItem] = field(default_factory=list)
    width: Optional[float] = None
    height: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    source_path: str
    parser_backend: str
    text: str
    pages: list[ParsedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentScreenshot:
    page_number: int
    image_path: str
    image_format: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_document_parser_config() -> dict:
    """Load the documents section from Hermes config."""
    try:
        from hermes_cli.config import load_config

        return load_config().get("documents", {}) or {}
    except Exception:
        return {}


def is_text_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TEXT_FILE_SUFFIXES


def supports_liteparse(path: str | Path) -> bool:
    return Path(path).suffix.lower() in LITEPARSE_FILE_SUFFIXES


def liteparse_python_available() -> bool:
    try:
        from liteparse import LiteParse  # noqa: F401

        return True
    except Exception:
        return False


def liteparse_cli_available() -> bool:
    return _find_liteparse_cli() is not None


def liteparse_available() -> bool:
    return liteparse_python_available() or liteparse_cli_available()


def resolve_document_parser_backend(
    path: str | Path,
    backend: Optional[str] = None,
    config: Optional[dict] = None,
) -> str:
    """Resolve the parser backend for the given file."""
    cfg = config if config is not None else load_document_parser_config()
    preferred = (backend or cfg.get("parser_backend") or "auto").strip().lower()

    if preferred not in {"auto", "basic", "liteparse"}:
        raise DocumentParsingError(f"Unknown document parser backend: {preferred}")

    if preferred == "basic":
        return "basic"

    if preferred == "liteparse":
        if not supports_liteparse(path):
            raise DocumentParsingError(
                f"LiteParse backend does not support {Path(path).suffix or 'this file type'}."
            )
        if not liteparse_available():
            raise DocumentParserUnavailable(
                "LiteParse backend requested but neither the Python package nor the `lit` CLI is available."
            )
        return "liteparse"

    if supports_liteparse(path) and liteparse_available():
        return "liteparse"
    return "basic"


def parse_document(
    path: str | Path,
    backend: Optional[str] = None,
    config: Optional[dict] = None,
    parse_options: Optional[dict] = None,
) -> ParsedDocument:
    """Parse *path* into a normalized document structure."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DocumentParsingError(f"Document not found: {source}")

    resolved_backend = resolve_document_parser_backend(source, backend=backend, config=config)
    if resolved_backend == "basic":
        return _parse_with_basic_backend(source)
    return _parse_with_liteparse(source, config=config, parse_options=parse_options)


def create_document_screenshots(
    path: str | Path,
    backend: Optional[str] = None,
    config: Optional[dict] = None,
    screenshot_options: Optional[dict] = None,
) -> list[DocumentScreenshot]:
    """Create page screenshots for a local document using LiteParse."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DocumentParsingError(f"Document not found: {source}")

    resolved_backend = resolve_document_parser_backend(source, backend=backend, config=config)
    if resolved_backend != "liteparse":
        raise DocumentParsingError("Screenshot generation requires the LiteParse backend.")

    return _create_liteparse_screenshots(source, config=config, screenshot_options=screenshot_options)


def _parse_with_basic_backend(path: Path) -> ParsedDocument:
    if not is_text_file(path):
        raise DocumentParsingError(
            f"Basic parser only supports text-like files; got {path.suffix or 'unknown type'}."
        )

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")

    return ParsedDocument(
        source_path=str(path),
        parser_backend="basic",
        text=text,
        pages=[ParsedPage(page_number=1, text=text)] if text else [],
        metadata={"page_count": 1 if text else 0},
    )


def _parse_with_liteparse(
    path: Path,
    config: Optional[dict] = None,
    parse_options: Optional[dict] = None,
) -> ParsedDocument:
    cfg = _liteparse_config(config, overrides=parse_options)
    cli_command = _find_liteparse_cli()
    python_error: Optional[Exception] = None

    if liteparse_python_available():
        try:
            return _parse_with_liteparse_python(path, cfg, cli_command=cli_command)
        except Exception as exc:
            python_error = exc

    if cli_command:
        try:
            return _parse_with_liteparse_cli(path, cfg, cli_command=cli_command)
        except Exception as exc:
            if python_error is not None:
                raise DocumentParsingError(
                    f"LiteParse Python backend failed ({python_error}); CLI fallback also failed ({exc})."
                ) from exc
            raise

    if python_error is not None:
        raise DocumentParsingError(f"LiteParse Python backend failed: {python_error}") from python_error

    raise DocumentParserUnavailable(
        "LiteParse parsing requested but no backend is available. Install `liteparse` or the `lit` CLI."
    )


def _liteparse_config(config: Optional[dict], overrides: Optional[dict] = None) -> dict:
    cfg = config if config is not None else load_document_parser_config()
    liteparse_cfg = cfg.get("liteparse", {}) or {}
    merged = {
        "ocr_enabled": liteparse_cfg.get("ocr_enabled", True),
        "ocr_server_url": liteparse_cfg.get("ocr_server_url") or "",
        "ocr_language": liteparse_cfg.get("ocr_language") or "en",
        "dpi": liteparse_cfg.get("dpi", 150),
        "target_pages": liteparse_cfg.get("target_pages") or "",
        "max_pages": liteparse_cfg.get("max_pages", 10000),
        "no_precise_bbox": bool(liteparse_cfg.get("no_precise_bbox", False)),
        "preserve_small_text": bool(liteparse_cfg.get("preserve_small_text", False)),
        "image_format": liteparse_cfg.get("image_format", "png"),
        "screenshot_output_dir": liteparse_cfg.get("screenshot_output_dir") or "",
    }
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return merged


def _find_liteparse_cli() -> Optional[str]:
    """Find the LiteParse CLI command using Hermes' normal external-tool pattern."""
    which_result = shutil.which("liteparse")
    if which_result:
        return which_result

    repo_root = Path(__file__).resolve().parents[1]
    local_candidates = [
        repo_root / "node_modules" / ".bin" / "liteparse",
        repo_root.parent / "node_modules" / ".bin" / "liteparse",
    ]
    for candidate in local_candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    npx_path = shutil.which("npx")
    if npx_path:
        return "npx liteparse"

    return None


def _create_liteparse_screenshots(
    path: Path,
    config: Optional[dict] = None,
    screenshot_options: Optional[dict] = None,
) -> list[DocumentScreenshot]:
    cfg = _liteparse_config(config, overrides=screenshot_options)
    cli_command = _find_liteparse_cli()
    python_error: Optional[Exception] = None

    if liteparse_python_available():
        try:
            return _create_liteparse_screenshots_python(path, cfg, cli_command=cli_command)
        except Exception as exc:
            python_error = exc

    if cli_command:
        try:
            return _create_liteparse_screenshots_cli(path, cfg, cli_command=cli_command)
        except Exception as exc:
            if python_error is not None:
                raise DocumentParsingError(
                    f"LiteParse Python screenshot backend failed ({python_error}); CLI fallback also failed ({exc})."
                ) from exc
            raise

    if python_error is not None:
        raise DocumentParsingError(f"LiteParse Python screenshot backend failed: {python_error}") from python_error

    raise DocumentParserUnavailable(
        "LiteParse screenshot support requested but no backend is available."
    )


def _parse_with_liteparse_python(path: Path, config: dict, cli_command: Optional[str] = None) -> ParsedDocument:
    from liteparse import LiteParse

    parser = LiteParse(
        cli_path=cli_command,
    )
    result = parser.parse(
        str(path),
        ocr_enabled=config["ocr_enabled"],
        ocr_server_url=config["ocr_server_url"] or None,
        ocr_language=config["ocr_language"],
        max_pages=config["max_pages"],
        target_pages=config["target_pages"] or None,
        dpi=config["dpi"],
        precise_bounding_box=not config["no_precise_bbox"],
        preserve_very_small_text=config["preserve_small_text"],
    )
    return _normalize_liteparse_python_result(path, result)


def _create_liteparse_screenshots_python(
    path: Path,
    config: dict,
    cli_command: Optional[str] = None,
) -> list[DocumentScreenshot]:
    from liteparse import LiteParse

    parser = LiteParse(cli_path=cli_command)
    output_dir = config["screenshot_output_dir"] or tempfile.mkdtemp(prefix="hermes_liteparse_")
    shots = parser.screenshot(
        str(path),
        output_dir=output_dir,
        target_pages=config["target_pages"] or None,
        dpi=config["dpi"],
        image_format=config["image_format"],
    )
    screenshots = []
    for shot in shots:
        page_number = getattr(shot, "page_num", None) or getattr(shot, "pageNum", None)
        image_path = getattr(shot, "image_path", None) or getattr(shot, "path", None)
        screenshots.append(
            DocumentScreenshot(
                page_number=int(page_number),
                image_path=str(image_path),
                image_format=config["image_format"],
            )
        )
    return screenshots


def _parse_with_liteparse_cli(path: Path, config: dict, cli_command: Optional[str] = None) -> ParsedDocument:
    resolved_command = cli_command or _find_liteparse_cli()
    if not resolved_command:
        raise DocumentParserUnavailable("LiteParse CLI not found.")

    cmd = shlex.split(resolved_command) + [
        "parse",
        str(path),
        "--format",
        "json",
        "-q",
        "--ocr-language",
        str(config["ocr_language"]),
        "--dpi",
        str(config["dpi"]),
        "--max-pages",
        str(config["max_pages"]),
    ]
    if not config["ocr_enabled"]:
        cmd.append("--no-ocr")
    if config["ocr_server_url"]:
        cmd.extend(["--ocr-server-url", str(config["ocr_server_url"])])
    if config["target_pages"]:
        cmd.extend(["--target-pages", str(config["target_pages"])])
    if config["no_precise_bbox"]:
        cmd.append("--no-precise-bbox")
    if config["preserve_small_text"]:
        cmd.append("--preserve-small-text")

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise DocumentParsingError(f"`lit parse` failed: {stderr}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DocumentParsingError(f"LiteParse CLI returned invalid JSON: {exc}") from exc

    return _normalize_liteparse_cli_result(path, payload)


def _create_liteparse_screenshots_cli(
    path: Path,
    config: dict,
    cli_command: Optional[str] = None,
) -> list[DocumentScreenshot]:
    resolved_command = cli_command or _find_liteparse_cli()
    if not resolved_command:
        raise DocumentParserUnavailable("LiteParse CLI not found.")

    output_dir = Path(config["screenshot_output_dir"] or tempfile.mkdtemp(prefix="hermes_liteparse_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = shlex.split(resolved_command) + [
        "screenshot",
        str(path),
        "-o",
        str(output_dir),
        "--format",
        str(config["image_format"]),
        "--dpi",
        str(config["dpi"]),
        "-q",
    ]
    if config["target_pages"]:
        cmd.extend(["--target-pages", str(config["target_pages"])])

    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise DocumentParsingError(f"`liteparse screenshot` failed: {stderr}")

    suffix = f".{config['image_format']}"
    screenshots = []
    for image_file in sorted(output_dir.glob(f"*{suffix}")):
        page_num = _extract_page_number(image_file.stem)
        if page_num is None:
            continue
        screenshots.append(
            DocumentScreenshot(
                page_number=page_num,
                image_path=str(image_file),
                image_format=config["image_format"],
            )
        )
    return screenshots


def _normalize_liteparse_python_result(path: Path, result: Any) -> ParsedDocument:
    pages: list[ParsedPage] = []
    raw_pages = getattr(result, "pages", None) or []
    for index, raw_page in enumerate(raw_pages, start=1):
        page = _normalize_python_page(raw_page, fallback_page_number=index)
        pages.append(page)

    text = getattr(result, "text", None)
    if not isinstance(text, str):
        text = "\n\n".join(page.text for page in pages if page.text).strip()

    return ParsedDocument(
        source_path=str(path),
        parser_backend="liteparse-python",
        text=text or "",
        pages=pages,
        metadata={"page_count": len(pages)},
    )


def _normalize_python_page(raw_page: Any, fallback_page_number: int) -> ParsedPage:
    page_number = (
        getattr(raw_page, "pageNum", None)
        or getattr(raw_page, "page_number", None)
        or getattr(raw_page, "page", None)
        or fallback_page_number
    )
    items = []
    raw_items = getattr(raw_page, "textItems", None) or getattr(raw_page, "text_items", None) or []
    for raw_item in raw_items:
        item_text = str(getattr(raw_item, "text", None) or getattr(raw_item, "str", "") or "")
        bbox = _coerce_bbox(getattr(raw_item, "bbox", None)) or _coerce_item_box(raw_item)
        confidence = getattr(raw_item, "confidence", None)
        items.append(ParsedTextItem(text=item_text, bbox=bbox, confidence=confidence))

    page_text = getattr(raw_page, "text", None)
    if not isinstance(page_text, str):
        page_text = "\n".join(item.text for item in items if item.text)

    raw_bboxes = getattr(raw_page, "boundingBoxes", None) or getattr(raw_page, "bounding_boxes", None) or []
    bounding_boxes = [bbox for bbox in (_coerce_bbox(raw_bbox) for raw_bbox in raw_bboxes) if bbox]

    return ParsedPage(
        page_number=int(page_number),
        text=page_text or "",
        items=items,
        width=_coerce_float(getattr(raw_page, "width", None)),
        height=_coerce_float(getattr(raw_page, "height", None)),
        metadata={"bounding_boxes": bounding_boxes},
    )


def _normalize_liteparse_cli_result(path: Path, payload: Dict[str, Any]) -> ParsedDocument:
    raw_pages = payload.get("pages", []) or payload.get("json", {}).get("pages", []) or []
    pages = [
        _normalize_cli_page(raw_page, fallback_page_number=index)
        for index, raw_page in enumerate(raw_pages, start=1)
    ]
    text = payload.get("text")
    if not isinstance(text, str):
        text = "\n\n".join(page.text for page in pages if page.text).strip()

    return ParsedDocument(
        source_path=str(path),
        parser_backend="liteparse-cli",
        text=text or "",
        pages=pages,
        metadata={"page_count": len(pages)},
    )


def _normalize_cli_page(raw_page: Dict[str, Any], fallback_page_number: int) -> ParsedPage:
    raw_items = raw_page.get("textItems") or raw_page.get("text_items") or []
    items = []
    for raw_item in raw_items:
        items.append(
            ParsedTextItem(
                text=str(raw_item.get("text", raw_item.get("str", "")) or ""),
                bbox=_coerce_bbox(raw_item.get("bbox")) or _coerce_item_box(raw_item),
                confidence=_coerce_float(raw_item.get("confidence")),
            )
        )

    page_text = raw_page.get("text")
    if not isinstance(page_text, str):
        page_text = "\n".join(item.text for item in items if item.text)
    raw_bboxes = raw_page.get("boundingBoxes") or raw_page.get("bounding_boxes") or []
    bounding_boxes = [bbox for bbox in (_coerce_bbox(raw_bbox) for raw_bbox in raw_bboxes) if bbox]

    return ParsedPage(
        page_number=int(
            raw_page.get("pageNum")
            or raw_page.get("page_number")
            or raw_page.get("page")
            or fallback_page_number
        ),
        text=page_text or "",
        items=items,
        width=_coerce_float(raw_page.get("width")),
        height=_coerce_float(raw_page.get("height")),
        metadata={"bounding_boxes": bounding_boxes},
    )


def _coerce_bbox(value: Any) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(key in value for key in keys):
            return {key: float(value[key]) for key in keys}
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x1, y1, x2, y2 = value
        return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_page_number(stem: str) -> Optional[int]:
    if stem.startswith("page_"):
        try:
            return int(stem.replace("page_", ""))
        except ValueError:
            return None
    return None


def _coerce_item_box(value: Any) -> Optional[Dict[str, float]]:
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
        width = value.get("width", value.get("w"))
        height = value.get("height", value.get("h"))
    else:
        x = getattr(value, "x", None)
        y = getattr(value, "y", None)
        width = getattr(value, "width", None)
        if width is None:
            width = getattr(value, "w", None)
        height = getattr(value, "height", None)
        if height is None:
            height = getattr(value, "h", None)

    if None in (x, y, width, height):
        return None
    return {
        "x1": float(x),
        "y1": float(y),
        "x2": float(x) + float(width),
        "y2": float(y) + float(height),
    }
