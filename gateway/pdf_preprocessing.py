"""Fail-closed PDF preprocessing for inbound gateway attachments.

A PDF path is never exposed to the model. It is first copied to a durable inbox,
converted by the configured page-complete converter, and replaced on the event by
the validated Markdown artifact. Failed inputs stay queued for recovery.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_cli.active_sessions import _FileLock
from utils import atomic_json_write


logger = logging.getLogger(__name__)


class PdfPreprocessingError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _magic(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(5)


def _root() -> Path:
    return Path(os.environ.get("HERMES_USER_FILES_DIR", "/root/hermes_user_files")).expanduser().resolve()


def _converter() -> Path:
    return Path(os.environ.get(
        "HERMES_PDF_PREPROCESSOR", "/opt/platform/scripts/pdf_to_markdown_complete.py"
    )).expanduser().resolve()


def _is_pdf(path: Path, media_type: str) -> bool:
    declared_pdf = (
        media_type.lower().split(";", 1)[0].strip() == "application/pdf"
        or path.suffix.lower() == ".pdf"
    )
    try:
        if not path.is_file():
            if declared_pdf:
                raise PdfPreprocessingError(f"declared PDF is unavailable: {path}")
            return False
        pdf_magic = _magic(path) == b"%PDF-"
    except OSError as exc:
        if declared_pdf:
            raise PdfPreprocessingError(f"declared PDF is unreadable: {path}: {exc}") from exc
        return False
    if declared_pdf and not pdf_magic:
        raise PdfPreprocessingError(f"declared PDF has invalid magic bytes: {path}")
    return pdf_magic


def _archive_pending(source: Path, digest: str) -> Path:
    pending_dir = _root() / "pdf-inbox" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    pending = pending_dir / f"{digest}.pdf"
    if not pending.exists() or _sha256(pending) != digest:
        temp = pending.with_suffix(".pdf.part")
        shutil.copy2(source, temp)
        if _sha256(temp) != digest:
            temp.unlink(missing_ok=True)
            raise PdfPreprocessingError("durable PDF archive failed hash verification")
        temp.replace(pending)
    return pending


def _failure_path(pending: Path) -> Path:
    return pending.with_suffix(".failure.json")


def _record_failure(pending: Path, digest: str, error: str) -> None:
    failure = _failure_path(pending)
    attempts = 0
    try:
        prior = json.loads(failure.read_text(encoding="utf-8-sig"))
        attempts = int(prior.get("attempts", 0)) if isinstance(prior, dict) else 0
    except (OSError, ValueError, TypeError):
        pass
    atomic_json_write(failure, {
        "status": "pending_recovery",
        "source_sha256": digest,
        "attempts": attempts + 1,
        "last_error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, fsync_dir=True)


def _validate_result(payload: dict[str, Any], digest: str) -> dict[str, Any]:
    if payload.get("ok") is not True or payload.get("count") != 1:
        raise PdfPreprocessingError("converter did not return one successful result")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise PdfPreprocessingError("converter result list is invalid")
    result = results[0]
    if not isinstance(result, dict):
        raise PdfPreprocessingError("converter result is not an object")
    manifest_path = Path(str(result.get("manifest_path", "")))
    if not manifest_path.is_file():
        raise PdfPreprocessingError("converter manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PdfPreprocessingError("converter manifest is unreadable") from exc
    pages = manifest.get("pages")
    md_path = Path(str(manifest.get("markdown_path", "")))
    copied_source = Path(str(manifest.get("copied_source", "")))
    valid = (
        manifest.get("status") == "complete"
        and manifest.get("all_pages_covered") is True
        and manifest.get("source_sha256") == digest
        and isinstance(manifest.get("page_count"), int)
        and manifest["page_count"] > 0
        and isinstance(pages, list)
        and len(pages) == manifest["page_count"]
        and all(isinstance(page, dict) for page in pages)
        and [p.get("page") for p in pages] == list(range(1, manifest["page_count"] + 1))
        and md_path.is_file()
        and _sha256(md_path) == manifest.get("markdown_sha256")
        and copied_source.is_file()
        and _sha256(copied_source) == digest
        and all(p.get("ocr_attempted") is True for p in pages)
        and all(
            Path(str(p.get("render_path", ""))).is_file()
            and _sha256(Path(str(p["render_path"]))) == p.get("render_sha256")
            for p in pages
        )
    )
    if not valid:
        raise PdfPreprocessingError("converter result failed complete-coverage readback")
    return manifest


def preprocess_pdf_path(source: str) -> dict[str, Any]:
    try:
        source_path = Path(source).expanduser().resolve(strict=True)
        digest = _sha256(source_path)
    except (OSError, RuntimeError) as exc:
        raise PdfPreprocessingError(f"PDF source is unavailable: {exc}") from exc
    lock_path = _root() / "pdf-inbox" / "locks" / f"{digest}.lock"
    with _FileLock(lock_path):
        pending = _archive_pending(source_path, digest)
        converter = _converter()
        if not converter.is_file() or not os.access(converter, os.X_OK):
            error = f"configured PDF converter is unavailable: {converter}"
            _record_failure(pending, digest, error)
            raise PdfPreprocessingError(error)
        last_error = "unknown converter failure"
        for attempt in range(2):
            argv = [str(converter), str(pending)]
            if attempt:
                argv.append("--force")
            try:
                proc = subprocess.run(
                    argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=int(os.environ.get("HERMES_PDF_PREPROCESS_TIMEOUT", "3600")),
                )
                payload = json.loads(proc.stdout or "{}")
                if proc.returncode != 0:
                    detail = proc.stderr.strip()[-1000:]
                    raise PdfPreprocessingError(
                        f"converter exited non-zero{f': {detail}' if detail else ''}"
                    )
                manifest = _validate_result(payload, digest)
                pending.unlink(missing_ok=True)
                _failure_path(pending).unlink(missing_ok=True)
                return manifest
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, PdfPreprocessingError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        _record_failure(pending, digest, last_error)
        raise PdfPreprocessingError(last_error)


def recover_pending_pdfs() -> dict[str, int]:
    """Reprocess the profile's durable PDF queue; safe to call at startup and periodically."""
    pending_dir = _root() / "pdf-inbox" / "pending"
    totals = {"found": 0, "recovered": 0, "failed": 0}
    if not pending_dir.is_dir():
        return totals
    for pending in sorted(pending_dir.glob("*.pdf")):
        totals["found"] += 1
        try:
            preprocess_pdf_path(str(pending))
            totals["recovered"] += 1
        except PdfPreprocessingError:
            totals["failed"] += 1
            logger.warning("Queued PDF recovery remains pending: %s", pending.name)
    return totals


async def preprocess_event_pdfs(event: Any) -> list[dict[str, Any]]:
    urls = list(getattr(event, "media_urls", None) or [])
    types = list(getattr(event, "media_types", None) or [])
    flags = list(getattr(event, "media_text_inlined", None) or [])
    while len(types) < len(urls):
        types.append("")
    while len(flags) < len(urls):
        flags.append(False)
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(urls):
        path = Path(str(raw)).expanduser()
        if not _is_pdf(path, types[index]):
            continue
        original_name = path.name.split("_", 2)[-1]
        manifest = await asyncio.to_thread(preprocess_pdf_path, str(path))
        urls[index] = manifest["markdown_path"]
        types[index] = "text/markdown"
        flags[index] = False
        records.append({
            "original_name": original_name,
            "source_sha256": manifest["source_sha256"],
            "page_count": manifest["page_count"],
            "markdown_path": manifest["markdown_path"],
            "manifest_path": manifest["manifest_path"],
            "all_pages_covered": True,
        })
    event.media_urls = urls
    event.media_types = types
    event.media_text_inlined = flags
    if records:
        metadata = dict(getattr(event, "metadata", None) or {})
        metadata["pdf_preprocessing"] = records
        event.metadata = metadata
    return records
