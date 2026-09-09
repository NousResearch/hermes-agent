"""Project-scoped Markdown file listing and preview routes.

Mounted at ``/api/plugins/markdown-viewer/`` by the dashboard plugin system.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query


router = APIRouter()

MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
}
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_LIST_FILES = 2_000
MANAGED_FILES_ROOT_ENV = "HERMES_DASHBOARD_FILES_ROOT"
HOSTED_MANAGED_FILES_ROOT = Path("/opt/data")


class MarkdownViewerError(ValueError):
    """A safe, user-facing Markdown viewer error."""


def _locked_files_root() -> Path | None:
    raw = os.environ.get(MANAGED_FILES_ROOT_ENV, "").strip()
    if raw:
        try:
            return Path(raw).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise MarkdownViewerError("The managed files root is unavailable.") from exc

    raw_home = os.environ.get("HERMES_HOME", "").strip()
    if raw_home:
        try:
            home = Path(raw_home).expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise MarkdownViewerError("The Hermes home path is invalid.") from exc
        if home == HOSTED_MANAGED_FILES_ROOT:
            return HOSTED_MANAGED_FILES_ROOT
    return None


def _resolve_root(root: str) -> Path:
    raw = str(root or "").strip()
    if not raw or "\x00" in raw:
        raise MarkdownViewerError("A valid Project root is required.")
    try:
        resolved = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MarkdownViewerError(f"Project root is unavailable: {raw}") from exc
    if not resolved.is_dir():
        raise MarkdownViewerError(f"Project root is not a directory: {resolved}")
    locked_root = _locked_files_root()
    if locked_root is not None and resolved != locked_root and locked_root not in resolved.parents:
        raise MarkdownViewerError("Project root is outside the managed files root.")
    return resolved


def _resolve_markdown(root: Path, requested: str) -> Path:
    raw = str(requested or "").strip()
    if not raw or "\x00" in raw:
        raise MarkdownViewerError("Choose a Markdown file.")
    unresolved = Path(raw).expanduser()
    candidate = unresolved if unresolved.is_absolute() else root / unresolved
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MarkdownViewerError(f"Markdown file is unavailable: {raw}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MarkdownViewerError("The requested file is outside the active Project.") from exc
    if not resolved.is_file():
        raise MarkdownViewerError(f"Markdown path is not a file: {raw}")
    if resolved.suffix.lower() not in MARKDOWN_SUFFIXES:
        raise MarkdownViewerError("Only Markdown files (.md, .markdown, or .mdx) can be opened.")
    return resolved


def list_markdown_files(root: str, query: str = "", limit: int = 500) -> dict[str, Any]:
    project_root = _resolve_root(root)
    needle = str(query or "").strip().casefold()
    safe_limit = max(1, min(int(limit), MAX_LIST_FILES))
    files: list[dict[str, Any]] = []
    truncated = False

    for current, dirs, names in os.walk(project_root, followlinks=False):
        dirs[:] = sorted(
            [name for name in dirs if name not in SKIP_DIRS and not name.startswith(".")],
            key=str.casefold,
        )
        for name in sorted(names, key=str.casefold):
            candidate = Path(current) / name
            if candidate.suffix.lower() not in MARKDOWN_SUFFIXES:
                continue
            try:
                resolved = candidate.resolve(strict=True)
                relative = resolved.relative_to(project_root).as_posix()
                stat = resolved.stat()
            except (OSError, RuntimeError, ValueError):
                continue
            if needle and needle not in relative.casefold():
                continue
            files.append(
                {
                    "name": resolved.name,
                    "relative": relative,
                    "path": str(resolved),
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                }
            )
            if len(files) >= safe_limit:
                truncated = True
                break
        if truncated:
            break

    files.sort(key=lambda item: item["relative"].casefold())
    return {
        "root": str(project_root),
        "files": files,
        "count": len(files),
        "truncated": truncated,
    }


def read_markdown_file(root: str, path: str) -> dict[str, Any]:
    project_root = _resolve_root(root)
    resolved = _resolve_markdown(project_root, path)
    try:
        with resolved.open("rb") as handle:
            data = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise MarkdownViewerError(f"Markdown file is unavailable: {path}") from exc
    if len(data) > MAX_FILE_BYTES:
        raise MarkdownViewerError(
            f"Markdown file is too large to preview ({len(data)} bytes read; limit {MAX_FILE_BYTES})."
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MarkdownViewerError("Markdown file is not valid UTF-8 text.") from exc
    stat = resolved.stat()
    return {
        "root": str(project_root),
        "path": str(resolved),
        "relative": resolved.relative_to(project_root).as_posix(),
        "name": resolved.name,
        "text": text,
        "size": len(data),
        "modified_ns": stat.st_mtime_ns,
    }


def _http_error(exc: MarkdownViewerError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.get("/files")
def files_endpoint(
    root: str,
    q: str = "",
    limit: Annotated[int, Query(ge=1, le=MAX_LIST_FILES)] = 500,
) -> dict[str, Any]:
    try:
        return list_markdown_files(root, query=q, limit=limit)
    except MarkdownViewerError as exc:
        raise _http_error(exc) from exc


@router.get("/read")
def read_endpoint(root: str, path: str) -> dict[str, Any]:
    try:
        return read_markdown_file(root, path)
    except MarkdownViewerError as exc:
        raise _http_error(exc) from exc
