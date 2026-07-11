"""Stable API adapter for the video material library capability."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from .batch import build_library_service, library_status, list_libraries, scan_library
from .config import resolve_library_config, resolve_source_path
from .management import add_library_source_root, migrate_legacy_library
from .service import VideoLibraryService
from .store import VideoLibraryStore


def _envelope(data: Any = None, *, ok: bool = True, error: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": ok, "data": data, "error": error}


def _error(message: str, code: str) -> dict[str, Any]:
    return _envelope(None, ok=False, error={"code": code, "message": message})


@lru_cache(maxsize=8)
def _service_for_root(root: str) -> VideoLibraryService:
    return VideoLibraryService(VideoLibraryStore(root=Path(root)))


def get_service() -> VideoLibraryService:
    return _service_for_root(str((get_hermes_home() / "video-library").resolve()))


def get_named_service(library_id: str) -> VideoLibraryService:
    return build_library_service(resolve_library_config(library_id))


def _request_service(library_id: str | None) -> VideoLibraryService:
    return get_named_service(library_id) if library_id else get_service()


def _failure(exc: Exception) -> tuple[int, dict[str, Any]]:
    if isinstance(exc, KeyError):
        return 404, _error(str(exc).strip("'"), "VIDEO_LIBRARY_NOT_FOUND")
    if isinstance(exc, (FileNotFoundError, ValueError)):
        return 400, _error(str(exc), "VIDEO_LIBRARY_INVALID_REQUEST")
    return 500, _error(str(exc), "VIDEO_LIBRARY_ERROR")


def import_asset_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        source_path = str(body.get("sourcePath") or body.get("source_path") or "").strip()
        if not source_path:
            raise ValueError("sourcePath is required")
        library_id = str(body.get("libraryId") or body.get("library_id") or "").strip()
        if library_id:
            library = resolve_library_config(library_id)
            source = resolve_source_path(library, source_path)
            asset = get_named_service(library_id).store.import_asset(
                source,
                source_mode=library.mode,
                library_id=library.id,
            )
        else:
            asset = get_service().import_asset(source_path)
        return 200, _envelope({"asset": asset})
    except Exception as exc:
        return _failure(exc)


def list_assets_data(*, library_id: str | None = None) -> tuple[int, dict[str, Any]]:
    try:
        assets = _request_service(library_id).store.list_assets()
        return 200, _envelope({"assets": assets, "total": len(assets)})
    except Exception as exc:
        return _failure(exc)


def analyze_asset_data(asset_id: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        library_id = str(body.get("libraryId") or body.get("library_id") or "").strip() or None
        result = _request_service(library_id).analyze_asset(
            asset_id,
            threshold=float(body.get("threshold", 0.32)),
            min_clip_seconds=float(body.get("minClipSeconds", body.get("min_clip_seconds", 1.0))),
            max_clips=int(body.get("maxClips", body.get("max_clips", 120))),
            fallback_clip_seconds=float(
                body.get("fallbackClipSeconds", body.get("fallback_clip_seconds", 5.0))
            ),
        )
        return 200, _envelope(result)
    except Exception as exc:
        return _failure(exc)


def list_clips_data(
    *,
    asset_id: str | None = None,
    library_id: str | None = None,
    limit: int = 50,
    query: str | None = None,
    tag: str | None = None,
) -> tuple[int, dict[str, Any]]:
    try:
        store = _request_service(library_id).store
        clips = (
            store.search_clips(query or "", tag=tag, limit=limit)
            if query
            else store.list_clips(asset_id=asset_id, tag=tag)
        )
        return 200, _envelope({"clips": clips, "total": len(clips)})
    except Exception as exc:
        return _failure(exc)


def replace_clip_tags_data(clip_id: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        raw_tags = body.get("tags")
        if not isinstance(raw_tags, list):
            raise ValueError("tags must be a list")
        normalized = [
            item if isinstance(item, dict) else {"name": str(item), "source": "manual"}
            for item in raw_tags
        ]
        library_id = str(body.get("libraryId") or body.get("library_id") or "").strip() or None
        tags = _request_service(library_id).store.replace_clip_tags(clip_id, normalized)
        return 200, _envelope({"clipId": clip_id, "tags": tags})
    except Exception as exc:
        return _failure(exc)


def create_timeline_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        clip_ids = body.get("clipIds", body.get("clip_ids"))
        if not isinstance(clip_ids, list):
            raise ValueError("clipIds must be a list")
        script = body.get("script")
        if script is not None and not isinstance(script, list):
            raise ValueError("script must be a list")
        library_id = str(body.get("libraryId") or body.get("library_id") or "").strip() or None
        timeline = _request_service(library_id).create_timeline(
            [str(item) for item in clip_ids],
            aspect=str(body.get("aspect") or "9:16"),
            library_id=library_id or "",
            script=script,
        )
        return 200, _envelope(timeline)
    except Exception as exc:
        return _failure(exc)


def list_libraries_data() -> tuple[int, dict[str, Any]]:
    try:
        return 200, _envelope(list_libraries())
    except Exception as exc:
        return _failure(exc)


def scan_library_data(library_id: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    try:
        raw = body or {}
        dry_run = bool(raw.get("dryRun", raw.get("dry_run", False)))
        return 200, _envelope(scan_library(library_id, dry_run=dry_run))
    except Exception as exc:
        return _failure(exc)


def library_status_data(library_id: str) -> tuple[int, dict[str, Any]]:
    try:
        return 200, _envelope(library_status(library_id))
    except Exception as exc:
        return _failure(exc)


def add_library_source_root_data(
    library_id: str, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    try:
        source_path = str(body.get("path") or "").strip()
        if not source_path:
            raise ValueError("path is required")
        return 200, _envelope(add_library_source_root(library_id, source_path))
    except Exception as exc:
        return _failure(exc)


def migrate_legacy_library_data(library_id: str) -> tuple[int, dict[str, Any]]:
    try:
        return 200, _envelope(migrate_legacy_library(library_id))
    except Exception as exc:
        return _failure(exc)
