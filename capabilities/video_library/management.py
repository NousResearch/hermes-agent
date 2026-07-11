"""Guarded management operations for named video libraries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_cli.config import read_raw_config, save_config
from hermes_constants import get_hermes_home

from .config import load_library_configs, resolve_library_config
from .store import VideoLibraryStore


def add_library_source_root(library_id: str, source_root: str | Path) -> dict[str, Any]:
    candidate = Path(source_root).expanduser().resolve(strict=True)
    if not candidate.is_dir():
        raise ValueError("video library source root must be a directory")

    raw = read_raw_config()
    libraries = load_library_configs(raw)
    normalized = str(library_id or "").strip().lower()
    if normalized not in libraries:
        raise KeyError(f"unknown video library: {normalized or '<empty>'}")

    entries = raw.get("video_libraries") or []
    entry = next(item for item in entries if str(item.get("id") or "").strip().lower() == normalized)
    roots = [str(Path(value).expanduser().resolve()) for value in entry.get("source_roots") or []]
    resolved = str(candidate)
    if resolved not in roots:
        roots.append(resolved)
        entry["source_roots"] = roots
        save_config(raw, strip_defaults=False, preserve_keys={("video_libraries",)})
    return {"library_id": normalized, "source_roots": roots}


def get_legacy_store() -> VideoLibraryStore:
    return VideoLibraryStore(root=(get_hermes_home() / "video-library").resolve())


def get_named_store(library_id: str) -> VideoLibraryStore:
    return VideoLibraryStore(root=resolve_library_config(library_id).root)


def migrate_legacy_library(library_id: str) -> dict[str, Any]:
    normalized = str(library_id or "").strip().lower()
    legacy = get_legacy_store()
    target = get_named_store(normalized)
    existing = {str(asset.get("sha256") or "") for asset in target.list_assets()}
    records: list[dict[str, str]] = []
    imported = skipped = failed = 0

    for asset in legacy.list_assets():
        source_asset_id = str(asset.get("id") or "")
        try:
            was_present = str(asset.get("sha256") or "") in existing
            migrated = target.import_asset(
                Path(str(asset.get("managed_path") or "")),
                source_mode="managed",
                library_id=normalized,
            )
            state = "skipped" if was_present else "imported"
            imported += int(state == "imported")
            skipped += int(state == "skipped")
            existing.add(str(migrated.get("sha256") or ""))
            records.append(
                {
                    "source_asset_id": source_asset_id,
                    "target_asset_id": str(migrated.get("id") or ""),
                    "state": state,
                    "error": "",
                }
            )
        except Exception as exc:
            failed += 1
            records.append(
                {
                    "source_asset_id": source_asset_id,
                    "target_asset_id": "",
                    "state": "failed",
                    "error": str(exc),
                }
            )

    return {
        "library_id": normalized,
        "total": len(records),
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "records": records,
    }


__all__ = [
    "add_library_source_root",
    "get_legacy_store",
    "get_named_store",
    "migrate_legacy_library",
]
