"""Batch scanning and resumable ingestion for configured video libraries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import VideoLibraryConfig, load_library_configs, resolve_library_config, resolve_source_path
from .obsidian import ObsidianProjector
from .semantic import analyze_keyframes
from .service import VideoLibraryService
from .store import SUPPORTED_VIDEO_SUFFIXES, VideoLibraryStore


@dataclass
class BatchScanResult:
    library_id: str
    complete: int = 0
    dry_run: bool = False
    errors: list[dict[str, str]] = field(default_factory=list)
    failed: int = 0
    low_confidence: int = 0
    skipped: int = 0
    total: int = 0
    unusable: int = 0
    writes_planned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VideoLibraryBatchRunner:
    def __init__(
        self,
        library: VideoLibraryConfig,
        *,
        semantic_analyzer=analyze_keyframes,
    ):
        self.library = library
        self.semantic_analyzer = semantic_analyzer
        self._service_instance: VideoLibraryService | None = None
        self.projector = ObsidianProjector(library.root)

    def _service(self) -> VideoLibraryService:
        if self._service_instance is None:
            store = VideoLibraryStore(
                root=self.library.root,
                db_path=self.library.database_path,
                assets_dir=self.library.metadata_dir / "managed-assets",
                clips_dir=self.library.selected_clips_dir,
                keyframes_dir=self.library.keyframes_dir,
            )
            self._service_instance = VideoLibraryService(
                store,
                semantic_analyzer=self.semantic_analyzer,
                taxonomy=self.library.taxonomy,
            )
        return self._service_instance

    def _discover(self) -> tuple[list[Path], list[dict[str, str]]]:
        files: list[Path] = []
        errors: list[dict[str, str]] = []
        for source_root in self.library.source_roots:
            if not source_root.is_dir():
                errors.append({"file": str(source_root), "message": "source root is missing", "stage": "authorization"})
                continue
            for candidate in sorted(source_root.rglob("*"), key=lambda path: str(path).casefold()):
                if candidate.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
                    continue
                try:
                    files.append(resolve_source_path(self.library, candidate))
                except (FileNotFoundError, ValueError) as exc:
                    errors.append({"file": str(candidate), "message": str(exc), "stage": "authorization"})
        return files, errors

    def process_file(self, path: Path) -> dict[str, Any]:
        service = self._service()
        asset = service.store.import_asset(
            path,
            source_mode=self.library.mode,
            library_id=self.library.id,
        )
        latest = service.store.latest_analysis_job(asset["id"])
        if (
            asset.get("status") == "analyzed"
            and latest is not None
            and latest.get("analyzer_version") == service.analyzer_version
            and latest.get("state") == "complete"
        ):
            return {"asset": asset, "status": "skipped"}

        analysis = service.analyze_asset(asset["id"])
        self.projector.write_asset(analysis["asset"], analysis["clips"])
        return {"analysis": analysis, "asset": analysis["asset"], "status": "complete"}

    def scan(self, *, dry_run: bool = False) -> BatchScanResult:
        files, authorization_errors = self._discover()
        result = BatchScanResult(
            library_id=self.library.id,
            dry_run=dry_run,
            errors=list(authorization_errors),
            failed=len(authorization_errors),
            total=len(files),
        )
        if dry_run:
            result.writes_planned = [
                str(self.library.database_path),
                str(self.library.keyframes_dir),
                str(self.library.analysis_dir),
            ]
            return result

        for path in files:
            try:
                outcome = self.process_file(path)
                if outcome.get("status") == "skipped":
                    result.skipped += 1
                    continue
                result.complete += 1
                analysis = outcome.get("analysis") or {}
                for clip in analysis.get("clips") or []:
                    if clip.get("status") == "low_confidence":
                        result.low_confidence += 1
                    elif clip.get("status") == "unusable":
                        result.unusable += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append({"file": str(path), "message": str(exc), "stage": "processing"})

        if self._service_instance is not None:
            self.projector.write_stats(
                self._service_instance.store.list_assets(),
                self._service_instance.store.list_clips(),
            )
        return result


__all__ = ["BatchScanResult", "VideoLibraryBatchRunner"]


def build_library_service(library: VideoLibraryConfig) -> VideoLibraryService:
    store = VideoLibraryStore(
        root=library.root,
        db_path=library.database_path,
        assets_dir=library.metadata_dir / "managed-assets",
        clips_dir=library.selected_clips_dir,
        keyframes_dir=library.keyframes_dir,
    )
    return VideoLibraryService(store, semantic_analyzer=analyze_keyframes, taxonomy=library.taxonomy)


def list_libraries(*, config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    libraries = load_library_configs(config)
    return {
        "libraries": [
            {
                "id": library.id,
                "mode": library.mode,
                "name": library.name,
                "root": str(library.root),
                "source_roots": [str(path) for path in library.source_roots],
                "taxonomy": library.taxonomy,
            }
            for library in libraries.values()
        ]
    }


def scan_library(library_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    library = resolve_library_config(library_id)
    return VideoLibraryBatchRunner(library).scan(dry_run=dry_run).to_dict()


def library_status(library_id: str) -> dict[str, Any]:
    library = resolve_library_config(library_id)
    if not library.database_path.is_file():
        return {
            "assets": 0,
            "clips": 0,
            "database_exists": False,
            "library_id": library.id,
            "root": str(library.root),
        }
    service = build_library_service(library)
    clips = service.store.list_clips()
    return {
        "assets": len(service.store.list_assets()),
        "clips": len(clips),
        "database_exists": True,
        "failed": sum(1 for clip in clips if clip.get("status") == "semantic_failed"),
        "library_id": library.id,
        "low_confidence": sum(1 for clip in clips if clip.get("status") == "low_confidence"),
        "root": str(library.root),
        "unusable": sum(1 for clip in clips if clip.get("status") == "unusable"),
    }


def search_library(library_id: str, query: str, *, tag: str = "", limit: int = 50) -> dict[str, Any]:
    library = resolve_library_config(library_id)
    if not library.database_path.is_file():
        return {"clips": [], "library_id": library.id, "query": query, "total": 0}
    clips = build_library_service(library).store.search_clips(query, tag=tag or None, limit=limit)
    return {"clips": clips, "library_id": library.id, "query": query, "total": len(clips)}


__all__ = [
    "BatchScanResult",
    "VideoLibraryBatchRunner",
    "build_library_service",
    "library_status",
    "list_libraries",
    "scan_library",
    "search_library",
]
