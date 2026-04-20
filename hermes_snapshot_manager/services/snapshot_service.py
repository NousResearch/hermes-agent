from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from hermes_snapshot_manager.core.compression import create_tar_gz, extract_tar_gz
from hermes_snapshot_manager.core.config import SnapshotManagerSettings, load_settings
from hermes_snapshot_manager.core.display import compression_ratio
from hermes_snapshot_manager.core.filesystem import ProgressCallback, SkipCallback, TraversalSkipCallback, copy_tree, should_include
from hermes_snapshot_manager.core.hashing import sha256_file
from hermes_snapshot_manager.core.locking import exclusive_lock
from hermes_snapshot_manager.core.paths import AppPaths, build_paths, ensure_app_dirs
from hermes_snapshot_manager.storage.db import connect, init_db
from hermes_snapshot_manager.storage.models import SnapshotSummary
from hermes_snapshot_manager.services.manifest_service import build_manifest, write_manifest

StageCallback = Callable[[str, int, str], None]


class SnapshotService:
    def __init__(self, paths: AppPaths | None = None, settings: SnapshotManagerSettings | None = None):
        self.paths = paths or build_paths()
        self.settings = settings or load_settings(self.paths)
        ensure_app_dirs(self.paths)
        init_db(self.paths.db_path)

    def _payload_archive_path(self, snapshot_id: str) -> Path:
        return self.paths.snapshot_root / snapshot_id / "files.tar.gz"

    def _extract_payload(self, snapshot_id: str, destination: Path) -> Path:
        archive_path = self._payload_archive_path(snapshot_id)
        if not archive_path.exists():
            raise FileNotFoundError(f"Snapshot payload archive missing: {archive_path}")
        extracted_root = extract_tar_gz(archive_path, destination)
        return extracted_root

    def create_snapshot(
        self,
        label: str | None = None,
        trigger_type: str = "manual",
        notes: str | None = None,
        progress_callback: ProgressCallback | None = None,
        stage_callback: StageCallback | None = None,
        skip_callback: SkipCallback | None = None,
        traversal_skip_callback: TraversalSkipCallback | None = None,
    ) -> SnapshotSummary:
        # Reload settings for every snapshot so changes made from the settings UI
        # apply to the very next run without requiring a service restart.
        self.settings = load_settings(self.paths)
        with exclusive_lock(self.paths.lock_path):
            return self._create_snapshot_unlocked(
                label=label,
                trigger_type=trigger_type,
                notes=notes,
                progress_callback=progress_callback,
                stage_callback=stage_callback,
                skip_callback=skip_callback,
            )

    def _create_snapshot_unlocked(
        self,
        label: str | None = None,
        trigger_type: str = "manual",
        notes: str | None = None,
        progress_callback: ProgressCallback | None = None,
        stage_callback: StageCallback | None = None,
        skip_callback: SkipCallback | None = None,
        traversal_skip_callback: TraversalSkipCallback | None = None,
    ) -> SnapshotSummary:
        if not self.paths.source_root.exists():
            raise FileNotFoundError(f"Hermes source root does not exist: {self.paths.source_root}")

        if stage_callback is not None:
            stage_callback("preparing", 5, "Preparing snapshot")

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        snapshot_id = f"{created_at}_{uuid.uuid4().hex[:6]}"
        snapshot_dir = self.paths.snapshot_root / snapshot_id
        files_root = snapshot_dir / "files" / ".hermes"
        metadata_path = snapshot_dir / "metadata.json"
        manifest_path = snapshot_dir / "manifest.json"
        archive_path = snapshot_dir / "files.tar.gz"
        skipped_log_path = snapshot_dir / "skipped-files.json"

        if snapshot_dir.exists():
            raise FileExistsError(f"Snapshot path already exists: {snapshot_dir}")
        files_root.mkdir(parents=True, exist_ok=False)
        _traversal_skipped_collector: list[tuple[str, str]] = []
        _included_files, skipped_files, is_degraded = copy_tree(
            self.paths.source_root,
            files_root,
            include_patterns=self.settings.include_patterns,
            exclude_patterns=self.settings.exclude_patterns,
            progress_callback=progress_callback,
            scan_progress_callback=(
                None
                if stage_callback is None
                else lambda scanned_count, current_item: stage_callback(
                    "scanning_files",
                    8,
                    f"Scanning files to include: {scanned_count} checked · {current_item}",
                )
            ),
            skip_callback=skip_callback,
            skip_stuck_patterns=self.settings.skip_stuck_patterns,
            max_consecutive_skipped=self.settings.max_consecutive_skipped,
            skip_fs_types=frozenset(self.settings.skip_fs_types),
            skip_mount_points=tuple(self.settings.skip_mount_points),
            cross_device=self.settings.cross_device,
            traversal_skip_callback=traversal_skip_callback,
            traversal_skipped_collector=_traversal_skipped_collector,
        )
        # Write a dedicated traversal-skipped log so it survives restart and is
        # readable from the snapshot directory.
        if _traversal_skipped_collector:
            traversal_skipped_log_path = snapshot_dir / "traversal-skipped.json"
            traversal_skipped_log_path.write_text(
                json.dumps(
                    [{"path": path, "reason": reason} for path, reason in _traversal_skipped_collector],
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        if skipped_files:
            skipped_log_path.write_text(
                json.dumps(
                    [
                        {"path": relative_path, "error": error_message}
                        for relative_path, error_message in skipped_files
                    ],
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        # Determine snapshot status: degraded if any files were skipped.
        snapshot_status = "degraded" if is_degraded else "created"
        if stage_callback is not None:
            stage_callback("building_manifest", 80, "Building manifest")
        file_paths = [path for path in files_root.rglob("*") if path.is_file()]
        manifest = build_manifest(snapshot_id, self.paths.source_root, files_root, label, trigger_type, file_paths)
        manifest_sha = write_manifest(manifest, manifest_path)
        if stage_callback is not None:
            stage_callback("compressing_payload", 90, "Compressing snapshot payload")
        create_tar_gz(files_root, archive_path)
        shutil.rmtree(files_root.parent)
        total_files = len(manifest["files"])
        total_bytes = sum(item["size"] for item in manifest["files"])
        traversal_log_name = "traversal-skipped.json" if _traversal_skipped_collector else None
        metadata = {
            "id": snapshot_id,
            "created_at": manifest["created_at"],
            "label": label,
            "trigger_type": trigger_type,
            "status": snapshot_status,
            "source_root": str(self.paths.source_root),
            "total_files": total_files,
            "total_bytes": total_bytes,
            "manifest_sha256": manifest_sha,
            "notes": notes,
            "payload_format": "tar.gz",
            "payload_archive": archive_path.name,
            "payload_archive_size": archive_path.stat().st_size,
            "compression_ratio": compression_ratio(total_bytes, archive_path.stat().st_size),
            "space_saved_bytes": max(total_bytes - archive_path.stat().st_size, 0),
            "skipped_files_count": len(skipped_files),
            "skipped_files_log": skipped_log_path.name if skipped_files else None,
            "traversal_skipped_count": len(_traversal_skipped_collector),
            "traversal_skipped_log": traversal_log_name,
            "is_degraded": is_degraded,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        if stage_callback is not None:
            stage_callback("recording_catalog", 95, "Recording snapshot in catalog")
        with connect(self.paths.db_path) as conn:
            conn.execute(
                """
                INSERT INTO snapshots (id, created_at, label, trigger_type, status, source_root, total_files, total_bytes, manifest_sha256, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    metadata["created_at"],
                    label,
                    trigger_type,
                    snapshot_status,
                    str(self.paths.source_root),
                    total_files,
                    total_bytes,
                    manifest_sha,
                    notes,
                ),
            )
            conn.executemany(
                """
                INSERT INTO snapshot_files (snapshot_id, relative_path, sha256, size, mtime, mode, file_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        snapshot_id,
                        item["path"],
                        item["sha256"],
                        item["size"],
                        item["mtime"],
                        item["mode"],
                        item["file_type"],
                    )
                    for item in manifest["files"]
                ],
            )
            conn.commit()

        return SnapshotSummary(
            id=snapshot_id,
            created_at=metadata["created_at"],
            label=label,
            trigger_type=trigger_type,
            status=snapshot_status,
            total_files=total_files,
            total_bytes=total_bytes,
            is_known_good=False,
        )

    def list_snapshots(self) -> list[SnapshotSummary]:
        with connect(self.paths.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, label, trigger_type, status, total_files, total_bytes, is_known_good
                FROM snapshots
                ORDER BY created_at DESC, rowid DESC
                """
            ).fetchall()
        return [SnapshotSummary(**dict(row)) for row in rows]

    def get_snapshot(self, snapshot_id: str) -> dict:
        with connect(self.paths.db_path) as conn:
            snapshot = conn.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()
            if snapshot is None:
                raise KeyError(f"Unknown snapshot: {snapshot_id}")
            files = conn.execute(
                "SELECT relative_path, sha256, size, mtime, mode, file_type FROM snapshot_files WHERE snapshot_id = ? ORDER BY relative_path",
                (snapshot_id,),
            ).fetchall()
        manifest_path = self.paths.snapshot_root / snapshot_id / "manifest.json"
        metadata_path = self.paths.snapshot_root / snapshot_id / "metadata.json"
        return {
            "snapshot": dict(snapshot),
            "files": [dict(row) for row in files],
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None,
            "metadata": json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None,
        }

    def verify_snapshot(self, snapshot_id: str) -> dict:
        detail = self.get_snapshot(snapshot_id)
        missing_files: list[str] = []
        changed_files: list[str] = []
        with tempfile.TemporaryDirectory(prefix=f"snapshot-verify-{snapshot_id}-") as temp_dir:
            payload_root = self._extract_payload(snapshot_id, Path(temp_dir))
            for item in detail["files"]:
                path = payload_root / item["relative_path"]
                if not path.exists():
                    missing_files.append(item["relative_path"])
                    continue
                if sha256_file(path) != item["sha256"]:
                    changed_files.append(item["relative_path"])
        ok = not missing_files and not changed_files
        with connect(self.paths.db_path) as conn:
            conn.execute("UPDATE snapshots SET status = ? WHERE id = ?", ("verified" if ok else "failed", snapshot_id))
            conn.commit()
        return {"ok": ok, "missing_files": missing_files, "changed_files": changed_files}

    def diff_snapshot_to_current(self, snapshot_id: str) -> dict:
        detail = self.get_snapshot(snapshot_id)
        snapshot_map = {item["relative_path"]: item["sha256"] for item in detail["files"]}
        current_map: dict[str, str] = {}
        for path in self.paths.source_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.paths.source_root).as_posix()
            if not should_include(relative, self.settings.include_patterns, self.settings.exclude_patterns):
                continue
            current_map[relative] = sha256_file(path)
        snapshot_paths = set(snapshot_map)
        current_paths = set(current_map)
        return {
            "added": sorted(current_paths - snapshot_paths),
            "removed": sorted(snapshot_paths - current_paths),
            "changed": sorted(path for path in snapshot_paths & current_paths if snapshot_map[path] != current_map[path]),
        }

    def mark_known_good(self, snapshot_id: str, value: bool = True) -> None:
        with connect(self.paths.db_path) as conn:
            conn.execute("UPDATE snapshots SET is_known_good = ? WHERE id = ?", (1 if value else 0, snapshot_id))
            conn.commit()

    def delete_snapshot(self, snapshot_id: str) -> None:
        snapshot_dir = self.paths.snapshot_root / snapshot_id
        with exclusive_lock(self.paths.lock_path):
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            with connect(self.paths.db_path) as conn:
                conn.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
                conn.commit()
