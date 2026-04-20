from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import tempfile

from hermes_snapshot_manager.core.filesystem import ProgressCallback, copy_files_with_progress
from hermes_snapshot_manager.core.locking import exclusive_lock
from hermes_snapshot_manager.core.paths import AppPaths, build_paths
from hermes_snapshot_manager.services.snapshot_service import SnapshotService
from hermes_snapshot_manager.storage.db import connect

StageCallback = Callable[[str, int, str], None]


class RestoreService:
    def __init__(self, paths: AppPaths | None = None):
        self.paths = paths or build_paths()
        self.snapshot_service = SnapshotService(self.paths)

    def _copy_snapshot_into_place(
        self,
        snapshot_dir: Path,
        destination: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        files = [path for path in snapshot_dir.rglob("*") if path.is_file()]
        copy_files_with_progress(snapshot_dir, destination, files, progress_callback=progress_callback)

    def _find_active_hermes_processes(self) -> list[str]:
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,args="],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return []
        markers = ("run_agent.py", "hermes ", "hermes-agent", "hermes_snapshot_manager.main:app")
        current_pid = str(Path('/proc/self').resolve().name) if Path('/proc/self').exists() else None
        lines: list[str] = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if current_pid and stripped.startswith(f"{current_pid} "):
                continue
            if any(marker in stripped for marker in markers):
                lines.append(stripped)
        return lines

    def _find_active_sqlite_sidecars(self) -> list[str]:
        matches: list[str] = []
        for suffix in ("-wal", "-journal", "-shm"):
            for path in self.paths.source_root.rglob(f"*{suffix}"):
                if path.is_file():
                    matches.append(str(path.relative_to(self.paths.source_root)))
        return sorted(matches)

    def _record_restore_attempt(
        self,
        snapshot_id: str,
        result: str,
        pre_restore_snapshot_id: str | None,
        notes: str | None,
    ) -> None:
        with connect(self.paths.db_path) as conn:
            conn.execute(
                "INSERT INTO restore_history (snapshot_id, restored_at, result, pre_restore_snapshot_id, notes) VALUES (?,?,?,?,?)",
                (snapshot_id, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), result, pre_restore_snapshot_id, notes),
            )
            conn.commit()

    def restore_snapshot(
        self,
        snapshot_id: str,
        notes: str | None = None,
        pre_snapshot_progress_callback: ProgressCallback | None = None,
        restore_progress_callback: ProgressCallback | None = None,
        stage_callback: StageCallback | None = None,
    ) -> dict:
        archive_path = self.snapshot_service._payload_archive_path(snapshot_id)
        if not archive_path.exists():
            raise FileNotFoundError(f"Snapshot payload archive missing: {archive_path}")

        if stage_callback is not None:
            stage_callback("checking_safety", 5, "Checking restore safety")
        active_processes = self._find_active_hermes_processes()
        if active_processes:
            raise RuntimeError(f"Cannot restore with active Hermes processes: {active_processes}")

        sqlite_sidecars = self._find_active_sqlite_sidecars()
        if sqlite_sidecars:
            raise RuntimeError(f"Cannot restore while SQLite activity is present: {sqlite_sidecars}")

        if stage_callback is not None:
            stage_callback("verifying_snapshot", 8, "Verifying snapshot before restore")
        verification = self.snapshot_service.verify_snapshot(snapshot_id)
        if not verification["ok"]:
            raise ValueError(f"Snapshot {snapshot_id} failed verification")

        with exclusive_lock(self.paths.lock_path):
            if stage_callback is not None:
                stage_callback("creating_pre_restore_snapshot", 10, "Creating safeguard snapshot")
            pre_restore = self.snapshot_service._create_snapshot_unlocked(
                label=f"pre-restore:{snapshot_id}",
                trigger_type="pre_restore",
                notes=f"Automatic safeguard before restoring {snapshot_id}",
                progress_callback=pre_snapshot_progress_callback,
            )
            backup_target = self.paths.source_root.parent / f"{self.paths.source_root.name}.restore-backup"
            if backup_target.exists():
                shutil.rmtree(backup_target)
            if self.paths.source_root.exists():
                shutil.move(str(self.paths.source_root), str(backup_target))
            try:
                with tempfile.TemporaryDirectory(prefix=f"snapshot-restore-{snapshot_id}-") as temp_dir:
                    extracted_root = self.snapshot_service._extract_payload(snapshot_id, Path(temp_dir))
                    if stage_callback is not None:
                        stage_callback("restoring_files", 55, "Restoring files")
                    if restore_progress_callback is not None:
                        self._copy_snapshot_into_place(extracted_root, self.paths.source_root, progress_callback=restore_progress_callback)
                    else:
                        self._copy_snapshot_into_place(extracted_root, self.paths.source_root)
            except Exception as exc:
                if self.paths.source_root.exists():
                    shutil.rmtree(self.paths.source_root)
                if backup_target.exists():
                    shutil.move(str(backup_target), str(self.paths.source_root))
                failure_notes = f"{notes or ''}\nrestore_error={exc}".strip()
                self._record_restore_attempt(snapshot_id, "failed", pre_restore.id, failure_notes)
                raise
            else:
                if backup_target.exists():
                    shutil.rmtree(backup_target)
                self._record_restore_attempt(snapshot_id, "success", pre_restore.id, notes)
        return {"restored_snapshot_id": snapshot_id, "pre_restore_snapshot_id": pre_restore.id, "result": "success"}

    def list_restore_history(self) -> list[dict]:
        with connect(self.paths.db_path) as conn:
            rows = conn.execute(
                "SELECT id, snapshot_id, restored_at, result, pre_restore_snapshot_id, notes FROM restore_history ORDER BY id DESC"
            ).fetchall()
        return [dict(row) for row in rows]
