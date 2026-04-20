from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
import threading
import uuid

from hermes_snapshot_manager.core.paths import AppPaths, build_paths, ensure_app_dirs
from hermes_snapshot_manager.services.restore_service import RestoreService
from hermes_snapshot_manager.services.snapshot_service import SnapshotService

# seconds after which a running operation is considered stale
STALE_THRESHOLD_SECONDS = 90

# seconds after which a completed or failed operation record is automatically
# cleared from disk.  Prevents UI clutter from old operations that finished
# (successfully or not) and are no longer relevant.  The operation state file
# is deleted so get_current_operation() returns None for expired records.
COMPLETED_OPERATION_EXPIRY_SECONDS = 3600  # 1 hour


class OperationService:
    def __init__(self, paths: AppPaths | None = None):
        self.paths = paths or build_paths()
        ensure_app_dirs(self.paths)
        self.snapshot_service = SnapshotService(self.paths)
        self.restore_service = RestoreService(self.paths)
        self._state_lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None
        self._running_operation_id: str | None = None
        # Clear any expired completed/failed operations on startup so the UI
        # never shows stale records from a previous session.
        self.get_current_operation()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _read_state(self) -> dict | None:
        if not self.paths.operation_status_path.exists():
            return None
        return json.loads(self.paths.operation_status_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict | None) -> None:
        if state is None:
            if self.paths.operation_status_path.exists():
                self.paths.operation_status_path.unlink()
            return
        self.paths.operation_status_path.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )

    # -------------------------------------------------------------------------
    # Stale / expired detection
    # -------------------------------------------------------------------------

    def _is_stale(self, state: dict) -> bool:
        """Return True when a running operation has not updated its heartbeat
        within the stale threshold."""
        if state.get("status") != "running":
            return False
        last_hb = state.get("last_heartbeat")
        if not last_hb:
            # No heartbeat ever written — check started_at as fallback
            started_str = state.get("started_at", "")
            try:
                started = datetime.strptime(started_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                return True  # malformed date = assume stale
            age = (datetime.now(timezone.utc) - started).total_seconds()
            return age > STALE_THRESHOLD_SECONDS
        try:
            last = datetime.strptime(last_hb, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        age = (datetime.now(timezone.utc) - last).total_seconds()
        return age > STALE_THRESHOLD_SECONDS

    def _is_expired_completed(self, state: dict) -> bool:
        """Return True when a completed or failed operation has been finished
        for longer than COMPLETED_OPERATION_EXPIRY_SECONDS and should be
        auto-cleared to keep the UI clean."""
        if state.get("status") not in ("completed", "failed"):
            return False
        finished_str = state.get("finished_at")
        if not finished_str:
            return False
        try:
            finished = datetime.strptime(finished_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return True  # malformed date = clear it
        age = (datetime.now(timezone.utc) - finished).total_seconds()
        return age > COMPLETED_OPERATION_EXPIRY_SECONDS

    def get_current_operation(self) -> dict | None:
        with self._state_lock:
            state = self._read_state()
            if not state:
                return None
            # Auto-clear expired completed/failed operations.
            if self._is_expired_completed(state):
                self._write_state(None)
                return None
            if self._is_stale(state):
                state = dict(state)
                state["status"] = "stale"
            return state

    # -------------------------------------------------------------------------
    # Partial snapshot cleanup
    # -------------------------------------------------------------------------

    def _cleanup_incomplete_snapshot_dirs(self) -> None:
        """Remove any snapshot directories that were left behind by a crashed
        or interrupted snapshot run (i.e. directories that exist but have no
        completed metadata.json or manifest.json).

        Safe to call concurrently: OSError from concurrent rmtree races is
        swallowed so that multiple /api/operations/current/clear or
        start_snapshot calls racing on the same incomplete dir never produce
        a 500.
        """
        if not self.paths.snapshot_root.exists():
            return
        for entry in self.paths.snapshot_root.iterdir():
            try:
                if not entry.is_dir():
                    continue
                has_metadata = (entry / "metadata.json").exists()
                has_manifest = (entry / "manifest.json").exists()
                # If neither is present, this directory is an incomplete leftover
                if not has_metadata and not has_manifest:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                # Directory was removed by a concurrent cleanup thread between
                # the iterdir() yield and the is_dir() / rmtree() call — safe to ignore.
                pass

    # -------------------------------------------------------------------------
    # Heartbeat thread
    # -------------------------------------------------------------------------

    def _start_heartbeat(self, operation_id: str) -> None:
        def heartbeat_loop():
            while True:
                if self._running_operation_id != operation_id:
                    break
                with self._state_lock:
                    self._update_state_nolock(operation_id, last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                threading.Event().wait(10)

        t = threading.Thread(target=heartbeat_loop, daemon=True)
        t.start()
        self._heartbeat_thread = t

    def _stop_heartbeat(self) -> None:
        self._running_operation_id = None
        # The thread is daemon so it will die naturally; signal it to exit
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.5)
            self._heartbeat_thread = None

    # -------------------------------------------------------------------------
    # State mutation helpers
    # -------------------------------------------------------------------------

    def _update_state_nolock(self, operation_id: str, **updates) -> None:
        """Must be called while holding _state_lock."""
        state = self._read_state()
        if not state or state.get("id") != operation_id:
            return
        state.update(updates)
        self._write_state(state)

    def _update_state(self, operation_id: str, **updates) -> None:
        with self._state_lock:
            self._update_state_nolock(operation_id, **updates)

    def _build_progress_callback(
        self, operation_id: str, stage: str, start_percent: int, end_percent: int, noun: str
    ):
        def callback(completed: int, total: int, current_item: str) -> None:
            # Guard: completed can exceed total due to rglob re-counting destination
            # files after symlink/skipped-placeholder divergence. Clamp to total so
            # progress never regresses and the UI never shows "80849/80378".
            # Also guard against total==0 (ZeroDivisionError) and negative completed.
            safe_completed = max(0, min(completed, total)) if total > 0 else 0
            percent = (
                end_percent
                if total <= 0
                else start_percent + int((safe_completed / total) * (end_percent - start_percent))
            )
            message = f"{noun}: {safe_completed}/{total}"
            self._update_state(
                operation_id,
                progress=min(percent, end_percent),
                stage=stage,
                message=message,
                current_item=current_item,
                last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )

        return callback

    def _build_skip_callback(self, operation_id: str):
        def callback(relative_path: str, error_message: str) -> None:
            with self._state_lock:
                state = self._read_state()
                if not state or state.get("id") != operation_id:
                    return
                skipped_count = int(state.get("skipped_files_count") or 0) + 1
                skipped_files = list(state.get("skipped_files") or [])
                skipped_files.append({"path": relative_path, "error": error_message})
                state.update(
                    {
                        "skipped_files_count": skipped_count,
                        "last_skipped_file": relative_path,
                        "message": f"Skipped file ({skipped_count}): {relative_path}",
                        "skipped_files": skipped_files[-20:],
                        "last_heartbeat": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )
                self._write_state(state)

        return callback

    def _build_traversal_skip_callback(self, operation_id: str):
        """Build a TraversalSkipCallback that records traversal-stage skips."""
        def callback(relative_path: str, reason: str) -> None:
            with self._state_lock:
                state = self._read_state()
                if not state or state.get("id") != operation_id:
                    return
                count = int(state.get("traversal_skipped_count") or 0) + 1
                skipped = list(state.get("traversal_skipped") or [])
                skipped.append({"path": relative_path, "reason": reason})
                state.update(
                    {
                        "traversal_skipped_count": count,
                        "last_traversal_skipped": relative_path,
                        "message": f"Skipped during traversal ({count}): {relative_path} — {reason}",
                        "traversal_skipped": skipped[-20:],
                        "last_heartbeat": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )
                self._write_state(state)

        return callback

    # -------------------------------------------------------------------------
    # Abort / Resume / Clear
    # -------------------------------------------------------------------------

    def abort_operation(self) -> dict | None:
        """Mark the current operation as aborted and clean up partial state."""
        with self._state_lock:
            state = self._read_state()
            if not state:
                return None
            self._stop_heartbeat()
            self._write_state(None)
        # Clean up any incomplete snapshot directories
        self._cleanup_incomplete_snapshot_dirs()
        return {"status": "aborted", "id": state["id"]}

    def clear_stale_operation(self) -> dict | None:
        """Clear a stale operation record and remove partial snapshot dirs."""
        with self._state_lock:
            state = self._read_state()
            if not state:
                return None
            if state.get("status") not in ("stale", "running"):
                return None
            self._stop_heartbeat()
            self._write_state(None)
        self._cleanup_incomplete_snapshot_dirs()
        return {"status": "cleared", "id": state["id"]}

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def _begin_operation(
        self, kind: str, target_id: str | None = None, detail: str | None = None
    ) -> dict:
        with self._state_lock:
            existing = self._read_state()
            if existing:
                # Auto-clear expired completed/failed operations so they don't
                # block new runs.
                if self._is_expired_completed(existing):
                    self._write_state(None)
                # Auto-clear stale running operations so they don't block new runs
                elif existing.get("status") in ("running", "stale") and self._is_stale(existing):
                    self._stop_heartbeat()
                    self._write_state(None)
                    self._cleanup_incomplete_snapshot_dirs()
                else:
                    raise RuntimeError(f"Another operation is already running: {existing['id']}")

            operation_id = f"op-{kind}-{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            state = {
                "id": operation_id,
                "kind": kind,
                "status": "running",
                "progress": 0,
                "stage": "queued",
                "message": f"Queued {kind} operation",
                "detail": detail,
                "target_id": target_id,
                "started_at": now,
                "last_heartbeat": now,
                "finished_at": None,
                "result": None,
                "error": None,
                "skipped_files_count": 0,
                "last_skipped_file": None,
                "skipped_files": [],
                "traversal_skipped_count": 0,
                "last_traversal_skipped": None,
                "traversal_skipped": [],
            }
            self._write_state(state)
            self._running_operation_id = operation_id
            return state

    def start_snapshot(self, label: str | None = None) -> dict:
        # Always try to clean up leftovers from crashed runs first
        self._cleanup_incomplete_snapshot_dirs()
        state = self._begin_operation("snapshot", detail=label)
        operation_id = state["id"]

        def run() -> None:
            try:
                self._start_heartbeat(operation_id)
                self._update_state(
                    operation_id,
                    progress=5,
                    stage="preparing",
                    message="Preparing snapshot",
                    last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
                snapshot = self.snapshot_service.create_snapshot(
                    label=label,
                    trigger_type="manual",
                    progress_callback=self._build_progress_callback(
                        operation_id, "copying_files", 10, 80, "Copying files"
                    ),
                    stage_callback=lambda stage, progress, message: self._update_state(
                        operation_id,
                        stage=stage,
                        progress=progress,
                        message=message,
                        last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ),
                    skip_callback=self._build_skip_callback(operation_id),
                    traversal_skip_callback=self._build_traversal_skip_callback(operation_id),
                )
                current_state = self.get_current_operation() or {}
                skipped_count = int(current_state.get("skipped_files_count") or 0)
                traversal_count = int(current_state.get("traversal_skipped_count") or 0)
                parts = []
                if skipped_count:
                    parts.append(f"{skipped_count} skipped file(s)")
                if traversal_count:
                    parts.append(f"{traversal_count} traversal skip(s)")
                completion_message = (
                    "Snapshot completed with " + ", ".join(parts)
                    if parts else "Snapshot completed"
                )
                self._update_state(
                    operation_id,
                    status="completed",
                    progress=100,
                    stage="completed",
                    message=completion_message,
                    result={
                        "snapshot_id": snapshot.id,
                        "skipped_files_count": skipped_count,
                        "traversal_skipped_count": traversal_count,
                    },
                    finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            except Exception as exc:
                self._update_state(
                    operation_id,
                    status="failed",
                    stage="failed",
                    message="Snapshot failed",
                    error=str(exc),
                    finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            finally:
                self._stop_heartbeat()

        threading.Thread(target=run, daemon=True).start()
        return {"operation_id": operation_id, "status": "running"}

    def start_restore(self, snapshot_id: str, notes: str | None = None) -> dict:
        state = self._begin_operation("restore", target_id=snapshot_id, detail=notes)
        operation_id = state["id"]

        def run() -> None:
            try:
                self._start_heartbeat(operation_id)
                self._update_state(
                    operation_id,
                    progress=5,
                    stage="preparing",
                    message="Preparing restore",
                    last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
                result = self.restore_service.restore_snapshot(
                    snapshot_id,
                    notes=notes,
                    pre_snapshot_progress_callback=self._build_progress_callback(
                        operation_id, "creating_pre_restore_snapshot", 10, 45, "Creating safeguard snapshot"
                    ),
                    restore_progress_callback=self._build_progress_callback(
                        operation_id, "restoring_files", 55, 95, "Restoring files"
                    ),
                    stage_callback=lambda stage, progress, message: self._update_state(
                        operation_id,
                        stage=stage,
                        progress=progress,
                        message=message,
                        last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ),
                )
                self._update_state(
                    operation_id,
                    status="completed",
                    progress=100,
                    stage="completed",
                    message="Restore completed",
                    result=result,
                    finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            except Exception as exc:
                self._update_state(
                    operation_id,
                    status="failed",
                    stage="failed",
                    message="Restore failed",
                    error=str(exc),
                    finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    last_heartbeat=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            finally:
                self._stop_heartbeat()

        threading.Thread(target=run, daemon=True).start()
        return {"operation_id": operation_id, "status": "running"}


