"""QQ intel worker assignment store and runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_time import now as hermes_now
from utils import atomic_json_write


INTEL_ASSIGNMENT_STORE_FILENAME = "qq_intel_assignments.json"
VALID_INTEL_WORKER_STATUSES = (
    "awaiting_group_approval",
    "active_collecting",
    "paused",
    "stopped",
    "failed",
    "rejected",
)


def _store_path() -> Path:
    return get_hermes_home() / INTEL_ASSIGNMENT_STORE_FILENAME


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _collect_unique_targets(values: list[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def normalize_intel_worker_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status not in VALID_INTEL_WORKER_STATUSES:
        raise ValueError(
            "Unsupported intel worker status. "
            "Use 'awaiting_group_approval', 'active_collecting', 'paused', "
            "'stopped', 'failed', or 'rejected'."
        )
    return status


def default_intel_worker(worker_name: str) -> dict[str, Any]:
    return {
        "worker_name": str(worker_name or "").strip(),
        "role": "intel",
        "status": "awaiting_group_approval",
        "target_group_ref": None,
        "target_group_id": None,
        "target_group_name": None,
        "objective": None,
        "daily_report_enabled": True,
        "daily_report_target": None,
        "manual_report_target": None,
        "notify_target": None,
        "notes": "",
        "last_error": None,
        "last_report_at": None,
        "last_status_at": None,
        "created_at": None,
        "created_by": None,
        "updated_at": None,
        "updated_by": None,
    }


def summarize_intel_worker_assignment(worker: dict[str, Any]) -> dict[str, Any]:
    daily_report_target = _normalize_optional_text(worker.get("daily_report_target"))
    manual_report_target = _normalize_optional_text(worker.get("manual_report_target"))
    notify_target = _normalize_optional_text(worker.get("notify_target"))
    status = normalize_intel_worker_status(worker.get("status", "awaiting_group_approval"))
    return {
        "worker_name": str(worker.get("worker_name") or "").strip(),
        "status": status,
        "collecting": status == "active_collecting",
        "target_group_ref": _normalize_optional_text(worker.get("target_group_ref")),
        "target_group_id": _normalize_optional_text(worker.get("target_group_id")),
        "target_group_name": _normalize_optional_text(worker.get("target_group_name")),
        "objective": _normalize_optional_text(worker.get("objective")),
        "daily_report_enabled": _normalize_bool(
            worker.get("daily_report_enabled"),
            default=True,
        ),
        "daily_report_target": daily_report_target,
        "manual_report_target": manual_report_target,
        "notify_target": notify_target,
        "has_daily_report_target": bool(daily_report_target),
        "has_manual_report_target": bool(manual_report_target),
        "has_notify_target": bool(notify_target),
        "last_error": _normalize_optional_text(worker.get("last_error")),
        "last_report_at": worker.get("last_report_at"),
        "last_status_at": worker.get("last_status_at"),
    }


def _normalize_joined_groups(joined_groups: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in joined_groups or []:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id") or item.get("groupCode") or "").strip()
        if not group_id:
            continue
        normalized.append(
            {
                "group_id": group_id,
                "group_name": str(item.get("group_name") or item.get("groupName") or group_id).strip(),
            }
        )
    return normalized


def _match_joined_group(target_group_ref: str | None, joined_groups: list[dict[str, str]]) -> dict[str, str] | None:
    ref = str(target_group_ref or "").strip()
    if not ref:
        return None
    lowered = ref.lower()
    numeric = ref
    if lowered.startswith("qq_napcat:group:"):
        numeric = ref.split(":", 2)[2]
    elif lowered.startswith("group:"):
        numeric = ref.split(":", 1)[1]

    if numeric.lstrip("-").isdigit():
        for item in joined_groups:
            if item["group_id"] == numeric:
                return item
        return None

    exact_matches = [
        item for item in joined_groups
        if str(item.get("group_name") or "").strip().lower() == lowered
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    partial_matches = [
        item for item in joined_groups
        if lowered in str(item.get("group_name") or "").strip().lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    return None


@dataclass
class QqIntelAssignmentStore:
    path: Path | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path or _store_path())
        self._lock = threading.RLock()
        self._last_mtime_ns: int | None = None
        self._path_exists = False
        self._reload_check_interval_seconds = 0.0
        self._next_reload_check_at = 0.0
        self._data = self._load_from_disk()

    def _load_from_disk(self) -> dict[str, Any]:
        if not self.path or not self.path.exists():
            self._path_exists = False
            self._last_mtime_ns = None
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "updated_at": None, "workers": {}}

        try:
            stat = self.path.stat()
            self._last_mtime_ns = int(stat.st_mtime_ns)
            self._path_exists = True
        except OSError:
            self._path_exists = False
            self._last_mtime_ns = None
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "updated_at": None, "workers": {}}
        try:
            import json

            payload = json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "updated_at": None, "workers": {}}
        workers = payload.get("workers")
        if not isinstance(workers, dict):
            workers = {}
        self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
        return {
            "version": 1,
            "updated_at": payload.get("updated_at"),
            "workers": workers,
        }

    def _refresh_if_stale(self) -> None:
        now = time.monotonic()
        if now < self._next_reload_check_at:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_reload_check_at:
                return
            self._next_reload_check_at = now + self._reload_check_interval_seconds
            exists = bool(self.path and self.path.exists())
            current_mtime_ns: int | None = None
            if exists and self.path is not None:
                try:
                    current_mtime_ns = int(self.path.stat().st_mtime_ns)
                except OSError:
                    exists = False
                    current_mtime_ns = None

            if exists != self._path_exists or current_mtime_ns != self._last_mtime_ns:
                self._data = self._load_from_disk()

    def _save_locked(self) -> None:
        if not self.path:
            return
        payload = {
            "version": 1,
            "updated_at": hermes_now().isoformat(),
            "workers": self._data.get("workers") or {},
        }
        atomic_json_write(self.path, payload)
        try:
            stat = self.path.stat()
            self._last_mtime_ns = int(stat.st_mtime_ns)
            self._path_exists = True
        except OSError:
            self._last_mtime_ns = None
            self._path_exists = False
        self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
        self._data = payload

    def get_worker(self, worker_name: str) -> dict[str, Any] | None:
        normalized_name = str(worker_name or "").strip()
        if not normalized_name:
            raise ValueError("worker_name is required")
        self._refresh_if_stale()
        with self._lock:
            stored = dict((self._data.get("workers") or {}).get(normalized_name) or {})
        if not stored:
            return None
        worker = default_intel_worker(normalized_name)
        worker.update(
            {
                "role": str(stored.get("role") or "intel"),
                "status": normalize_intel_worker_status(stored.get("status", worker["status"])),
                "target_group_ref": _normalize_optional_text(stored.get("target_group_ref")),
                "target_group_id": _normalize_optional_text(stored.get("target_group_id")),
                "target_group_name": _normalize_optional_text(stored.get("target_group_name")),
                "objective": _normalize_optional_text(stored.get("objective")),
                "daily_report_enabled": _normalize_bool(
                    stored.get("daily_report_enabled"),
                    default=worker["daily_report_enabled"],
                ),
                "daily_report_target": _normalize_optional_text(stored.get("daily_report_target")),
                "manual_report_target": _normalize_optional_text(stored.get("manual_report_target")),
                "notify_target": _normalize_optional_text(stored.get("notify_target")),
                "notes": str(stored.get("notes") or "").strip(),
                "last_error": _normalize_optional_text(stored.get("last_error")),
                "last_report_at": stored.get("last_report_at"),
                "last_status_at": stored.get("last_status_at"),
                "created_at": stored.get("created_at"),
                "created_by": stored.get("created_by"),
                "updated_at": stored.get("updated_at"),
                "updated_by": stored.get("updated_by"),
            }
        )
        return worker

    def list_workers(self, *, status: str | None = None) -> list[dict[str, Any]]:
        self._refresh_if_stale()
        with self._lock:
            names = sorted((self._data.get("workers") or {}).keys())
        workers = [self.get_worker(name) for name in names]
        result = [item for item in workers if item is not None]
        if status:
            normalized_status = normalize_intel_worker_status(status)
            result = [item for item in result if item["status"] == normalized_status]
        return result

    def hire_worker(
        self,
        *,
        worker_name: str,
        target_group_ref: str,
        objective: str | None = None,
        daily_report_enabled: bool = True,
        daily_report_target: str | None = None,
        manual_report_target: str | None = None,
        notify_target: str | None = None,
        notes: str | None = None,
        updated_by: str | None = None,
        joined_groups: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_name = str(worker_name or "").strip()
        normalized_ref = str(target_group_ref or "").strip()
        if not normalized_name:
            raise ValueError("worker_name is required")
        if not normalized_ref:
            raise ValueError("target_group_ref is required")

        joined = _normalize_joined_groups(joined_groups)
        matched_group = _match_joined_group(normalized_ref, joined)
        now_iso = hermes_now().isoformat()

        self._refresh_if_stale()
        with self._lock:
            workers = dict(self._data.get("workers") or {})
            current = default_intel_worker(normalized_name)
            current.update(dict(workers.get(normalized_name) or {}))
            created_at = current.get("created_at") or now_iso
            created_by = current.get("created_by") or (str(updated_by or "").strip() or None)
            status = "active_collecting" if matched_group else "awaiting_group_approval"
            last_error = None if matched_group else "目标群当前未加入，等待入群通过后自动转入潜伏采集。"
            workers[normalized_name] = {
                "role": "intel",
                "status": status,
                "target_group_ref": normalized_ref,
                "target_group_id": matched_group["group_id"] if matched_group else None,
                "target_group_name": matched_group["group_name"] if matched_group else None,
                "objective": _normalize_optional_text(objective),
                "daily_report_enabled": bool(daily_report_enabled),
                "daily_report_target": _normalize_optional_text(daily_report_target),
                "manual_report_target": _normalize_optional_text(manual_report_target),
                "notify_target": _normalize_optional_text(notify_target),
                "notes": str(notes or current.get("notes") or "").strip(),
                "last_error": last_error,
                "last_report_at": current.get("last_report_at"),
                "last_status_at": now_iso,
                "created_at": created_at,
                "created_by": created_by,
                "updated_at": now_iso,
                "updated_by": str(updated_by or "").strip() or None,
            }
            self._data["workers"] = workers
            self._save_locked()
        worker = self.get_worker(normalized_name)
        assert worker is not None
        return worker

    def update_worker(
        self,
        worker_name: str,
        *,
        objective: str | None = None,
        daily_report_enabled: bool | None = None,
        daily_report_target: str | None = None,
        manual_report_target: str | None = None,
        notify_target: str | None = None,
        last_report_at: str | None = None,
        notes: str | None = None,
        updated_by: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_worker(worker_name)
        if current is None:
            raise ValueError(f"Intel worker '{worker_name}' does not exist.")
        now_iso = hermes_now().isoformat()
        with self._lock:
            workers = dict(self._data.get("workers") or {})
            workers[current["worker_name"]] = {
                **workers.get(current["worker_name"], {}),
                "objective": current["objective"] if objective is None else _normalize_optional_text(objective),
                "daily_report_enabled": current["daily_report_enabled"] if daily_report_enabled is None else bool(daily_report_enabled),
                "daily_report_target": current["daily_report_target"] if daily_report_target is None else _normalize_optional_text(daily_report_target),
                "manual_report_target": current["manual_report_target"] if manual_report_target is None else _normalize_optional_text(manual_report_target),
                "notify_target": current["notify_target"] if notify_target is None else _normalize_optional_text(notify_target),
                "last_report_at": current["last_report_at"] if last_report_at is None else _normalize_optional_text(last_report_at),
                "notes": current["notes"] if notes is None else str(notes or "").strip(),
                "updated_at": now_iso,
                "updated_by": str(updated_by or "").strip() or None,
            }
            self._data["workers"] = workers
            self._save_locked()
        updated = self.get_worker(worker_name)
        assert updated is not None
        return updated

    def set_worker_status(
        self,
        worker_name: str,
        *,
        status: str,
        updated_by: str | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_worker(worker_name)
        if current is None:
            raise ValueError(f"Intel worker '{worker_name}' does not exist.")
        now_iso = hermes_now().isoformat()
        normalized_status = normalize_intel_worker_status(status)
        with self._lock:
            workers = dict(self._data.get("workers") or {})
            workers[current["worker_name"]] = {
                **workers.get(current["worker_name"], {}),
                "status": normalized_status,
                "last_error": _normalize_optional_text(last_error),
                "last_status_at": now_iso,
                "updated_at": now_iso,
                "updated_by": str(updated_by or "").strip() or None,
            }
            self._data["workers"] = workers
            self._save_locked()
        updated = self.get_worker(worker_name)
        assert updated is not None
        return updated

    def resume_worker(
        self,
        worker_name: str,
        *,
        joined_groups: list[dict[str, Any]] | None = None,
        updated_by: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_worker(worker_name)
        if current is None:
            raise ValueError(f"Intel worker '{worker_name}' does not exist.")
        normalized_groups = _normalize_joined_groups(joined_groups)
        matched_group = _match_joined_group(current.get("target_group_ref"), normalized_groups)
        if matched_group:
            return self._apply_group_match(
                worker_name,
                matched_group,
                status="active_collecting",
                updated_by=updated_by,
                last_error=None,
            )
        return self.set_worker_status(
            worker_name,
            status="awaiting_group_approval",
            updated_by=updated_by,
            last_error="已恢复任务，等待进入目标群后自动转入采集。",
        )

    def reconcile_joined_groups(
        self,
        joined_groups: list[dict[str, Any]] | None,
        *,
        updated_by: str | None = None,
    ) -> dict[str, Any]:
        normalized_groups = _normalize_joined_groups(joined_groups)
        changes: list[dict[str, Any]] = []
        for worker in self.list_workers():
            if worker["status"] in {"stopped", "rejected"}:
                continue
            matched_group = _match_joined_group(worker.get("target_group_ref"), normalized_groups)
            if matched_group and worker["status"] in {"awaiting_group_approval", "failed"}:
                updated = self._apply_group_match(
                    worker["worker_name"],
                    matched_group,
                    status="active_collecting",
                    updated_by=updated_by,
                    last_error=None,
                )
                changes.append(
                    {
                        "worker_name": updated["worker_name"],
                        "from_status": worker["status"],
                        "to_status": updated["status"],
                        "group_id": updated["target_group_id"],
                        "group_name": updated["target_group_name"],
                    }
                )
                continue

            if not matched_group and worker["status"] == "active_collecting":
                updated = self.set_worker_status(
                    worker["worker_name"],
                    status="failed",
                    updated_by=updated_by,
                    last_error="QQ 账号当前不在目标群，潜伏任务已失联。",
                )
                changes.append(
                    {
                        "worker_name": updated["worker_name"],
                        "from_status": worker["status"],
                        "to_status": updated["status"],
                        "group_id": worker.get("target_group_id"),
                        "group_name": worker.get("target_group_name"),
                    }
                )
        return {
            "success": True,
            "changed": len(changes),
            "changes": changes,
        }

    def _apply_group_match(
        self,
        worker_name: str,
        matched_group: dict[str, str],
        *,
        status: str,
        updated_by: str | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_worker(worker_name)
        if current is None:
            raise ValueError(f"Intel worker '{worker_name}' does not exist.")
        now_iso = hermes_now().isoformat()
        with self._lock:
            workers = dict(self._data.get("workers") or {})
            workers[current["worker_name"]] = {
                **workers.get(current["worker_name"], {}),
                "status": normalize_intel_worker_status(status),
                "target_group_id": matched_group["group_id"],
                "target_group_name": matched_group["group_name"],
                "last_error": _normalize_optional_text(last_error),
                "last_status_at": now_iso,
                "updated_at": now_iso,
                "updated_by": str(updated_by or "").strip() or None,
            }
            self._data["workers"] = workers
            self._save_locked()
        updated = self.get_worker(worker_name)
        assert updated is not None
        return updated

    def get_group_monitoring_overlay(self, group_id: str) -> dict[str, Any]:
        normalized_group_id = str(group_id or "").strip()
        workers = [
            worker
            for worker in self.list_workers(status="active_collecting")
            if str(worker.get("target_group_id") or "").strip() == normalized_group_id
        ]
        worker_assignments = [
            summarize_intel_worker_assignment(worker)
            for worker in workers
        ]
        daily_report_targets = _collect_unique_targets(
            [worker.get("daily_report_target") for worker in workers if bool(worker.get("daily_report_enabled"))]
        )
        manual_report_targets = _collect_unique_targets(
            [worker.get("manual_report_target") for worker in workers]
        )
        notify_targets = _collect_unique_targets(
            [worker.get("notify_target") for worker in workers]
        )
        return {
            "active": bool(workers),
            "mode": "collect_only" if workers else "default",
            "archive_enabled": bool(workers),
            "active_worker_count": len(workers),
            "daily_report_enabled": any(bool(worker.get("daily_report_enabled")) for worker in workers),
            "monitoring_intent": "intel_worker_collect_only" if workers else "default",
            "worker_names": _collect_unique_targets([worker.get("worker_name") for worker in workers]),
            "daily_report_targets": daily_report_targets,
            "manual_report_targets": manual_report_targets,
            "notify_targets": notify_targets,
            "worker_assignments": worker_assignments,
            "report_control": {
                "daily_report_enabled": any(
                    bool(worker.get("daily_report_enabled")) for worker in workers
                ),
                "daily_report_targets": daily_report_targets,
                "manual_report_targets": manual_report_targets,
                "notify_targets": notify_targets,
            },
            "workers": workers,
        }

    def list_active_daily_report_workers_for_group(self, group_id: str) -> list[dict[str, Any]]:
        normalized_group_id = str(group_id or "").strip()
        return [
            worker
            for worker in self.list_workers(status="active_collecting")
            if str(worker.get("target_group_id") or "").strip() == normalized_group_id
            and bool(worker.get("daily_report_enabled"))
            and str(worker.get("daily_report_target") or "").strip()
        ]


_store_singleton: QqIntelAssignmentStore | None = None
_store_singleton_lock = threading.Lock()


def get_intel_assignment_store() -> QqIntelAssignmentStore:
    global _store_singleton
    path = _store_path()
    with _store_singleton_lock:
        if _store_singleton is None or _store_singleton.path != path:
            _store_singleton = QqIntelAssignmentStore(path=path)
    return _store_singleton


def hire_intel_worker(**kwargs: Any) -> dict[str, Any]:
    return get_intel_assignment_store().hire_worker(**kwargs)


def get_intel_worker(worker_name: str) -> dict[str, Any] | None:
    return get_intel_assignment_store().get_worker(worker_name)


def list_intel_workers(*, status: str | None = None) -> list[dict[str, Any]]:
    return get_intel_assignment_store().list_workers(status=status)


def update_intel_worker(worker_name: str, **kwargs: Any) -> dict[str, Any]:
    return get_intel_assignment_store().update_worker(worker_name, **kwargs)


def set_intel_worker_status(worker_name: str, **kwargs: Any) -> dict[str, Any]:
    return get_intel_assignment_store().set_worker_status(worker_name, **kwargs)


def reconcile_intel_workers(
    joined_groups: list[dict[str, Any]] | None,
    *,
    updated_by: str | None = None,
) -> dict[str, Any]:
    return get_intel_assignment_store().reconcile_joined_groups(joined_groups, updated_by=updated_by)


def resume_intel_worker(
    worker_name: str,
    *,
    joined_groups: list[dict[str, Any]] | None = None,
    updated_by: str | None = None,
) -> dict[str, Any]:
    return get_intel_assignment_store().resume_worker(
        worker_name,
        joined_groups=joined_groups,
        updated_by=updated_by,
    )


def get_group_monitoring_overlay(group_id: str) -> dict[str, Any]:
    return get_intel_assignment_store().get_group_monitoring_overlay(group_id)


def list_active_daily_report_workers_for_group(group_id: str) -> list[dict[str, Any]]:
    return get_intel_assignment_store().list_active_daily_report_workers_for_group(group_id)
