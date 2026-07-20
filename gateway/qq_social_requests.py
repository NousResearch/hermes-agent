"""Persistent store for QQ/NapCat social request events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_time import now as hermes_now
from utils import atomic_json_write


QQ_SOCIAL_REQUEST_STORE_FILENAME = "qq_social_requests.json"
VALID_SOCIAL_REQUEST_TYPES = ("friend", "group")
VALID_SOCIAL_REQUEST_STATUSES = ("pending", "approved", "rejected", "ignored")


def _store_path() -> Path:
    return get_hermes_home() / QQ_SOCIAL_REQUEST_STORE_FILENAME


def normalize_social_request_type(value: Any) -> str:
    request_type = str(value or "").strip().lower()
    if request_type not in VALID_SOCIAL_REQUEST_TYPES:
        raise ValueError("Unsupported QQ social request type. Use 'friend' or 'group'.")
    return request_type


def normalize_social_request_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status not in VALID_SOCIAL_REQUEST_STATUSES:
        raise ValueError("Unsupported QQ social request status. Use 'pending', 'approved', 'rejected', or 'ignored'.")
    return status


def build_social_request_key(request_type: Any, flag: Any) -> str:
    normalized_type = normalize_social_request_type(request_type)
    normalized_flag = str(flag or "").strip()
    if not normalized_flag:
        raise ValueError("QQ social request flag is required.")
    return f"{normalized_type}:{normalized_flag}"


def parse_social_request_key(request_key: Any) -> tuple[str, str]:
    text = str(request_key or "").strip()
    if ":" not in text:
        raise ValueError("QQ social request key must use '<type>:<flag>'.")
    request_type, flag = text.split(":", 1)
    normalized_type = normalize_social_request_type(request_type)
    normalized_flag = str(flag or "").strip()
    if not normalized_flag:
        raise ValueError("QQ social request key is missing the flag portion.")
    return normalized_type, normalized_flag


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_optional_numeric_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _observed_at_from_payload(payload: dict[str, Any]) -> str:
    raw_time = payload.get("time")
    if raw_time is None:
        return hermes_now().isoformat()
    try:
        timestamp = float(raw_time)
    except (TypeError, ValueError):
        return hermes_now().isoformat()
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def default_social_request(request_key: str, *, request_type: str, flag: str) -> dict[str, Any]:
    return {
        "request_key": request_key,
        "request_type": request_type,
        "flag": flag,
        "sub_type": None,
        "status": "pending",
        "user_id": None,
        "group_id": None,
        "comment": None,
        "observed_at": None,
        "handled_at": None,
        "handled_by": None,
        "handled_via": None,
        "decision_note": None,
        "raw_event": {},
    }


def describe_social_request_state(request: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(request or {})
    request_type = str(normalized.get("request_type") or "").strip().lower()
    sub_type = str(normalized.get("sub_type") or "").strip().lower()
    status = str(normalized.get("status") or "pending").strip().lower() or "pending"
    handled_via = _normalize_optional_text(normalized.get("handled_via"))

    request_kind = "unknown"
    if request_type == "friend":
        request_kind = "friend_request"
    elif request_type == "group":
        request_kind = "group_invite" if sub_type == "invite" else "group_add_request"

    is_pending = status == "pending"
    return {
        "request_kind": request_kind,
        "status": status,
        "is_pending": is_pending,
        "handled_via": handled_via,
        "handled_automatically": bool(handled_via and handled_via.startswith("auto_")),
        "available_actions": ["approve_request", "reject_request"] if is_pending else [],
    }


def summarize_social_requests(requests: list[dict[str, Any]] | None) -> dict[str, Any]:
    items = [dict(item) for item in (requests or []) if isinstance(item, dict)]
    by_status = {status: 0 for status in VALID_SOCIAL_REQUEST_STATUSES}
    by_type = {request_type: 0 for request_type in VALID_SOCIAL_REQUEST_TYPES}
    actionable = 0

    for item in items:
        state = describe_social_request_state(item)
        status = str(state.get("status") or "").strip().lower()
        request_type = str(item.get("request_type") or "").strip().lower()
        if status in by_status:
            by_status[status] += 1
        if request_type in by_type:
            by_type[request_type] += 1
        if state["is_pending"]:
            actionable += 1

    return {
        "total": len(items),
        "actionable": actionable,
        "by_status": by_status,
        "by_type": by_type,
    }


@dataclass
class QqSocialRequestStore:
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
            return {"version": 1, "updated_at": None, "requests": {}}

        try:
            stat = self.path.stat()
            self._last_mtime_ns = int(stat.st_mtime_ns)
            self._path_exists = True
        except OSError:
            self._path_exists = False
            self._last_mtime_ns = None
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "updated_at": None, "requests": {}}

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "updated_at": None, "requests": {}}

        requests = payload.get("requests")
        if not isinstance(requests, dict):
            requests = {}
        self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
        return {
            "version": 1,
            "updated_at": payload.get("updated_at"),
            "requests": requests,
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
            "requests": self._data.get("requests") or {},
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

    def get_request(self, request_key: str) -> dict[str, Any] | None:
        normalized_key = str(request_key or "").strip()
        if not normalized_key:
            return None
        self._refresh_if_stale()
        with self._lock:
            request = dict((self._data.get("requests") or {}).get(normalized_key) or {})
        if not request:
            return None
        request.setdefault("request_key", normalized_key)
        return request

    def list_requests(
        self,
        *,
        status: str | None = None,
        request_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self._refresh_if_stale()
        with self._lock:
            keys = sorted((self._data.get("requests") or {}).keys())
        requests = [self.get_request(key) for key in keys]
        result = [item for item in requests if item is not None]
        if status:
            normalized_status = normalize_social_request_status(status)
            result = [item for item in result if item.get("status") == normalized_status]
        if request_type:
            normalized_type = normalize_social_request_type(request_type)
            result = [item for item in result if item.get("request_type") == normalized_type]
        result.sort(
            key=lambda item: (
                str(item.get("observed_at") or ""),
                str(item.get("request_key") or ""),
            ),
            reverse=True,
        )
        return result[: max(1, int(limit))]

    def record_request_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_type = normalize_social_request_type(payload.get("request_type"))
        flag = str(payload.get("flag") or "").strip()
        if not flag:
            raise ValueError("QQ social request payload is missing flag.")
        request_key = build_social_request_key(request_type, flag)
        existing = self.get_request(request_key) or default_social_request(
            request_key,
            request_type=request_type,
            flag=flag,
        )
        status = str(existing.get("status") or "pending").strip().lower()
        if status not in VALID_SOCIAL_REQUEST_STATUSES:
            status = "pending"
        updated = {
            **existing,
            "request_key": request_key,
            "request_type": request_type,
            "flag": flag,
            "sub_type": _normalize_optional_text(payload.get("sub_type")),
            "status": status,
            "user_id": _normalize_optional_numeric_text(payload.get("user_id")),
            "group_id": _normalize_optional_numeric_text(payload.get("group_id")),
            "comment": _normalize_optional_text(payload.get("comment")),
            "observed_at": _observed_at_from_payload(payload),
            "handled_via": _normalize_optional_text(existing.get("handled_via")),
            "raw_event": dict(payload),
        }
        with self._lock:
            requests = dict(self._data.get("requests") or {})
            requests[request_key] = updated
            self._data["requests"] = requests
            self._save_locked()
        stored = self.get_request(request_key)
        assert stored is not None
        return stored

    def update_request_status(
        self,
        request_key: str,
        *,
        status: str,
        handled_by: str | None = None,
        handled_via: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_request(request_key)
        if current is None:
            raise ValueError(f"QQ social request '{request_key}' does not exist.")
        normalized_status = normalize_social_request_status(status)
        now_iso = hermes_now().isoformat()
        updated = {
            **current,
            "status": normalized_status,
            "handled_at": now_iso,
            "handled_by": _normalize_optional_text(handled_by),
            "handled_via": _normalize_optional_text(handled_via),
            "decision_note": _normalize_optional_text(note),
        }
        with self._lock:
            requests = dict(self._data.get("requests") or {})
            requests[current["request_key"]] = updated
            self._data["requests"] = requests
            self._save_locked()
        stored = self.get_request(request_key)
        assert stored is not None
        return stored


_store_singleton: QqSocialRequestStore | None = None
_store_singleton_lock = threading.Lock()


def get_social_request_store() -> QqSocialRequestStore:
    global _store_singleton
    path = _store_path()
    with _store_singleton_lock:
        if _store_singleton is None or _store_singleton.path != path:
            _store_singleton = QqSocialRequestStore(path=path)
    return _store_singleton


def record_social_request_event(payload: dict[str, Any]) -> dict[str, Any]:
    return get_social_request_store().record_request_event(payload)


def get_social_request(request_key: str) -> dict[str, Any] | None:
    return get_social_request_store().get_request(request_key)


def list_social_requests(
    *,
    status: str | None = None,
    request_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return get_social_request_store().list_requests(
        status=status,
        request_type=request_type,
        limit=limit,
    )


def update_social_request_status(
    request_key: str,
    *,
    status: str,
    handled_by: str | None = None,
    handled_via: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return get_social_request_store().update_request_status(
        request_key,
        status=status,
        handled_by=handled_by,
        handled_via=handled_via,
        note=note,
    )
