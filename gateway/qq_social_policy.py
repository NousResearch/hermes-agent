"""Account-wide QQ/NapCat social auto-handling policy store."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_time import now as hermes_now
from utils import atomic_json_write


QQ_SOCIAL_POLICY_STORE_FILENAME = "qq_social_policy.json"
_AUTO_APPROVAL_SCOPE_FIELDS = (
    ("auto_approve_friend_requests", "friend_requests"),
    ("auto_approve_group_add_requests", "group_add_requests"),
    ("auto_approve_group_invites", "group_invites"),
)


def _store_path() -> Path:
    return get_hermes_home() / QQ_SOCIAL_POLICY_STORE_FILENAME


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


def default_social_policy() -> dict[str, Any]:
    return {
        "auto_approve_friend_requests": False,
        "auto_approve_group_add_requests": False,
        "auto_approve_group_invites": False,
        "notify_target": None,
        "notes": "",
        "updated_at": None,
        "updated_by": None,
    }


def describe_social_policy_state(policy: dict[str, Any] | None) -> dict[str, Any]:
    normalized = default_social_policy()
    normalized.update(dict(policy or {}))
    enabled_scopes = [
        scope_name
        for field_name, scope_name in _AUTO_APPROVAL_SCOPE_FIELDS
        if bool(normalized.get(field_name))
    ]
    notify_target = _normalize_optional_text(normalized.get("notify_target"))
    return {
        "auto_approval_enabled": bool(enabled_scopes),
        "enabled_scope_count": len(enabled_scopes),
        "enabled_scopes": enabled_scopes,
        "notify_configured": bool(notify_target),
        "notify_target": notify_target,
        "updated_at": normalized.get("updated_at"),
        "updated_by": normalized.get("updated_by"),
        "notes": str(normalized.get("notes") or "").strip(),
    }


@dataclass
class QqSocialPolicyStore:
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
            return {"version": 1, "policy": default_social_policy()}

        try:
            stat = self.path.stat()
            self._last_mtime_ns = int(stat.st_mtime_ns)
            self._path_exists = True
        except OSError:
            self._path_exists = False
            self._last_mtime_ns = None
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "policy": default_social_policy()}

        try:
            import json

            payload = json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "policy": default_social_policy()}

        policy = payload.get("policy")
        if not isinstance(policy, dict):
            policy = default_social_policy()
        self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
        return {
            "version": 1,
            "policy": policy,
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
            "policy": self._data.get("policy") or default_social_policy(),
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

    def get_policy(self) -> dict[str, Any]:
        self._refresh_if_stale()
        with self._lock:
            stored = dict(self._data.get("policy") or {})
        policy = default_social_policy()
        policy.update(
            {
                "auto_approve_friend_requests": _normalize_bool(
                    stored.get("auto_approve_friend_requests"),
                    default=policy["auto_approve_friend_requests"],
                ),
                "auto_approve_group_add_requests": _normalize_bool(
                    stored.get("auto_approve_group_add_requests"),
                    default=policy["auto_approve_group_add_requests"],
                ),
                "auto_approve_group_invites": _normalize_bool(
                    stored.get("auto_approve_group_invites"),
                    default=policy["auto_approve_group_invites"],
                ),
                "notify_target": _normalize_optional_text(stored.get("notify_target")),
                "notes": str(stored.get("notes") or "").strip(),
                "updated_at": stored.get("updated_at"),
                "updated_by": stored.get("updated_by"),
            }
        )
        return policy

    def set_policy(
        self,
        *,
        auto_approve_friend_requests: bool | None = None,
        auto_approve_group_add_requests: bool | None = None,
        auto_approve_group_invites: bool | None = None,
        notify_target: str | None = None,
        notes: str | None = None,
        updated_by: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_policy()
        now_iso = hermes_now().isoformat()
        updated = {
            "auto_approve_friend_requests": current["auto_approve_friend_requests"] if auto_approve_friend_requests is None else bool(auto_approve_friend_requests),
            "auto_approve_group_add_requests": current["auto_approve_group_add_requests"] if auto_approve_group_add_requests is None else bool(auto_approve_group_add_requests),
            "auto_approve_group_invites": current["auto_approve_group_invites"] if auto_approve_group_invites is None else bool(auto_approve_group_invites),
            "notify_target": current["notify_target"] if notify_target is None else _normalize_optional_text(notify_target),
            "notes": current["notes"] if notes is None else str(notes or "").strip(),
            "updated_at": now_iso,
            "updated_by": _normalize_optional_text(updated_by),
        }
        with self._lock:
            self._data["policy"] = updated
            self._save_locked()
        return self.get_policy()

    def clear_policy(self, *, updated_by: str | None = None) -> dict[str, Any]:
        cleared = default_social_policy()
        cleared["updated_at"] = hermes_now().isoformat()
        cleared["updated_by"] = _normalize_optional_text(updated_by)
        with self._lock:
            self._data["policy"] = cleared
            self._save_locked()
        return self.get_policy()


_store_singleton: QqSocialPolicyStore | None = None
_store_singleton_lock = threading.Lock()


def get_social_policy_store() -> QqSocialPolicyStore:
    global _store_singleton
    path = _store_path()
    with _store_singleton_lock:
        if _store_singleton is None or _store_singleton.path != path:
            _store_singleton = QqSocialPolicyStore(path=path)
    return _store_singleton


def get_social_policy() -> dict[str, Any]:
    return get_social_policy_store().get_policy()


def set_social_policy(**kwargs: Any) -> dict[str, Any]:
    return get_social_policy_store().set_policy(**kwargs)


def clear_social_policy(*, updated_by: str | None = None) -> dict[str, Any]:
    return get_social_policy_store().clear_policy(updated_by=updated_by)
