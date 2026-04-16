"""Platform-neutral group policy store keyed by ``scope_key``."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_time import now as hermes_now
from utils import atomic_json_write


POLICY_STORE_FILENAME = "qq_group_policies.json"
VALID_GROUP_POLICY_MODES = ("default", "collect_only", "project_mode", "disabled")
LEGACY_DEFAULT_PLATFORM = "qq_napcat"


def _policy_store_path() -> Path:
    return get_hermes_home() / POLICY_STORE_FILENAME


def group_policy_store_path() -> Path:
    return _policy_store_path()


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


def normalize_group_policy_mode(mode: Any) -> str:
    normalized = str(mode or "default").strip().lower()
    if normalized not in VALID_GROUP_POLICY_MODES:
        raise ValueError(
            "Unsupported group policy mode. "
            "Use 'default', 'collect_only', 'project_mode', or 'disabled'."
        )
    return normalized


def normalize_group_scope_key(platform_or_scope: str, chat_id: str | None = None) -> str:
    platform_value = str(platform_or_scope or "").strip()
    if chat_id is None:
        if ":" not in platform_value:
            raise ValueError("scope_key must be in '<platform>:<chat_id>' format")
        platform, normalized_chat_id = split_group_scope_key(platform_value)
        return f"{platform}:{normalized_chat_id}"
    platform = platform_value
    normalized_chat_id = str(chat_id or "").strip()
    if not platform:
        raise ValueError("platform is required")
    if not normalized_chat_id:
        raise ValueError("chat_id is required")
    return f"{platform}:{normalized_chat_id}"


def split_group_scope_key(scope_key: str) -> tuple[str, str]:
    normalized = str(scope_key or "").strip()
    if not normalized or ":" not in normalized:
        raise ValueError("scope_key must be in '<platform>:<chat_id>' format")
    platform, chat_id = normalized.split(":", 1)
    platform = platform.strip()
    chat_id = chat_id.strip()
    if not platform or not chat_id:
        raise ValueError("scope_key must be in '<platform>:<chat_id>' format")
    return platform, chat_id


def default_scope_policy(scope_key: str) -> dict[str, Any]:
    normalized_scope_key = normalize_group_scope_key(scope_key)
    platform, chat_id = split_group_scope_key(normalized_scope_key)
    return {
        "scope_key": normalized_scope_key,
        "platform": platform,
        "chat_id": chat_id,
        "mode": "default",
        "archive_enabled": False,
        "daily_report_enabled": False,
        "daily_report_target": None,
        "manual_report_target": None,
        "purge_raw_after_rollup": True,
        "chat_name": "",
        "notes": "",
        "updated_at": None,
        "updated_by": None,
    }


@dataclass
class GroupPolicyStore:
    path: Path | None = None
    legacy_default_platform: str = LEGACY_DEFAULT_PLATFORM

    def __post_init__(self) -> None:
        self.path = Path(self.path or _policy_store_path())
        self._lock = threading.RLock()
        self._last_mtime_ns: int | None = None
        self._path_exists = False
        self._reload_check_interval_seconds = 0.0
        self._next_reload_check_at = 0.0
        self._data = self._load_from_disk()

    def _normalize_loaded_scope_key(self, raw_key: Any) -> str | None:
        key_text = str(raw_key or "").strip()
        if not key_text:
            return None
        if ":" in key_text:
            try:
                return normalize_group_scope_key(key_text)
            except ValueError:
                return None
        try:
            return normalize_group_scope_key(self.legacy_default_platform, key_text)
        except ValueError:
            return None

    def _normalize_loaded_groups(self, groups: Any) -> dict[str, Any]:
        if not isinstance(groups, dict):
            return {}
        normalized: dict[str, Any] = {}
        legacy_candidates: dict[str, Any] = {}
        for raw_key, raw_value in groups.items():
            normalized_scope_key = self._normalize_loaded_scope_key(raw_key)
            if not normalized_scope_key:
                continue
            value = dict(raw_value or {}) if isinstance(raw_value, dict) else {}
            if ":" in str(raw_key or ""):
                normalized[normalized_scope_key] = value
            elif normalized_scope_key not in normalized:
                legacy_candidates[normalized_scope_key] = value
        for scope_key, value in legacy_candidates.items():
            normalized.setdefault(scope_key, value)
        return normalized

    def _load_from_disk(self) -> dict[str, Any]:
        if not self.path or not self.path.exists():
            self._path_exists = False
            self._last_mtime_ns = None
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "updated_at": None, "groups": {}}

        try:
            stat = self.path.stat()
            self._last_mtime_ns = int(stat.st_mtime_ns)
            self._path_exists = True
        except OSError:
            self._path_exists = False
            self._last_mtime_ns = None
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "updated_at": None, "groups": {}}
        try:
            import json

            payload = json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return {"version": 1, "updated_at": None, "groups": {}}
        normalized_groups = self._normalize_loaded_groups(payload.get("groups"))
        self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
        return {
            "version": 1,
            "updated_at": payload.get("updated_at"),
            "groups": normalized_groups,
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
            "groups": self._data.get("groups") or {},
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

    def get_policy(self, scope_key: str) -> dict[str, Any]:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        self._refresh_if_stale()
        with self._lock:
            stored = dict((self._data.get("groups") or {}).get(normalized_scope_key) or {})
        policy = default_scope_policy(normalized_scope_key)
        policy.update(
            {
                "mode": normalize_group_policy_mode(stored.get("mode", policy["mode"])),
                "archive_enabled": _normalize_bool(
                    stored.get("archive_enabled"),
                    default=policy["archive_enabled"],
                ),
                "daily_report_enabled": _normalize_bool(
                    stored.get("daily_report_enabled"),
                    default=policy["daily_report_enabled"],
                ),
                "daily_report_target": _normalize_optional_text(stored.get("daily_report_target")),
                "manual_report_target": _normalize_optional_text(stored.get("manual_report_target")),
                "purge_raw_after_rollup": _normalize_bool(
                    stored.get("purge_raw_after_rollup"),
                    default=policy["purge_raw_after_rollup"],
                ),
                "chat_name": str(stored.get("chat_name") or stored.get("group_name") or "").strip(),
                "notes": str(stored.get("notes") or "").strip(),
                "updated_at": stored.get("updated_at"),
                "updated_by": stored.get("updated_by"),
            }
        )
        return policy

    def list_policies(self) -> list[dict[str, Any]]:
        self._refresh_if_stale()
        with self._lock:
            scope_keys = sorted((self._data.get("groups") or {}).keys())
        return [self.get_policy(scope_key) for scope_key in scope_keys]

    def has_policy(self, scope_key: str) -> bool:
        normalized_scope_key = str(scope_key or "").strip()
        if not normalized_scope_key:
            return False
        try:
            normalized_scope_key = normalize_group_scope_key(normalized_scope_key)
        except ValueError:
            return False
        self._refresh_if_stale()
        with self._lock:
            return normalized_scope_key in (self._data.get("groups") or {})

    def set_policy(
        self,
        scope_key: str,
        *,
        mode: str,
        archive_enabled: bool | None = None,
        daily_report_enabled: bool | None = None,
        daily_report_target: str | None = None,
        manual_report_target: str | None = None,
        purge_raw_after_rollup: bool | None = None,
        chat_name: str | None = None,
        notes: str | None = None,
        updated_by: str | None = None,
    ) -> dict[str, Any]:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        normalized_mode = normalize_group_policy_mode(mode)
        self._refresh_if_stale()
        with self._lock:
            groups = dict(self._data.get("groups") or {})
            current = default_scope_policy(normalized_scope_key)
            current.update(dict(groups.get(normalized_scope_key) or {}))
            current_chat_name = str(current.get("chat_name") or current.get("group_name") or "").strip()
            archive_default = (
                current["archive_enabled"]
                if normalized_scope_key in groups
                else (normalized_mode == "collect_only")
            )
            daily_default = current["daily_report_enabled"] if normalized_scope_key in groups else False
            daily_target_value = (
                current.get("daily_report_target") if daily_report_target is None else daily_report_target
            )
            manual_target_value = (
                current.get("manual_report_target") if manual_report_target is None else manual_report_target
            )
            daily_target = _normalize_optional_text(daily_target_value)
            manual_target = _normalize_optional_text(manual_target_value)
            archive_value = _normalize_bool(archive_enabled, default=archive_default)
            daily_enabled = _normalize_bool(daily_report_enabled, default=daily_default)
            purge_value = _normalize_bool(
                purge_raw_after_rollup,
                default=_normalize_bool(current.get("purge_raw_after_rollup"), default=True),
            )

            if normalized_mode == "disabled":
                archive_value = False
                daily_enabled = False
            elif daily_enabled or daily_target or manual_target:
                archive_value = True

            policy = {
                "mode": normalized_mode,
                "archive_enabled": archive_value,
                "daily_report_enabled": daily_enabled,
                "daily_report_target": daily_target,
                "manual_report_target": manual_target,
                "purge_raw_after_rollup": purge_value,
                "chat_name": current_chat_name if chat_name is None else str(chat_name or "").strip(),
                "notes": current["notes"] if notes is None else str(notes or "").strip(),
                "updated_at": hermes_now().isoformat(),
                "updated_by": str(updated_by or "").strip() or None,
            }
            if (
                policy["mode"] == "default"
                and not policy["archive_enabled"]
                and not policy["daily_report_enabled"]
                and not policy["daily_report_target"]
                and not policy["manual_report_target"]
                and bool(policy["purge_raw_after_rollup"])
                and not policy["chat_name"]
                and not policy["notes"]
            ):
                groups.pop(normalized_scope_key, None)
            else:
                groups[normalized_scope_key] = policy
            self._data["groups"] = groups
            self._save_locked()
        return self.get_policy(normalized_scope_key)

    def clear_policy(self, scope_key: str) -> dict[str, Any]:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        with self._lock:
            groups = dict(self._data.get("groups") or {})
            groups.pop(normalized_scope_key, None)
            self._data["groups"] = groups
            self._save_locked()
        return self.get_policy(normalized_scope_key)


_policy_store_singleton: GroupPolicyStore | None = None
_policy_store_singleton_lock = threading.Lock()


def get_policy_store() -> GroupPolicyStore:
    global _policy_store_singleton
    path = _policy_store_path()
    with _policy_store_singleton_lock:
        if _policy_store_singleton is None or _policy_store_singleton.path != path:
            _policy_store_singleton = GroupPolicyStore(path=path)
    return _policy_store_singleton


def get_scope_policy(scope_key: str) -> dict[str, Any]:
    return get_policy_store().get_policy(scope_key)


def list_scope_policies() -> list[dict[str, Any]]:
    return get_policy_store().list_policies()


def has_scope_policy(scope_key: str) -> bool:
    return get_policy_store().has_policy(scope_key)


def set_scope_policy(scope_key: str, **kwargs: Any) -> dict[str, Any]:
    return get_policy_store().set_policy(scope_key, **kwargs)


def clear_scope_policy(scope_key: str) -> dict[str, Any]:
    return get_policy_store().clear_policy(scope_key)
