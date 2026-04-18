"""Dynamic employee-route store persisted under HERMES_HOME."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_time import now as hermes_now
from utils import atomic_json_write

from gateway.config import Platform
from gateway.employee_route_schema import normalize_employee_route


EMPLOYEE_ROUTE_STORE_FILENAME = "employee_routes.json"


def _store_path() -> Path:
    return get_hermes_home() / EMPLOYEE_ROUTE_STORE_FILENAME


def employee_route_store_path() -> Path:
    return _store_path()


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


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


def _default_store_payload() -> dict[str, Any]:
    return {"version": 1, "updated_at": None, "platforms": {}}


def _normalize_platform_value(platform: Platform | str) -> str:
    if isinstance(platform, Platform):
        return platform.value
    return Platform(str(platform)).value


def _normalize_stored_route(route: dict[str, Any]) -> dict[str, Any] | None:
    normalized = normalize_employee_route(route)
    if normalized is None:
        return None
    normalized["updated_at"] = route.get("updated_at")
    normalized["updated_by"] = _normalize_optional_text(route.get("updated_by"))
    normalized["enabled"] = _normalize_bool(route.get("enabled"), default=True)
    return normalized


@dataclass
class EmployeeRouteStore:
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
            return _default_store_payload()

        try:
            stat = self.path.stat()
            self._last_mtime_ns = int(stat.st_mtime_ns)
            self._path_exists = True
        except OSError:
            self._path_exists = False
            self._last_mtime_ns = None
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return _default_store_payload()

        try:
            import json

            payload = json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
            return _default_store_payload()

        platforms = payload.get("platforms")
        if not isinstance(platforms, dict):
            platforms = {}
        self._next_reload_check_at = time.monotonic() + self._reload_check_interval_seconds
        return {
            "version": 1,
            "updated_at": payload.get("updated_at"),
            "platforms": platforms,
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
            "platforms": self._data.get("platforms") or {},
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

    def list_routes(self, platform: Platform | str) -> list[dict[str, Any]]:
        platform_key = _normalize_platform_value(platform)
        self._refresh_if_stale()
        with self._lock:
            routes = dict(((self._data.get("platforms") or {}).get(platform_key) or {}))
        normalized: list[dict[str, Any]] = []
        for worker_name in sorted(routes.keys()):
            item = routes.get(worker_name)
            if not isinstance(item, dict):
                continue
            normalized_item = _normalize_stored_route(item)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return normalized

    def set_route(
        self,
        platform: Platform | str,
        *,
        worker_name: str,
        aliases: list[str] | None = None,
        preloaded_skills: list[str] | None = None,
        match_modes: list[str] | tuple[str, ...] | None = None,
        action_terms: list[str] | None = None,
        subject_terms: list[str] | None = None,
        pain_terms: list[str] | None = None,
        enabled: bool = True,
        updated_by: str | None = None,
    ) -> dict[str, Any]:
        platform_key = _normalize_platform_value(platform)
        route = _normalize_stored_route(
            {
                "worker_name": worker_name,
                "aliases": aliases,
                "preloaded_skills": preloaded_skills,
                "match_modes": list(match_modes) if match_modes is not None else None,
                "action_terms": action_terms,
                "subject_terms": subject_terms,
                "pain_terms": pain_terms,
                "enabled": enabled,
                "updated_at": hermes_now().isoformat(),
                "updated_by": updated_by,
            }
        )
        if route is None:
            raise ValueError("worker_name is required")

        persisted_route = {
            "worker_name": route["worker_name"],
            "aliases": route["aliases"],
            "preloaded_skills": route["preloaded_skills"],
            "match_modes": list(route["match_modes"]),
            "action_terms": list(route["action_terms"]),
            "subject_terms": list(route["subject_terms"]),
            "pain_terms": list(route["pain_terms"]),
            "enabled": route["enabled"],
            "updated_at": route["updated_at"],
            "updated_by": route["updated_by"],
        }

        self._refresh_if_stale()
        with self._lock:
            platforms = dict(self._data.get("platforms") or {})
            platform_routes = dict((platforms.get(platform_key) or {}))
            platform_routes[route["worker_name"]] = persisted_route
            platforms[platform_key] = platform_routes
            self._data["platforms"] = platforms
            self._save_locked()
        return route

    def clear_route(
        self,
        platform: Platform | str,
        worker_name: str,
        *,
        updated_by: str | None = None,
    ) -> dict[str, Any] | None:
        del updated_by
        platform_key = _normalize_platform_value(platform)
        normalized_worker_name = str(worker_name or "").strip()
        self._refresh_if_stale()
        with self._lock:
            platforms = dict(self._data.get("platforms") or {})
            platform_routes = dict((platforms.get(platform_key) or {}))
            removed = platform_routes.pop(normalized_worker_name, None)
            if not platform_routes:
                platforms.pop(platform_key, None)
            else:
                platforms[platform_key] = platform_routes
            self._data["platforms"] = platforms
            self._save_locked()
        if not isinstance(removed, dict):
            return None
        return _normalize_stored_route(removed)


_store_singleton: EmployeeRouteStore | None = None
_store_singleton_lock = threading.Lock()


def get_employee_route_store() -> EmployeeRouteStore:
    global _store_singleton
    path = _store_path()
    with _store_singleton_lock:
        if _store_singleton is None or _store_singleton.path != path:
            _store_singleton = EmployeeRouteStore(path=path)
    return _store_singleton


def list_employee_routes(platform: Platform | str) -> list[dict[str, Any]]:
    return get_employee_route_store().list_routes(platform)


def set_employee_route(platform: Platform | str, **kwargs: Any) -> dict[str, Any]:
    return get_employee_route_store().set_route(platform, **kwargs)


def clear_employee_route_store(
    platform: Platform | str,
    worker_name: str,
    *,
    updated_by: str | None = None,
) -> dict[str, Any] | None:
    return get_employee_route_store().clear_route(
        platform,
        worker_name,
        updated_by=updated_by,
    )
