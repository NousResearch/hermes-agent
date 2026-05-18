"""Durable local state for the Google Drive artifact pipeline."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from hermes_constants import get_hermes_home


DEFAULT_GOOGLE_DRIVE_PIPELINE_STORE_FILENAME = "google_drive_pipeline_store.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_google_drive_pipeline_store_path(path: str | Path | None = None) -> Path:
    if path is not None:
        explicit = str(path).strip()
        if explicit:
            return Path(explicit)

    env_path = os.getenv("GOOGLE_DRIVE_PIPELINE_STORE_PATH", "").strip()
    if env_path:
        return Path(env_path)

    return get_hermes_home() / DEFAULT_GOOGLE_DRIVE_PIPELINE_STORE_FILENAME


class GoogleDrivePipelineStore:
    """JSON-backed durable store for Drive publish records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state: dict[str, dict[str, Any]] = {
            "records": {},
            "source_index": {},
            "folder_cache": {},
        }
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            if not isinstance(data, dict):
                return
            self._state["records"] = dict(data.get("records") or {})
            self._state["source_index"] = dict(data.get("source_index") or {})
            self._state["folder_cache"] = dict(data.get("folder_cache") or {})

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=str(self.path.parent), delete=False) as tmp:
            json.dump(self._state, tmp, indent=2, sort_keys=True)
            tmp.flush()
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)

    def list_records(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return deepcopy(self._state["records"])

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._state["records"].get(record_id)
            return deepcopy(record) if isinstance(record, dict) else None

    def upsert_record(self, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            existing = self._state["records"].get(record_id, {})
            merged = {**existing, **deepcopy(payload)}
            merged["record_id"] = record_id
            merged.setdefault("created_at", existing.get("created_at") or _utc_now_iso())
            merged["updated_at"] = _utc_now_iso()
            self._state["records"][record_id] = merged

            source_key = str(merged.get("source_key") or "").strip()
            if source_key:
                self._state["source_index"][source_key] = record_id

            folder_key = str(merged.get("folder_cache_key") or "").strip()
            folder = merged.get("folder")
            if folder_key and isinstance(folder, dict):
                self._state["folder_cache"][folder_key] = deepcopy(folder)

            self._persist()
            return deepcopy(merged)

    def get_record_by_source_key(self, source_key: str) -> dict[str, Any] | None:
        with self._lock:
            record_id = self._state["source_index"].get(source_key)
            if not record_id:
                return None
            record = self._state["records"].get(record_id)
            return deepcopy(record) if isinstance(record, dict) else None

    def get_cached_folder(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._state["folder_cache"].get(cache_key)
            return deepcopy(record) if isinstance(record, dict) else None
