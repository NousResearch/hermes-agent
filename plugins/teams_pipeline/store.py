"""Durable local state for the Teams pipeline plugin."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from functools import partialmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home


DEFAULT_TEAMS_PIPELINE_STORE_FILENAME = "teams_pipeline_store.json"
# Persisted top-level buckets; the on-disk JSON shape must stay stable for existing user stores.
_BUCKETS = ("subscriptions", "notification_receipts", "event_timestamps", "jobs", "sink_records")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_teams_pipeline_store_path(path: str | Path | None = None) -> Path:
    explicit = str(path).strip() if path is not None else ""
    env_path = os.getenv("MSGRAPH_WEBHOOK_STORE_PATH", "").strip()
    return Path(explicit or env_path) if (explicit or env_path) else get_hermes_home() / DEFAULT_TEAMS_PIPELINE_STORE_FILENAME


class TeamsPipelineStore:
    """JSON-backed durable store for Teams pipeline state; every write is an atomic temp-file replace."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state: Dict[str, Dict[str, Any]] = {bucket: {} for bucket in _BUCKETS}
        self._load()

    def _load(self) -> None:
        with self._lock:
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}") if self.path.exists() else None
            if isinstance(data, dict):
                self._state = {bucket: dict(data.get(bucket) or {}) for bucket in _BUCKETS}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=str(self.path.parent), delete=False) as tmp:
            json.dump(self._state, tmp, indent=2, sort_keys=True)
            tmp.flush()
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)

    def _list(self, bucket: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return deepcopy(self._state[bucket])

    def _get(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._state[bucket].get(key)
            return deepcopy(record) if isinstance(record, dict) else None

    def _upsert(self, bucket: str, id_field: str, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ``payload`` over the existing record, stamping ``id_field`` and created/updated timestamps."""
        with self._lock:
            existing = self._state[bucket].get(key, {})
            merged = {**existing, **deepcopy(payload)}
            merged[id_field] = key
            merged.setdefault("created_at", existing.get("created_at") or _utc_now_iso())
            merged["updated_at"] = _utc_now_iso()
            self._state[bucket][key] = merged
            self._persist()
            return deepcopy(merged)

    list_subscriptions = partialmethod(_list, "subscriptions")
    get_subscription = partialmethod(_get, "subscriptions")
    upsert_subscription = partialmethod(_upsert, "subscriptions", "subscription_id")
    list_jobs = partialmethod(_list, "jobs")
    get_job = partialmethod(_get, "jobs")
    upsert_job = partialmethod(_upsert, "jobs", "job_id")
    get_sink_record = partialmethod(_get, "sink_records")
    upsert_sink_record = partialmethod(_upsert, "sink_records", "sink_key")

    def delete_subscription(self, subscription_id: str) -> bool:
        with self._lock:
            if self._state["subscriptions"].pop(subscription_id, None) is None:
                return False
            self._persist()
            return True

    @classmethod
    def build_notification_receipt_key(cls, notification: Dict[str, Any]) -> str:
        if explicit_id := notification.get("id"):
            return f"id:{explicit_id}"
        canonical = json.dumps(notification, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    def record_notification_receipt(self, receipt_key: str, payload: Optional[Dict[str, Any]] = None, *, received_at: Optional[str] = None) -> bool:
        """Record a receipt once; returns False when the key was already seen (duplicate delivery)."""
        with self._lock:
            if receipt_key in self._state["notification_receipts"]:
                return False
            self._state["notification_receipts"][receipt_key] = {"received_at": received_at or _utc_now_iso(),
                                                                 "payload": deepcopy(payload) if isinstance(payload, dict) else payload}
            self._persist()
            return True

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {bucket: len(self._state[bucket]) for bucket in _BUCKETS}
