"""JSON-file backed dedup store implementing lark_oapi DedupStore Protocol.

Backward compatibility: older dedup files were saved as a plain list
(``[message_id, ...]``); the current schema is a dict
(``{message_id: timestamp}``). ``__init__`` reads either shape and can
namespace legacy Feishu message ids into SDK dedup keys; writes always use
the dict form.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

# Persistent dedup-store tunables — module-level defaults consumed by
# ``JsonFileDedupStore`` and the SDK ``DedupConfig`` builder.
_FEISHU_DEDUP_DEFAULT_TTL_SECONDS = 24 * 3600


class JsonFileDedupStore:
    """Hermes implementation of ``lark_oapi.channel.normalize.DedupStore``.

    Synchronous contract — SDK ``Deduper`` calls ``seen`` / ``mark`` from a
    background thread. State is held in an ``OrderedDict[str, float]``
    (key → expiry_ts) under a ``threading.Lock``. Writes flush immediately
    via atomic JSON file rename (``tmp + os.replace``). Backward-compatible
    loaders cover the legacy plain-list format (``["msg_id_1", ...]``) and
    a transitional dict-of-timestamps format
    (``{"message_ids": {id: wallclock_ts}}``).

    Construction is keyword-only. After construction, callers interact only
    via ``seen`` / ``mark`` / ``flush``. Hermes injects this into
    ``FeishuChannel(dedup_store=...)`` once per adapter lifetime;
    ``FeishuAdapter.disconnect()`` calls ``flush()`` to ensure pending
    entries hit disk.
    """

    def __init__(
        self,
        *,
        path: Path,
        max_entries: int,
        default_ttl_seconds: int = _FEISHU_DEDUP_DEFAULT_TTL_SECONDS,
        account_id: str = "",
    ) -> None:
        self._path = Path(path)
        self._max = int(max_entries)
        self._default_ttl = int(default_ttl_seconds)
        self._account_id = str(account_id or "").strip()
        self._lock = threading.Lock()
        self._dirty_count = 0
        # Lazy-load from disk; loader handles missing / corrupt files
        # gracefully so ctor never raises on cold start.
        self._data: "OrderedDict[str, float]" = self._load_from_disk()

    # -- DedupStore Protocol --------------------------------------------------

    def seen(self, key: str) -> bool:
        with self._lock:
            exp = self._data.get(key)
            if exp is None:
                return False
            if exp <= time.time():
                self._data.pop(key, None)
                return False
            self._data.move_to_end(key)
            return True

    def mark(self, key: str, ttl_seconds: int) -> None:
        with self._lock:
            self._data[key] = time.time() + max(int(ttl_seconds), 0)
            self._data.move_to_end(key)
            self._evict_locked()
            self._dirty_count += 1
            self._flush_locked()

    # -- Lifecycle ------------------------------------------------------------

    def flush(self) -> None:
        """Force pending writes to disk synchronously. Call from
        ``FeishuAdapter.disconnect()`` to ensure clean shutdown."""
        with self._lock:
            self._flush_locked()

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    # -- Internals ------------------------------------------------------------

    def _load_from_disk(self) -> "OrderedDict[str, float]":
        if not self._path.exists():
            return OrderedDict()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "[Feishu dedup] failed to load %s — starting empty store",
                self._path, exc_info=True,
            )
            return OrderedDict()
        now = time.time()
        if isinstance(raw, list):
            # Legacy format: plain list of IDs, no timestamps.
            return self._prune_loaded(
                OrderedDict(
                    (self._legacy_message_key(str(k)), now + self._default_ttl)
                    for k in raw if isinstance(k, str) and k.strip()
                )
            )
        if isinstance(raw, dict):
            # Transitional format: {"message_ids": {id: wallclock_ts}}.
            inner = raw.get("message_ids") if "message_ids" in raw else None
            if isinstance(inner, dict):
                # Tolerate malformed timestamp values: skip the bad key,
                # keep loading the rest. A single corrupt entry must not
                # crash the store on cold start.
                out: "OrderedDict[str, float]" = OrderedDict()
                for k, v in inner.items():
                    if not (isinstance(k, str) and k.strip()):
                        continue
                    try:
                        out[self._legacy_message_key(str(k))] = (
                            float(v) + self._default_ttl
                        )
                    except (TypeError, ValueError):
                        logger.warning(
                            "[Feishu dedup] skipping malformed timestamp "
                            "for key %r in %s (value=%r)",
                            k, self._path, v,
                        )
                return self._prune_loaded(out)
            if isinstance(inner, list):
                # Even older variant: {"message_ids": ["om_a", ...]}.
                return self._prune_loaded(
                    OrderedDict(
                        (self._legacy_message_key(str(k)), now + self._default_ttl)
                        for k in inner if isinstance(k, str) and k.strip()
                    )
                )
            # Current format: {key: expiry_ts}.
            out = OrderedDict()
            for k, v in raw.items():
                if not (isinstance(k, str) and k.strip()):
                    continue
                try:
                    out[self._legacy_message_key(str(k))] = float(v)
                except (TypeError, ValueError):
                    logger.warning(
                        "[Feishu dedup] skipping malformed timestamp "
                        "for key %r in %s (value=%r)",
                        k, self._path, v,
                    )
            return self._prune_loaded(out)
        return OrderedDict()

    def _legacy_message_key(self, key: str) -> str:
        if self._account_id and key.startswith("om_"):
            return f"msg:{self._account_id}:{key}"
        return key

    def _prune_loaded(
        self, data: "OrderedDict[str, float]"
    ) -> "OrderedDict[str, float]":
        now = time.time()
        valid = OrderedDict(
            (key, exp)
            for key, exp in data.items()
            if exp > now
        )
        while len(valid) > self._max:
            valid.popitem(last=False)
        return valid

    def _evict_locked(self) -> None:
        # Drop oldest entries until under cap. ``mark()`` does
        # ``move_to_end``, so insertion order matches LRU; popping
        # ``last=False`` evicts the least-recently-used entry.
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def _flush_locked(self) -> None:
        if self._dirty_count == 0:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(dict(self._data), ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)   # POSIX atomic rename.
            self._dirty_count = 0
        except OSError:
            logger.warning(
                "[Feishu dedup] failed to flush to %s",
                self._path, exc_info=True,
            )
