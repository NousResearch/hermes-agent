"""Fetch-first merge with keep-both conflicts for cognition memory.

Push never blindly overwrites: it fetches the committed state for the entry's
ref first and compares it against the base the writer saw. When the base still
matches, the push applies. When something else committed in between, both
versions are kept — the existing entry stays under its ref and the incoming
one lands under a deterministic conflict ref
``{ref}-conflict-{unix_ts}`` — so no write ever loses another writer's facts.

Concurrency: one single-writer lock per entry id serializes concurrent pushes
to the same ref. Dream (read-only synthesis) sees committed state only: it
reads through :func:`committed_snapshot`, which can never observe the push
queue or in-flight merges.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .provenance import validate_provenance


def conflict_ref(ref: str, now: Optional[float] = None) -> str:
    """Deterministic conflict ref: ``{ref}-conflict-{unix_ts}``."""
    return "%s-conflict-%d" % (ref, int(now if now is not None else time.time()))


class EntryWriteLock:
    """Single-writer lock per entry id (re-entrant per thread)."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: Dict[str, threading.RLock] = {}

    @contextmanager
    def hold(self, entry_id: str) -> Iterator[None]:
        """Hold the exclusive write lock for ``entry_id``."""
        with self._guard:
            lock = self._locks.get(entry_id)
            if lock is None:
                lock = self._locks[entry_id] = threading.RLock()
        with lock:
            yield


class CommitLog:
    """File-backed committed state: ref -> entry (atomic JSON persistence)."""

    def __init__(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._entries: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read().strip()
            if raw:
                self._entries = json.loads(raw)

    def _persist_locked(self) -> None:
        directory = os.path.dirname(self._path)
        fd, tmp = tempfile.mkstemp(dir=directory or ".", prefix=".commitlog-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._entries, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get(self, ref: str) -> Optional[Dict[str, Any]]:
        """Committed entry for ``ref`` (deep copy; None when absent)."""
        with self._lock:
            entry = self._entries.get(ref)
            return copy.deepcopy(entry) if entry is not None else None

    def refs(self) -> List[str]:
        """All committed refs."""
        with self._lock:
            return sorted(self._entries)

    def entries_for_repo(self, origin_repo: str) -> List[Dict[str, Any]]:
        """Committed entries routed to one internal repo (multi-repo routing)."""
        with self._lock:
            return [
                copy.deepcopy(entry)
                for entry in self._entries.values()
                if isinstance(entry.get("provenance"), dict)
                and entry["provenance"].get("origin_repo") == origin_repo
            ]

    def _apply_locked(self, entry: Dict[str, Any]) -> None:
        validate_provenance(entry)
        self._entries[entry["ref"]] = copy.deepcopy(entry)
        self._persist_locked()

    def remove(self, ref: str) -> Optional[Dict[str, Any]]:
        """Remove a committed entry; returns the removed copy (None if absent)."""
        with self._lock:
            entry = self._entries.pop(ref, None)
            if entry is not None:
                self._persist_locked()
            return entry


# One process-wide lock registry: concurrent pushes to the same ref from any
# CommitLog instance serialize on the entry id, not the instance.
_WRITE_LOCKS = EntryWriteLock()


def fetch_first_push(
    log: CommitLog,
    entry: Dict[str, Any],
    base: Optional[Dict[str, Any]],
    *,
    now: Optional[float] = None,
) -> Tuple[str, str]:
    """Fetch-first push of ``entry`` against ``log``.

    ``base`` is the committed state the writer saw when it built the entry
    (None for a brand-new ref). Returns ``(status, ref)`` where status is
    ``"applied"`` (fast path: base still matches committed) or ``"conflict"``
    (keep-both: existing entry untouched, incoming stored under a conflict
    ref). The returned ref is where the incoming content landed.
    """
    validate_provenance(entry)
    ref = entry.get("ref")
    if not ref:
        raise ValueError("entry has no ref")
    with _WRITE_LOCKS.hold(ref):
        committed = log.get(ref)
        if committed is None or committed == base:
            with log._lock:
                log._apply_locked(entry)
            return ("applied", ref)
        incoming = copy.deepcopy(entry)
        with log._lock:
            target = conflict_ref(ref, now)
            suffix = 2
            while target in log._entries:  # same-second collision: stay unique
                target = "%s-%d" % (conflict_ref(ref, now), suffix)
                suffix += 1
            incoming["ref"] = target
            incoming["conflict_with"] = ref
            log._apply_locked(incoming)
        return ("conflict", incoming["ref"])


def committed_snapshot(log: CommitLog) -> List[Dict[str, Any]]:
    """Deep copy of all committed entries — the only state Dream may read."""
    with log._lock:
        return copy.deepcopy(list(log._entries.values()))
