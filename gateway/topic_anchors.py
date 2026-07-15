"""Reply-anchor tracking for background sends into Telegram private DM topics.

Telegram private-chat forum topics (Kanstantsin's per-project DM sub-topics)
only accept a NEW bot message when it replies to a message already inside
that topic — there is no equivalent of a group's ``message_thread_id`` for
DMs. A live agent turn always has a fresh inbound message to reply to, but
background pushes (the Kanban notifier, cron auto-delivery, etc.) have no
such message — they only know ``(platform, chat_id, thread_id)``.

This module records the most recent inbound message id per
``(platform, chat_id, thread_id)`` to a small shared JSON file, so those
background pushes can look one up and use it as ``telegram_reply_to_message_id``
(see ``gateway.run.GatewayRunner._thread_metadata_for_target`` and
``gateway.delivery``, which both require that key for legacy numeric DM
topics). Without an anchor, the Telegram adapter hard-refuses the send
("Telegram DM topic delivery requires a reply anchor") rather than silently
landing the message in the wrong place. It also stores observed human topic
names for later Kanban overview rendering.

The store is process-shared (``kanban_home()``-rooted, not per-profile) —
the profile that records an inbound message (e.g. the auditor gateway
handling a live DM turn) is not necessarily the profile whose notifier
later needs the anchor.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ANCHORS_FILENAME = "telegram_topic_anchors.json"
_ANCHOR_LOCK_SUFFIX = ".lock"
_ANCHOR_LOCK_TIMEOUT_SECONDS = 2.0

# ``flock`` is process-scoped on POSIX, so it does not serialize threads in
# one gateway process. Keep this lock alongside the cross-process file lock.
_anchor_write_lock = threading.Lock()

# Cap file growth — this is a rolling "last seen" cache, not a history log.
_MAX_ENTRIES = 2000
_MAX_TOPIC_LABEL_LENGTH = 128


def _anchors_path() -> Path:
    try:
        from hermes_cli.kanban_db import kanban_home
        root = kanban_home()
    except Exception:
        root = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return root / _ANCHORS_FILENAME


def _key(platform: str, chat_id: str, thread_id: str) -> str:
    return f"{platform}:{chat_id}:{thread_id}"


def _clean_topic_label(value: object) -> str:
    """Return one bounded line suitable for a human-facing topic label."""
    return " ".join(str(value or "").split())[:_MAX_TOPIC_LABEL_LENGTH]


def _acquire_anchor_file_lock(path: Path):
    """Return a held cross-process lock for the anchor cache, if available.

    Anchor updates are read-modify-replace operations. Atomic ``os.replace``
    prevents torn JSON but not a lost update when two gateway profiles read
    the same old cache and publish different replacements. Reuse the
    gateway's cross-platform advisory-lock primitive and wait briefly rather
    than silently overwriting another profile's freshly recorded topic metadata.
    """
    try:
        from gateway.status import _release_file_lock, _try_acquire_file_lock
    except ImportError:
        return None

    lock_path = path.with_name(path.name + _ANCHOR_LOCK_SUFFIX)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")
    except OSError:
        return None

    deadline = time.monotonic() + _ANCHOR_LOCK_TIMEOUT_SECONDS
    while not _try_acquire_file_lock(handle):
        if time.monotonic() >= deadline:
            handle.close()
            return None
        time.sleep(0.01)
    return handle, _release_file_lock


def _load(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("topic_anchors: failed to read %s", path, exc_info=True)
        return {}


def _record_topic_metadata(
    platform: str, chat_id: str, thread_id: str, **metadata: str,
) -> None:
    """Merge best-effort durable metadata for one Telegram topic lane."""
    platform = str(platform or "").strip().lower()
    chat_id = str(chat_id or "").strip()
    thread_id = str(thread_id or "").strip()
    metadata = {key: str(value or "").strip() for key, value in metadata.items()}
    if not platform or not chat_id or not thread_id or not metadata or not all(metadata.values()):
        return

    path = _anchors_path()
    with _anchor_write_lock:
        file_lock = _acquire_anchor_file_lock(path)
        if file_lock is None:
            logger.debug("topic_anchors: could not lock %s; skipping topic metadata update", path)
            return
        handle, release_file_lock = file_lock
        try:
            data = _load(path)
            key = _key(platform, chat_id, thread_id)
            entry = data.get(key)
            if not isinstance(entry, dict):
                entry = {}
            entry.update(metadata)
            data[key] = entry
            if len(data) > _MAX_ENTRIES:
                # Drop oldest-inserted entries first (dict preserves insertion
                # order); this is an approximate LRU, good enough for a cache.
                for stale_key in list(data.keys())[: len(data) - _MAX_ENTRIES]:
                    data.pop(stale_key, None)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), prefix=".topic_anchors_", suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            logger.debug("topic_anchors: failed to record topic metadata", exc_info=True)
        finally:
            release_file_lock(handle)
            try:
                handle.close()
            except OSError:
                pass


def record_topic_anchor(
    platform: str, chat_id: str, thread_id: str, message_id: str,
) -> None:
    """Remember ``message_id`` as the latest reply anchor for this lane."""
    _record_topic_metadata(
        platform, chat_id, thread_id, message_id=str(message_id or "").strip(),
    )


def record_topic_name(platform: str, chat_id: str, thread_id: str, name: str) -> None:
    """Remember a human topic name observed on an inbound Telegram event."""
    clean_name = _clean_topic_label(name)
    if clean_name:
        _record_topic_metadata(platform, chat_id, thread_id, name=clean_name)


def get_topic_anchor(platform: str, chat_id: str, thread_id: str) -> Optional[str]:
    """Return the last recorded inbound message id for this lane, if any."""
    platform = str(platform or "").strip().lower()
    chat_id = str(chat_id or "").strip()
    thread_id = str(thread_id or "").strip()
    if not platform or not chat_id or not thread_id:
        return None
    entry = _load(_anchors_path()).get(_key(platform, chat_id, thread_id))
    if not isinstance(entry, dict):
        return None
    message_id = entry.get("message_id")
    return str(message_id) if message_id else None


def get_topic_name(platform: str, chat_id: str, thread_id: str) -> Optional[str]:
    """Return a durable human name for a topic lane, if one was observed."""
    platform = str(platform or "").strip().lower()
    chat_id = str(chat_id or "").strip()
    thread_id = str(thread_id or "").strip()
    if not platform or not chat_id or not thread_id:
        return None
    entry = _load(_anchors_path()).get(_key(platform, chat_id, thread_id))
    if not isinstance(entry, dict):
        return None
    name = _clean_topic_label(entry.get("name"))
    return name or None


def resolve_topic_name(platform: str, chat_id: str, thread_id: str) -> str:
    """Resolve a durable topic name with a bounded fallback for unknown lanes."""
    name = get_topic_name(platform, chat_id, thread_id)
    if name:
        return name
    fallback_id = _clean_topic_label(thread_id)
    return f"Топик {fallback_id or 'неизвестен'}"
