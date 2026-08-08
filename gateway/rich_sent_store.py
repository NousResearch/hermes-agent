"""Local bounded index of messages Hermes has sent through Telegram.

Telegram does NOT echo a rich message's content back in ``reply_to_message``
when a user replies to it (verified: ``.text``/``.caption`` empty,
``.api_kwargs`` None). So replies to the launchd briefings / any rich send
arrive with no quotable text and the agent is blind to what was referenced.

The adapter records every successful send/edit here. The same
``message_id -> text`` lookup supports quoted context, while optional routing
metadata lets platform-native events identify Hermes-authored messages without
adding another store or poller.

Best-effort and dependency-free: every operation swallows errors and degrades
to a no-op / ``None`` so it can never break a send or an inbound message.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000
_STORE_LOCK = threading.Lock()


def _store_path() -> str:
    # Resolve via get_hermes_home() so the active profile override is honored.
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return os.path.join(str(home), "state", "rich_sent_index.json")


def _key(chat_id, message_id) -> str:
    return f"{chat_id}:{message_id}"


def _store_paths() -> list[str]:
    """Return the current, base, and named-profile index paths."""
    paths = [_store_path()]
    try:
        from hermes_constants import get_default_hermes_root

        base_home = get_default_hermes_root()
        paths.append(os.path.join(str(base_home), "state", "rich_sent_index.json"))
        profiles_dir = os.path.join(str(base_home), "profiles")
        # Reaction lookups are rare, but still keep one event bounded even on
        # a damaged installation with an unexpectedly large profiles folder.
        for name in sorted(os.listdir(profiles_dir))[:128]:
            profile_home = os.path.join(profiles_dir, name)
            if os.path.isdir(profile_home):
                paths.append(
                    os.path.join(profile_home, "state", "rich_sent_index.json")
                )
    except (OSError, TypeError, ValueError):
        pass

    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        try:
            normalized = os.path.normcase(os.path.abspath(os.fspath(path)))
        except (TypeError, ValueError):
            continue
        if normalized not in seen:
            seen.add(normalized)
            unique.append(path)
    return unique


def _load_entry(path: str, key: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data.get(key) if isinstance(data, dict) else None
        return dict(entry) if isinstance(entry, dict) else None
    except (FileNotFoundError, OSError, ValueError, AttributeError, TypeError):
        return None


def _entry_timestamp(entry: dict) -> float:
    try:
        return float(entry.get("ts", 0))
    except (TypeError, ValueError):
        return 0.0


def record(
    chat_id,
    message_id,
    text: Optional[str],
    *,
    thread_id: Optional[str] = None,
    sender_id: Optional[str] = None,
) -> None:
    """Persist a sent-message entry for ``(chat_id, message_id)``.

    The optional routing fields are additive metadata for adapters that need
    to identify their own messages and route a later platform event back to
    the original conversation. Older entries contain only ``t``/``ts`` and
    remain valid. Empty text is retained so media/control messages are still
    known as Hermes-authored targets, while :func:`lookup` continues to return
    ``None`` when no quoted text is available.
    """
    if message_id is None or chat_id is None:
        return
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _STORE_LOCK:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    data = {}
            except (FileNotFoundError, OSError, ValueError, TypeError):
                data = {}

            existing = data.get(_key(chat_id, message_id))
            entry = dict(existing) if isinstance(existing, dict) else {}
            entry.update({
                "t": (text or "")[:_MAX_TEXT_CHARS],
                "ts": time.time(),
            })
            # Routing metadata is additive. In particular, an edit that has
            # no new thread value must not erase the thread on the original
            # send. Old entries with only t/ts remain valid.
            for key, value in (
                ("thread_id", thread_id),
                ("sender_id", sender_id),
            ):
                if value is not None and str(value) != "":
                    entry[key] = str(value)
            data[_key(chat_id, message_id)] = entry
            # Trim oldest by timestamp when over cap.
            if len(data) > _MAX_ENTRIES:
                ordered = sorted(
                    data.items(),
                    key=lambda kv: _entry_timestamp(kv[1])
                    if isinstance(kv[1], dict)
                    else 0.0,
                )
                for k, _ in ordered[: len(data) - _MAX_ENTRIES]:
                    data.pop(k, None)
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, path)  # atomic within this process-level lock
    except Exception:
        return


def lookup(chat_id, message_id) -> Optional[str]:
    """Return stored text for ``(chat_id, message_id)`` or ``None``."""
    entry = lookup_entry(chat_id, message_id)
    if not entry:
        return None
    return entry.get("t") or None


def lookup_entry(chat_id, message_id, *, all_profiles: bool = False) -> Optional[dict]:
    """Return the stored entry for ``(chat_id, message_id)``.

    This is intentionally a shallow, backward-compatible read helper: old
    entries without reaction-routing metadata are still returned as valid
    Hermes-authored entries. ``all_profiles`` is reserved for transports that
    need profile-wide ownership recovery; conflicting metadata then fails
    closed. Normal reply-text lookup remains scoped to the active profile.
    """
    if message_id is None or chat_id is None:
        return None
    key = _key(chat_id, message_id)
    candidates: list[dict] = []
    paths = _store_paths() if all_profiles else [_store_path()]
    for path in paths:
        entry = _load_entry(path, key)
        if entry is not None:
            candidates.append(entry)
    if not candidates:
        return None

    # A chat/message id is globally unique in Telegram. Conflicting routing
    # metadata across profile indexes therefore indicates stale or corrupted
    # state; never guess a topic or sending bot in that case.
    thread_ids = {
        str(entry.get("thread_id"))
        for entry in candidates
        if entry.get("thread_id") not in {None, ""}
    }
    sender_ids = {
        str(entry.get("sender_id"))
        for entry in candidates
        if entry.get("sender_id") not in {None, ""}
    }
    if len(thread_ids) > 1 or len(sender_ids) > 1:
        return None

    newest = dict(max(candidates, key=_entry_timestamp))
    if not newest.get("thread_id") and thread_ids:
        newest["thread_id"] = next(iter(thread_ids))
    if not newest.get("sender_id") and sender_ids:
        newest["sender_id"] = next(iter(sender_ids))
    return newest
