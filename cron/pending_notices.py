"""Pending cron delivery notices (push side of cron session-awareness).

Cron deliveries do NOT enter the interactive conversation history. The
assistant-role mirror that used to do that was removed in #2313 because two
assistant turns in a row violate message alternation (#2221).

Instead, each successful delivery is recorded here, keyed by destination, and
the gateway message handler folds any pending notices into the SYSTEM PROMPT of
that chat's next interactive turn (alternation-safe, mirroring the auto-reset
context-note precedent) and then drains the buffer. The net effect: the agent
becomes aware of what its crons sent, without polluting the message array.

Standalone and best-effort: callable from the in-process scheduler tick, and
every failure is swallowed so it can never break delivery (which has already
happened by the time we record) or a user turn.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Scheduler ticks and gateway message handling run in the same process; a
# module-level lock plus atomic replace is enough to keep the store consistent.
_LOCK = threading.Lock()

# Cap entries per destination so a chat that goes unread for a long time can't
# grow the store (or the next turn's system prompt) without bound.
_MAX_PER_KEY = 20


def _store_path(base_dir: Optional[Path] = None) -> Path:
    if base_dir is not None:
        base = Path(base_dir)
    else:
        from hermes_constants import get_hermes_home
        base = get_hermes_home() / "cron"
    return base / "pending_notices.json"


def _key(platform: str, chat_id) -> str:
    return f"{str(platform).lower()}:{chat_id}"


def new_notice_id() -> str:
    """Short unique id for a buffered notice.

    Kept short so it fits inside a Telegram inline-button ``callback_data``
    (64-byte cap) as ``cron:accept:<id>``.
    """
    return uuid.uuid4().hex[:8]


def normalize_notify_mode(value) -> str:
    """Normalize a ``cron.notify_session`` config value to off / auto / button.

    Back-compatible with the original boolean knob: True (or any recognized
    on-ish value) means auto, False / None / off-ish means off. "button" opts
    into inline accept/dismiss buttons. An unrecognized but present value stays
    on (auto), matching the old "any truthy config value enabled it" behavior.
    """
    if value is True:
        return "auto"
    if value is False or value is None:
        return "off"
    s = str(value).strip().lower()
    if s in {"button", "buttons"}:
        return "button"
    if s in {"off", "no", "false", "0", "disabled", "none", ""}:
        return "off"
    return "auto"


def _load(path: Path) -> Dict[str, List[dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.debug("pending notices: unreadable store %s (%s); starting empty", path, e)
        return {}


def _save(path: Path, data: Dict[str, List[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def record(
    platform: str,
    chat_id,
    job_name: str,
    text: str,
    thread_id: Optional[str] = None,
    notice_id: Optional[str] = None,
    inject: bool = True,
    base_dir: Optional[Path] = None,
) -> Union[str, bool]:
    """Buffer one delivered cron message for ``platform``/``chat_id``.

    ``inject`` controls how the next interactive turn treats the entry: True
    (auto mode) folds it into the system prompt right away; False (button mode)
    holds it until the user accepts it via an inline button, at which point
    :func:`mark_accepted` flips it. ``notice_id`` lets the caller pre-mint the id
    it put on the button so the two stay in sync; one is generated otherwise.

    Returns the notice id (str) on success, or False on empty input / error.
    """
    text = (text or "").strip()
    if not text or chat_id in (None, ""):
        return False
    nid = notice_id or new_notice_id()
    try:
        with _LOCK:
            path = _store_path(base_dir)
            data = _load(path)
            key = _key(platform, chat_id)
            entries = data.get(key, [])
            entries.append({
                "id": nid,
                "ts": datetime.now().isoformat(timespec="seconds"),
                "job_name": job_name or "",
                "thread_id": thread_id,
                "text": text,
                "inject": bool(inject),
            })
            data[key] = entries[-_MAX_PER_KEY:]
            _save(path, data)
        return nid
    except Exception as e:
        logger.debug("pending notice record failed for %s:%s: %s", platform, chat_id, e)
        return False


def drain(
    platform: str,
    chat_id,
    base_dir: Optional[Path] = None,
) -> List[dict]:
    """Return and clear the injectable notices for ``platform``/``chat_id``.

    Only entries with ``inject`` truthy are returned and removed; entries still
    awaiting an accept button (``inject`` False) are left in place for a later
    :func:`mark_accepted`. Entries written before the inject flag existed default
    to injectable, preserving the original drain-everything behavior.

    Returns an empty list when nothing is injectable or on any error.
    """
    try:
        with _LOCK:
            path = _store_path(base_dir)
            data = _load(path)
            key = _key(platform, chat_id)
            entries = data.get(key, [])
            if not entries:
                return []
            injectable = [e for e in entries if e.get("inject", True)]
            remaining = [e for e in entries if not e.get("inject", True)]
            if injectable:
                if remaining:
                    data[key] = remaining
                else:
                    data.pop(key, None)
                _save(path, data)
            return injectable
    except Exception as e:
        logger.debug("pending notice drain failed for %s:%s: %s", platform, chat_id, e)
        return []


def mark_accepted(
    platform: str,
    chat_id,
    notice_id: str,
    base_dir: Optional[Path] = None,
) -> bool:
    """Flip a held (button-mode) notice to injectable.

    Called when the user taps the accept button. Returns True if a matching
    entry was found, False otherwise (already drained, dismissed, or unknown id).
    """
    try:
        with _LOCK:
            path = _store_path(base_dir)
            data = _load(path)
            key = _key(platform, chat_id)
            for e in data.get(key, []):
                if e.get("id") == notice_id:
                    e["inject"] = True
                    _save(path, data)
                    return True
        return False
    except Exception as e:
        logger.debug("pending notice mark_accepted failed for %s:%s: %s", platform, chat_id, e)
        return False


def dismiss(
    platform: str,
    chat_id,
    notice_id: str,
    base_dir: Optional[Path] = None,
) -> bool:
    """Drop a held notice the user declined. Returns True if one was removed."""
    try:
        with _LOCK:
            path = _store_path(base_dir)
            data = _load(path)
            key = _key(platform, chat_id)
            entries = data.get(key, [])
            kept = [e for e in entries if e.get("id") != notice_id]
            if len(kept) == len(entries):
                return False
            if kept:
                data[key] = kept
            else:
                data.pop(key, None)
            _save(path, data)
            return True
    except Exception as e:
        logger.debug("pending notice dismiss failed for %s:%s: %s", platform, chat_id, e)
        return False
