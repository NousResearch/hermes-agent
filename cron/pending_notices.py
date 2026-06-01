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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    base_dir: Optional[Path] = None,
) -> bool:
    """Buffer one delivered cron message for ``platform``/``chat_id``.

    Returns True if stored, False on empty input or any error.
    """
    text = (text or "").strip()
    if not text or chat_id in (None, ""):
        return False
    try:
        with _LOCK:
            path = _store_path(base_dir)
            data = _load(path)
            key = _key(platform, chat_id)
            entries = data.get(key, [])
            entries.append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "job_name": job_name or "",
                "thread_id": thread_id,
                "text": text,
            })
            data[key] = entries[-_MAX_PER_KEY:]
            _save(path, data)
        return True
    except Exception as e:
        logger.debug("pending notice record failed for %s:%s: %s", platform, chat_id, e)
        return False


def drain(
    platform: str,
    chat_id,
    base_dir: Optional[Path] = None,
) -> List[dict]:
    """Return and clear all pending notices for ``platform``/``chat_id``.

    Returns an empty list when there is nothing pending or on any error.
    """
    try:
        with _LOCK:
            path = _store_path(base_dir)
            data = _load(path)
            key = _key(platform, chat_id)
            entries = data.pop(key, [])
            if entries:
                _save(path, data)
        return entries
    except Exception as e:
        logger.debug("pending notice drain failed for %s:%s: %s", platform, chat_id, e)
        return []
