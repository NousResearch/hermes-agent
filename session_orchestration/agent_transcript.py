"""
Agent transcript reader — relay clean assistant text, not pane scrapings.

omp records every session as JSONL under
``~/.omp/agent/sessions/<cwd-slug>/<iso-ts>_<uuid>.jsonl``:

    {"type": "session", "id": ..., "timestamp": ..., "cwd": "/abs/workdir", ...}
    {"type": "message", "message": {"role": "user"|"assistant"|"toolResult",
                                    "content": [{"type": "text"|"thinking"|"toolCall", ...}]}}

That file is the authoritative, ANSI-free record of what the agent actually
said.  Relaying assistant ``text`` blocks from it (skipping ``thinking`` and
``toolCall``) gives the Discord thread exactly the agent's response — no build
logs, no tool spam, and never the user's own pasted reply (``role=user``).

Two pure helpers:

``discover_omp_session_file(workdir, launch_ts, claimed=())``
    Map a spawned session to its transcript file by cwd-slug + launch-time
    proximity.  ``claimed`` excludes files already owned by other registry rows
    (parallel spawns in the same repo).

``read_assistant_texts(path, after_line)``
    Return assistant text blocks appended after *after_line* plus the new
    total line count (the caller persists it as ``transcript_line_offset``).

Both are read-only and never raise on malformed input (skip-and-continue,
matching the marker-protocol convention).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Root of omp's per-cwd session transcript tree.
_OMP_SESSIONS_ROOT = Path.home() / ".omp" / "agent" / "sessions"

#: A transcript file must have been created within this window around the
#: tmux launch to be considered the spawned session's file.
_DISCOVERY_WINDOW_BEFORE = timedelta(seconds=30)
_DISCOVERY_WINDOW_AFTER = timedelta(seconds=180)


def _cwd_slug(workdir: str) -> str:
    """omp names the per-cwd directory by home-relativising then '/'→'-'.

    ``/Users/zeke/dev/z-harness`` → ``-dev-z-harness`` (home stripped);
    ``/private/tmp/x``            → ``-private-tmp-x`` (non-home kept absolute).
    """
    path = workdir.rstrip("/")
    home = str(Path.home()).rstrip("/")
    if path == home:
        path = ""
    elif path.startswith(home + "/"):
        path = path[len(home):]
    return path.replace("/", "-")


def _parse_filename_ts(name: str) -> Optional[datetime]:
    """Parse the leading ISO timestamp out of ``2026-07-02T04-40-31-230Z_<id>.jsonl``."""
    stem = name.split("_", 1)[0]
    # 2026-07-02T04-40-31-230Z  ->  2026-07-02T04:40:31.230+00:00
    try:
        date_part, time_part = stem.split("T", 1)
        time_part = time_part.rstrip("Z")
        h, m, s, ms = time_part.split("-")
        return datetime.fromisoformat(f"{date_part}T{h}:{m}:{s}.{ms.ljust(3, '0')}").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, IndexError):
        return None


def discover_omp_session_file(
    workdir: str,
    launch_ts: datetime,
    claimed: Iterable[str] = (),
    *,
    sessions_root: Optional[Path] = None,
) -> Optional[str]:
    """Find the omp transcript file created by a session launched at *launch_ts*.

    Scans ``<root>/<cwd-slug>/*.jsonl`` for files whose filename timestamp
    falls inside the discovery window around *launch_ts*, excludes paths in
    *claimed*, and returns the one closest to the launch time (or ``None``).
    """
    root = (sessions_root or _OMP_SESSIONS_ROOT) / _cwd_slug(workdir)
    if not root.is_dir():
        return None
    if launch_ts.tzinfo is None:
        launch_ts = launch_ts.replace(tzinfo=timezone.utc)

    claimed_set = {str(p) for p in claimed}
    best: Optional[Tuple[float, str]] = None
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        logger.debug("agent_transcript: cannot list %s: %s", root, exc)
        return None

    for entry in entries:
        if not entry.name.endswith(".jsonl") or not entry.is_file():
            continue
        if str(entry) in claimed_set:
            continue
        ts = _parse_filename_ts(entry.name)
        if ts is None:
            continue
        if not (launch_ts - _DISCOVERY_WINDOW_BEFORE <= ts <= launch_ts + _DISCOVERY_WINDOW_AFTER):
            continue
        delta = abs((ts - launch_ts).total_seconds())
        if best is None or delta < best[0]:
            best = (delta, str(entry))

    return best[1] if best else None


def read_assistant_texts(path: str, after_line: int = 0) -> Tuple[List[str], int]:
    """Return assistant text blocks appended after *after_line*.

    Parameters
    ----------
    path:
        Transcript JSONL path (from ``discover_omp_session_file``).
    after_line:
        Number of lines already consumed (``transcript_line_offset``).

    Returns
    -------
    (texts, total_lines):
        ``texts`` — one string per assistant message that contained at least
        one ``text`` block (blocks within a message joined by newlines);
        ``thinking``/``toolCall`` blocks and non-assistant roles are skipped.
        ``total_lines`` — new line count to persist as the offset.

    Never raises on malformed lines; on an unreadable file returns
    ``([], after_line)`` so the caller retries next tick.
    """
    texts: List[str] = []
    total = after_line
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                total = idx + 1
                if idx < after_line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if obj.get("type") != "message":
                    continue
                message = obj.get("message")
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                blocks = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
                ]
                if blocks:
                    texts.append("\n".join(blocks).strip())
    except OSError as exc:
        logger.debug("agent_transcript: cannot read %s: %s", path, exc)
        return [], after_line

    return texts, total
