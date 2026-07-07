"""Reader for the per-topic raw Bot-API "recent posts" log.

Raw Bot API posters (cron digests, watchdog alerts, the review bridge,
``tg-send``) post into forum topics via ``sendMessage``. Those posts create NO
Hermes session and NO ``state.db`` row, so ``gateway.topic_backfill``'s
SessionDB-sibling scan is structurally blind to them. Each poster appends to a
per-topic rolling JSON log (write side: ``~/.hermes/scripts/tg_topic_recent_posts.py``).
This module READS that log so ``build_topic_backfill`` can merge it as a SECOND
source beside sibling sessions.

SHARED ON-DISK FORMAT CONTRACT (must match the script writer byte-for-byte):
    path:  $HERMES_HOME/state/topic-recent-posts/<platform>/<chat_id>/<thread_id>.json
    shape: {"posts": [{"role", "text", "timestamp", "label", "source", "context_text"?}, ...]}

Read-only and defensive: returns ``[]`` on any error; never raises. Paths are
resolved at CALL time (not import) so a relocated ``HERMES_HOME`` in tests is
honored. Mirrors the import-time path-constant discipline in
``gateway.topic_backfill``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _hermes_home():
    """Resolve the Hermes home at CALL time (not import) so a relocated
    ``HERMES_HOME`` in tests is honored."""
    from hermes_cli.config import get_hermes_home

    return get_hermes_home()


def _log_path(platform: str, chat_id: str, thread_id: Optional[str]):
    thread = "" if thread_id is None else str(thread_id)
    return (
        _hermes_home()
        / "state"
        / "topic-recent-posts"
        / str(platform).lower()
        / str(chat_id)
        / f"{thread}.json"
    )


def read_recent_bot_posts(
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Return raw post entries for one topic. ``[]`` on missing file / error.

    Each returned dict is normalized to the shape ``topic_backfill`` expects:
    ``{"label", "role", "text", "timestamp"}``. Group-root posts (no thread) are
    never logged by the writer, so a ``None``/empty ``thread_id`` finds no file
    and returns ``[]``.
    """
    if thread_id is None or str(thread_id).strip() == "":
        return []
    try:
        p = _log_path(platform, chat_id, thread_id)
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("topic_recent_posts: read failed: %s", e)
        return []

    posts = data.get("posts") if isinstance(data, dict) else None
    if not isinstance(posts, list):
        return []

    out: List[Dict[str, Any]] = []
    for row in posts:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        item = {
            "label": row.get("label") or "bot",
            "role": row.get("role") or "assistant",
            "text": text,
            "timestamp": row.get("timestamp"),
        }
        context_text = row.get("context_text")
        if isinstance(context_text, str) and context_text.strip():
            item["context_text"] = context_text.strip()
        out.append(item)
    return out
