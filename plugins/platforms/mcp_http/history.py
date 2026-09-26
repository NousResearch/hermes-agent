"""Per-conversation transcript persistence for the MCP HTTP platform.

Remote coding agents restart often and lose their own context; ``history()`` lets them
re-read the recent exchange. Stored outside the session store as plain JSONL under the
profile cache so it survives gateway restarts and context compaction alike.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

HISTORY_KEEP = 20  # exchanges (peer message + reply) kept per conversation


def safe_conversation_id(value: str, max_len: int = 96) -> str:
    """Conversation ids become file names and session keys: slugify to a safe charset."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (slug or "default")[:max_len]


def _path(conversation_id: str) -> Path:
    return get_hermes_home() / "cache" / "mcp_http" / "history" / f"{safe_conversation_id(conversation_id)}.jsonl"


def load(conversation_id: str) -> list[dict]:
    path = _path(conversation_id)
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        logger.debug("MCP HTTP: history read failed for %s", conversation_id, exc_info=True)
        return []


def append(conversation_id: str, role: str, text: str) -> None:
    """Append one message and trim to the last ``HISTORY_KEEP`` exchanges. Never raises:
    a persistence hiccup must not turn into a failed reply for the caller."""
    rows = load(conversation_id)
    rows.append({"ts": time.time(), "role": role, "text": text})
    rows = rows[-(HISTORY_KEEP * 2):]
    try:
        path = _path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    except OSError:
        logger.debug("MCP HTTP: history write failed for %s", conversation_id, exc_info=True)


def render(conversation_id: str, limit: int) -> str:
    limit = max(1, min(int(limit or 10), HISTORY_KEEP))
    rows = load(conversation_id)[-(limit * 2):]
    if not rows:
        return f"no history for conversation_id={conversation_id}."
    out = [f"history conversation_id={conversation_id} (last {len(rows)} messages, oldest first)"]
    for r in rows:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("ts", 0)))
        text = (r.get("text") or "").strip()
        if len(text) > 1500:
            text = text[:1500] + f" …[{len(text) - 1500} more chars]"
        out.append(f"[{ts}] {r.get('role')}: {text}")
    return "\n\n".join(out)
