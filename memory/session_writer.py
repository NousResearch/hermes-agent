"""Immutable append-only JSONL session writer.

Each turn is appended as a single JSON line to a per-session file.
This is the source of truth — SQLite index.db is a searchable overlay.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from memory.config import SESSIONS_DIR

logger = logging.getLogger(__name__)


def append_turn_jsonl(
    session_id: str,
    role: str,
    content: str,
    tool_calls: Optional[list] = None,
    tool_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Path:
    """Append one turn to the session JSONL file.

    Args:
        session_id: Unique session identifier.
        role: 'user', 'assistant', or 'tool'.
        content: Message content.
        tool_calls: List of tool call dicts (for assistant messages).
        tool_name: Tool name (for tool result messages).
        timestamp: Unix timestamp. Defaults to now.

    Returns:
        Path to the JSONL file written to.
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{session_id}.jsonl"

    record = {
        "role": role,
        "content": content,
        "timestamp": timestamp or time.time(),
    }
    if tool_calls:
        record["tool_calls"] = tool_calls
    if tool_name:
        record["tool_name"] = tool_name

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path
