"""Transcript rescue for a store replaced under a live process: pending messages go to a JSONL file instead."""

import json
from pathlib import Path
from typing import Optional

from hermes_dir_acl import make_user_dir


def divert_session_transcript_jsonl(session_id: str, messages, *, sessions_dir: Path) -> Optional[Path]:
    """Append pending messages to ``<sessions_dir>/<id>.jsonl`` (state.db was replaced under a
    live process). Returns the path, or None if nothing to write."""
    sid = str(session_id or "").strip()
    if not sid or not messages:
        return None
    make_user_dir(sessions_dir, parents=True, exist_ok=True)
    path = sessions_dir / f"{sid}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for msg in messages:
            if msg is not None:
                record = msg if isinstance(msg, dict) else {"content": str(msg)}
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path
