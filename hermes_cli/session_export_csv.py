"""Message-per-row CSV transcript rendering (JSONL remains the raw backup format)."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable


CSV_FIELDS = ("session_id", "index", "created_at", "role", "content",
              "tool_calls", "tool_call_id", "tool_name")


def _spreadsheet_text(value: Any) -> str:
    """Neutralize formula/control prefixes; CSV quoting alone does not do this."""
    text = "" if value is None else str(value)
    start = 0
    while start < len(text) and (text[start].isspace() or ord(text[start]) < 32 or text[start] == "\x7f"):
        start += 1
    if text[start:start + 1] in ("=", "+", "-", "@") or any(ord(c) < 32 or c == "\x7f" for c in text[:start]):
        return "'" + text
    return text


def render_csv_export(sessions: Iterable[dict[str, Any]], *, prompts_only: bool = False) -> str:
    from hermes_cli.session_export import (
        _format_timestamp, _message_text, _messages, _session_id, iter_user_prompt_records,
    )

    output = io.StringIO(newline="")
    # LF avoids doubled CR on Windows stdout; quote all also protects embedded CR.
    writer = csv.writer(output, lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writerow(CSV_FIELDS)
    if prompts_only:
        for prompt in iter_user_prompt_records(sessions):
            writer.writerow(_spreadsheet_text(value) for value in (
                prompt["session_id"], prompt["index"], prompt["created_at"], "user", prompt["text"], "", "", "",
            ))
        return output.getvalue()
    for session in sessions:
        session_id = _session_id(session)
        messages = [m for m in _messages(session)
                    if m.get("role") != "system"]
        for index, message in enumerate(messages, start=1):
            calls = message.get("tool_calls")
            writer.writerow(_spreadsheet_text(value) for value in (
                session_id, index, _format_timestamp(message.get("timestamp"), session_id),
                message.get("role") or "", _message_text(message.get("content")),
                json.dumps(calls, ensure_ascii=False) if calls is not None else "",
                message.get("tool_call_id") or "", message.get("tool_name") or message.get("name") or "",
            ))
    return output.getvalue()
