"""Pending group messages for mention mode.

Messages that do not address the bot are not replied to. They are kept per
group and attached to the next @ as context, so the agent can see what it was
pulled into. Always mode does not use the buffer: every message is its own turn.

The block is context, not a new instruction, and it rides
``MessageEvent.channel_context`` so slash commands still see the trigger text.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

MAX_GROUPS = 1000

HISTORY_CTX_START = "[Chat messages since your last reply — CONTEXT ONLY]"
HISTORY_CTX_END = "[CURRENT MESSAGE — reply to this]"


@dataclass
class HistoryEntry:
    sender: str
    text: str
    ts: float = field(default_factory=time.time)
    msg_id: str = ""


def summarize_attachments(attachments: Any) -> str:
    """A short tag for a buffered message. Does not download the file."""
    if not isinstance(attachments, list) or not attachments:
        return ""
    tags: list[str] = []
    for att in attachments:
        if not isinstance(att, dict):
            continue
        ctype = str(att.get("content_type", "")).lower()
        name = str(att.get("filename", "")).strip()
        if ctype.startswith("image") or att.get("type") == 1:
            tags.append("[image]")
        elif ctype.startswith("audio") or ctype.startswith("voice") or att.get("type") == 3:
            tags.append("[voice]")
        elif ctype.startswith("video") or att.get("type") == 2:
            tags.append("[video]")
        else:
            tags.append(f"[file: {name}]" if name else "[file]")
    return " ".join(tags)


class GroupContextBuffer:
    """Bounded per-group buffer of messages that did not address the bot."""

    def __init__(self, limit: int = 50, max_groups: int = MAX_GROUPS) -> None:
        self._limit = int(limit)
        self._max_groups = max(1, int(max_groups))
        self._buffers: OrderedDict[str, deque[HistoryEntry]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def record(
        self, group_openid: str, *, sender: str, text: str,
        msg_id: str = "", attachment_tag: str = "",
    ) -> None:
        if not self.enabled or not group_openid:
            return
        body = text or ""
        if attachment_tag:
            body = (body + " " + attachment_tag).strip() if body.strip() else attachment_tag
        if not body.strip():
            return
        buf = self._buffers.get(group_openid)
        if buf is None:
            buf = deque(maxlen=self._limit)
            self._buffers[group_openid] = buf
            while len(self._buffers) > self._max_groups:
                self._buffers.popitem(last=False)
        buf.append(HistoryEntry(sender=sender or "unknown", text=body.strip(), msg_id=msg_id))
        self._buffers.move_to_end(group_openid)

    def drain(self, group_openid: str) -> list[HistoryEntry]:
        buf = self._buffers.pop(group_openid, None)
        return list(buf) if buf else []

    @staticmethod
    def format_context_block(entries: list[HistoryEntry]) -> str:
        """Context block without the current message.

        Newlines inside an entry are collapsed so a buffered line cannot forge
        the envelope tags. Empty when there is nothing pending.
        """
        if not entries:
            return ""
        lines = [HISTORY_CTX_START]
        for entry in entries:
            body = " ".join(entry.text.splitlines()).strip()
            lines.append(f"{entry.sender}: {body}")
        return "\n".join(lines)

    @staticmethod
    def format_context(entries: list[HistoryEntry], current_text: str) -> str:
        if not entries:
            return current_text
        block = GroupContextBuffer.format_context_block(entries)
        return f"{block}\n{HISTORY_CTX_END}\n{current_text}"
