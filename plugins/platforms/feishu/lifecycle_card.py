"""Per-turn lifecycle status card: one Feishu interactive card, patched in place.

Interim gateway sends (heartbeats, tool progress, busy/redirect acks, mid-turn
commentary) become entries on a single card instead of separate messages.
Entries are addressed as ``<card_message_id>#e<idx>`` so gateway edit loops
(heartbeat, tool-progress accumulate) keep editing their own entry in place.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

ENTRY_SEP = "#e"

_ENTRY_CAP = 14
_ENTRY_TEXT_CAP = 700
_ENTRY_TAIL_LINES = 8
_BODY_CHAR_BUDGET = 7000
_TITLE_CAP = 80

_STATE_STYLE = {
    "working": ("⏳", "blue"),
    "done": ("✅", "green"),
    "failed": ("❌", "red"),
    "cancelled": ("⏹", "grey"),
}

FINAL_STATES = frozenset({"done", "failed", "cancelled"})


def split_entry_ref(message_id: str) -> Tuple[str, Optional[int]]:
    if message_id and ENTRY_SEP in message_id:
        base, _, idx = message_id.rpartition(ENTRY_SEP)
        if base and idx.isdigit():
            return base, int(idx)
    return message_id, None


def _clip(text: str) -> str:
    lines = [ln for ln in str(text).strip().splitlines() if ln.strip()]
    if len(lines) > _ENTRY_TAIL_LINES:
        lines = ["…"] + lines[-_ENTRY_TAIL_LINES:]
    clipped = "\n".join(lines)
    if len(clipped) > _ENTRY_TEXT_CAP:
        clipped = "…" + clipped[-_ENTRY_TEXT_CAP:]
    return clipped


_MENTION_BLOCK_RE = re.compile(r"\[Mentioned:([^\]]*)\]")
_MENTION_TOKEN_RE = re.compile(r"@\S+")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")


def _table_cells(row: str) -> str:
    return " · ".join(c for c in (c.strip() for c in row.strip().strip("|").split("|")) if c)


def to_lark_md(text: str) -> str:
    # Lark's card markdown renders `inline code`, #-headings and pipe tables
    # literally — remap to bold / bullet rows, leaving fenced blocks untouched.
    out: list[str] = []
    in_fence = False
    in_table = False
    lines = str(text).splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            in_table = False
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if _TABLE_ROW_RE.match(line):
            if in_table:
                if not _TABLE_SEP_RE.match(line):
                    row = _INLINE_CODE_RE.sub(r"**\1**", _table_cells(line))
                    if row:
                        out.append(f"- {row}")
                continue
            if i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
                in_table = True
                header = _INLINE_CODE_RE.sub(r"**\1**", _table_cells(line))
                if header:
                    out.append(f"**{header}**")
                continue
        else:
            in_table = False
        line = _HEADING_RE.sub(r"**\1**", line)
        line = _INLINE_CODE_RE.sub(r"**\1**", line)
        out.append(line)
    return "\n".join(out)


def strip_mentions(text: str) -> str:
    names: list[str] = []

    def _collect_names(match: "re.Match[str]") -> str:
        for part in match.group(1).split(","):
            name = part.split("(open_id=")[0].strip()
            if name:
                names.append(name)
        return " "

    cleaned = _MENTION_BLOCK_RE.sub(_collect_names, str(text or "")).replace("\\", "")
    for name in names:
        cleaned = cleaned.replace(f"@{name}", " ")
    cleaned = _MENTION_TOKEN_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def make_title(text: str) -> str:
    collapsed = strip_mentions(text)
    if len(collapsed) > _TITLE_CAP:
        collapsed = collapsed[: _TITLE_CAP - 1] + "…"
    return collapsed


@dataclass
class _Entry:
    text: str
    updated_ts: float


@dataclass
class LifecycleCard:
    chat_id: str
    key: str = ""
    title: str = ""
    requester: str = ""
    anchor_message_id: str = ""
    thread_anchor_id: str = ""
    message_id: str = ""
    state: str = "working"
    answer: str = ""
    answer_absorbed: bool = False
    entries: Dict[int, _Entry] = field(default_factory=dict)
    started_ts: float = field(default_factory=time.time)
    _next_idx: int = 0

    def matches_anchor(self, reply_to: Optional[str]) -> bool:
        # In Lark threads the gateway anchors the final reply to the thread
        # root, not the triggering mention — accept either.
        return bool(reply_to) and reply_to in {
            a for a in (self.anchor_message_id, self.thread_anchor_id) if a
        }

    def add_entry(self, content: str) -> int:
        idx = self._next_idx
        self._next_idx += 1
        self.entries[idx] = _Entry(_clip(content), time.time())
        if len(self.entries) > _ENTRY_CAP:
            # Evict the stalest entry so live-edited entries (heartbeat,
            # tool progress) survive even though they were created first.
            stalest = min(
                (i for i in self.entries if i != idx),
                key=lambda i: self.entries[i].updated_ts,
            )
            del self.entries[stalest]
        return idx

    def update_entry(self, idx: int, content: str) -> None:
        # Upsert: an evicted entry that the gateway keeps editing (heartbeat,
        # tool progress) is resurrected rather than silently dropped.
        self.entries[idx] = _Entry(_clip(content), time.time())

    def finalize(self, state: str, answer: str = "") -> None:
        self.state = state
        if answer:
            self.answer = answer
            self.answer_absorbed = True

    def _elapsed(self) -> str:
        seconds = max(0, int(time.time() - self.started_ts))
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m {seconds % 60:02d}s"

    def _body(self) -> str:
        parts: list[str] = []
        budget = _BODY_CHAR_BUDGET
        for idx in sorted(self.entries, reverse=True):
            text = self.entries[idx].text
            if not text:
                continue
            if budget - len(text) < 0:
                parts.append("…")
                break
            parts.append(text)
            budget -= len(text)
        parts.reverse()
        return "\n".join(parts)

    def build_card(self) -> dict:
        emoji, template = _STATE_STYLE.get(self.state, _STATE_STYLE["working"])
        title = f"{emoji} {self.title}".strip() if self.title else f"{emoji} Working"
        elements: list[dict] = []
        if self.answer:
            elements.append({"tag": "markdown", "content": to_lark_md(self.answer)})
        else:
            body = self._body()
            if body:
                elements.append({"tag": "markdown", "content": to_lark_md(body)})
        if elements:
            elements.append({"tag": "hr"})
        footer_bits = []
        if self.requester:
            footer_bits.append(f"requested by {self.requester}")
        footer_bits.append(self._elapsed())
        if self.answer and self.entries:
            footer_bits.append(f"{len(self.entries)} status updates")
        elements.append(
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": " · ".join(footer_bits)}],
            }
        )
        return {
            "config": {"wide_screen_mode": True, "update_multi": True},
            "header": {
                "title": {"tag": "plain_text", "content": title[:100]},
                "template": template,
            },
            "elements": elements,
        }
