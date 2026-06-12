"""Compact Discord channel operating memory for the gateway.

This module keeps durable, bounded, channel-scoped context outside the main
conversation transcript.  It deliberately stores compact summaries and source
receipts, not raw Discord channel logs.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from gateway.config import Platform
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from utils import atomic_json_write

_STATE_SUBDIR = Path("state") / "discord_channel_memory"
_DEFAULT_MAX_PROMPT_CHARS = 4000
_MAX_BULLETS = 24
_MAX_RECEIPTS = 40
_SIGNAL_PREFIXES = (
    "decision:",
    "decided:",
    "approved:",
    "implement:",
    "implemented:",
    "verified:",
    "canonical:",
    "link:",
    "file:",
    "repo:",
    "open loop:",
    "next action:",
    "todo:",
    "preference:",
)
_SIGNAL_WORDS = (
    "decision",
    "approved",
    "implemented",
    "verified",
    "canonical",
    "open loop",
    "next action",
)
_LOCK = threading.RLock()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean_part(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = text.strip("-._")
    return text[:80] or fallback


def _compact_line(text: str, *, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _dedupe_append(items: list[str], new_items: Iterable[str], *, max_items: int) -> list[str]:
    seen = {item.lower() for item in items}
    for item in new_items:
        item = _compact_line(item)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        items.append(item)
        seen.add(key)
    if len(items) > max_items:
        return items[-max_items:]
    return items


def _extract_durable_bullets(user_text: str, assistant_text: str) -> list[str]:
    """Conservative deterministic extractor for durable channel facts.

    This intentionally avoids summarizing every turn.  It only lifts explicit
    decision/implementation/link/open-loop style lines or short sentences with
    durable signal words.
    """
    combined = f"{user_text or ''}\n{assistant_text or ''}"
    candidates: list[str] = []
    for raw in re.split(r"[\n\r]+", combined):
        line = raw.strip(" -*\t")
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(_SIGNAL_PREFIXES) or any(word in lowered for word in _SIGNAL_WORDS):
            candidates.append(_compact_line(line))
    return candidates[:8]


def default_obsidian_root() -> Path:
    """Return the human-readable mirror root.

    Pete's configured vault path is used when present; otherwise we keep the
    mirror inside the profile state directory so non-Pete installations remain
    self-contained.
    """
    pete_vault = Path("/home/pete/winbuntu/vault/09 Hermes/Discord Channel Context")
    if pete_vault.parent.exists():
        return pete_vault
    return get_hermes_home() / _STATE_SUBDIR / "obsidian"


@dataclass(frozen=True)
class ChannelKey:
    platform: str
    guild_id: str
    chat_id: str
    parent_chat_id: str
    thread_id: str

    @property
    def storage_key(self) -> str:
        scope = self.thread_id or self.chat_id
        parts = [self.platform, self.guild_id or "noguild"]
        if self.thread_id:
            parts.extend(["parent", self.parent_chat_id or "noparent", "thread", scope])
        else:
            parts.extend(["channel", scope])
        return "__".join(_clean_part(part) for part in parts)


class DiscordChannelMemoryStore:
    """Profile-safe JSON state + Obsidian mirror for Discord channel briefs."""

    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        obsidian_root: Path | None = None,
        max_prompt_chars: int = _DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else get_hermes_home() / _STATE_SUBDIR
        self.channels_dir = self.base_dir / "channels"
        self.index_path = self.base_dir / "index.json"
        self.obsidian_root = Path(obsidian_root) if obsidian_root else default_obsidian_root()
        self.max_prompt_chars = max(500, int(max_prompt_chars or _DEFAULT_MAX_PROMPT_CHARS))

    def key_for_source(self, source: SessionSource) -> ChannelKey | None:
        if not source or source.platform != Platform.DISCORD:
            return None
        return ChannelKey(
            platform=source.platform.value,
            guild_id=str(source.guild_id or ""),
            chat_id=str(source.chat_id or ""),
            parent_chat_id=str(source.parent_chat_id or ""),
            thread_id=str(source.thread_id or ""),
        )

    def state_path_for_key(self, key: ChannelKey) -> Path:
        return self.channels_dir / f"{key.storage_key}.json"

    def load_record(self, source: SessionSource) -> dict[str, Any] | None:
        key = self.key_for_source(source)
        if key is None:
            return None
        path = self.state_path_for_key(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and not data.get("disabled"):
                return data
        except Exception:
            return None
        return None

    def build_injection_prompt(self, source: SessionSource) -> str:
        record = self.load_record(source)
        if not record:
            return ""
        bullets = list(record.get("durable_bullets") or [])
        receipts = list(record.get("source_receipts") or [])[-5:]
        lines = [
            "## Discord Channel Operating Memory",
            "Scope: guild/channel/thread only. Do not use in other channels.",
            f"Channel: {record.get('chat_name') or record.get('chat_id') or 'unknown'}",
        ]
        if record.get("chat_topic"):
            lines.append(f"Purpose/topic: {record['chat_topic']}")
        if record.get("thread_id"):
            lines.append(f"Thread: {record.get('thread_id')} (parent channel: {record.get('parent_chat_id') or 'unknown'})")
        if bullets:
            lines.append("Durable notes:")
            lines.extend(f"- {bullet}" for bullet in bullets)
        else:
            lines.append("Durable notes: none yet.")
        if receipts:
            lines.append("Sources:")
            for receipt in receipts:
                parts = []
                if receipt.get("session_id"):
                    parts.append(f"session={receipt['session_id']}")
                if receipt.get("message_id"):
                    parts.append(f"message={receipt['message_id']}")
                if receipt.get("timestamp"):
                    parts.append(f"time={receipt['timestamp']}")
                if parts:
                    lines.append(f"- {'; '.join(parts)}")
        if record.get("updated_at"):
            lines.append(f"Last updated: {record['updated_at']}")
        prompt = "\n".join(lines)
        if len(prompt) <= self.max_prompt_chars:
            return prompt
        suffix = "\n[Discord Channel Operating Memory truncated to fit configured cap.]"
        return prompt[: self.max_prompt_chars - len(suffix)].rstrip() + suffix

    def update_after_turn(
        self,
        source: SessionSource,
        *,
        user_text: str,
        assistant_text: str,
        session_id: str = "",
        message_id: str = "",
    ) -> dict[str, Any] | None:
        key = self.key_for_source(source)
        if key is None:
            return None
        bullets = _extract_durable_bullets(user_text, assistant_text)
        now = _now_iso()
        path = self.state_path_for_key(key)
        with _LOCK:
            record: dict[str, Any]
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    record = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    record = {}
            else:
                record = {}
            created_at = record.get("created_at") or now
            record.update(
                {
                    "platform": "discord",
                    "guild_id": source.guild_id or "",
                    "chat_id": source.chat_id or "",
                    "parent_chat_id": source.parent_chat_id or "",
                    "thread_id": source.thread_id or "",
                    "chat_name": source.chat_name or "",
                    "chat_type": source.chat_type or "",
                    "chat_topic": source.chat_topic or "",
                    "storage_key": key.storage_key,
                    "created_at": created_at,
                    "updated_at": now,
                    "last_seen_at": now,
                    "max_prompt_chars": self.max_prompt_chars,
                }
            )
            existing_bullets = list(record.get("durable_bullets") or [])
            record["durable_bullets"] = _dedupe_append(existing_bullets, bullets, max_items=_MAX_BULLETS)
            receipts = list(record.get("source_receipts") or [])
            receipts.append(
                {
                    "session_id": str(session_id or ""),
                    "message_id": str(message_id or source.message_id or ""),
                    "timestamp": now,
                }
            )
            record["source_receipts"] = receipts[-_MAX_RECEIPTS:]
            note_path = self._write_obsidian_mirror(record)
            record["obsidian_note_path"] = str(note_path)
            self.channels_dir.mkdir(parents=True, exist_ok=True)
            atomic_json_write(path, record)
            self._write_index()
            return record

    def _write_index(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for path in sorted(self.channels_dir.glob("*.json")) if self.channels_dir.exists() else []:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            entries.append(
                {
                    "storage_key": record.get("storage_key") or path.stem,
                    "chat_name": record.get("chat_name") or "",
                    "guild_id": record.get("guild_id") or "",
                    "chat_id": record.get("chat_id") or "",
                    "thread_id": record.get("thread_id") or "",
                    "updated_at": record.get("updated_at") or "",
                    "state_path": str(path),
                    "obsidian_note_path": record.get("obsidian_note_path") or "",
                }
            )
        atomic_json_write(self.index_path, {"updated_at": _now_iso(), "channels": entries})

    def _write_obsidian_mirror(self, record: dict[str, Any]) -> Path:
        safe_name = _clean_part(record.get("chat_name") or record.get("chat_id") or record.get("storage_key"))
        if record.get("thread_id"):
            safe_name = f"{safe_name}__thread-{_clean_part(record.get('thread_id'))}"
        note_dir = self.obsidian_root / "channels"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{safe_name}.md"
        bullets = record.get("durable_bullets") or []
        receipts = record.get("source_receipts") or []
        lines = [
            "# Discord Channel Context",
            "",
            f"Updated: {record.get('updated_at') or ''}",
            "",
            "## Identity",
            f"- Platform: discord",
            f"- Guild ID: `{record.get('guild_id') or ''}`",
            f"- Channel ID: `{record.get('chat_id') or ''}`",
            f"- Parent channel ID: `{record.get('parent_chat_id') or ''}`",
            f"- Thread ID: `{record.get('thread_id') or ''}`",
            f"- Channel name: {record.get('chat_name') or ''}",
            f"- Topic: {record.get('chat_topic') or ''}",
            "",
            "## Durable notes",
        ]
        lines.extend(f"- {bullet}" for bullet in bullets) if bullets else lines.append("- None yet.")
        lines.extend(["", "## Source receipts"])
        for receipt in receipts[-12:]:
            lines.append(
                f"- session `{receipt.get('session_id') or ''}`; message `{receipt.get('message_id') or ''}`; {receipt.get('timestamp') or ''}"
            )
        note_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        self._write_obsidian_index()
        return note_path

    def _write_obsidian_index(self) -> None:
        self.obsidian_root.mkdir(parents=True, exist_ok=True)
        index_path = self.obsidian_root / "00 Discord Channel Context Index.md"
        channel_dir = self.obsidian_root / "channels"
        lines = ["# Discord Channel Context Index", "", f"Updated: {_now_iso()}", "", "## Channels"]
        for note in sorted(channel_dir.glob("*.md")) if channel_dir.exists() else []:
            lines.append(f"- [[channels/{note.stem}|{note.stem}]]")
        index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
