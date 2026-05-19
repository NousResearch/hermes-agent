"""File-backed queue store abstraction for gateway busy queue.

Keeps persistence logic isolated from GatewayRunner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource


class FileBusyQueueStore:
    def __init__(self, path: Path, fingerprint_fn):
        self.path = path
        self._fingerprint_fn = fingerprint_fn

    def save(self, queued_events: Dict[str, List[dict]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, List[dict]] = {}
        for session_key, items in (queued_events or {}).items():
            serial_items: List[dict] = []
            for item in items:
                ev = item.get("event")
                if ev is None:
                    continue
                source = ev.source
                serial_items.append(
                    {
                        "priority": item.get("priority", "P1"),
                        "created_at": item.get("created_at", time.time()),
                        "fingerprint": item.get("fingerprint") or self._fingerprint_fn(ev),
                        "event": {
                            "text": ev.text,
                            "message_type": ev.message_type.value if ev.message_type else "text",
                            "source": {
                                "platform": source.platform.value if source and source.platform else None,
                                "user_id": source.user_id if source else None,
                                "chat_id": source.chat_id if source else None,
                                "chat_type": source.chat_type if source else None,
                                "thread_id": source.thread_id if source else None,
                            },
                            "message_id": ev.message_id,
                            "media_urls": list(ev.media_urls or []),
                            "media_types": list(ev.media_types or []),
                            "reply_to_message_id": ev.reply_to_message_id,
                            "reply_to_text": ev.reply_to_text,
                            "channel_prompt": ev.channel_prompt,
                            "auto_skill": ev.auto_skill,
                        },
                    }
                )
            if serial_items:
                payload[session_key] = serial_items
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> Dict[str, List[dict]]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        if not isinstance(data, dict):
            return {}

        loaded: Dict[str, List[dict]] = {}
        for session_key, items in data.items():
            bucket: List[dict] = []
            for item in (items or []):
                evd = item.get("event") or {}
                srcd = evd.get("source") or {}
                platform = srcd.get("platform")
                source = SessionSource(
                    platform=Platform(platform) if platform else None,
                    user_id=srcd.get("user_id"),
                    chat_id=srcd.get("chat_id"),
                    chat_type=srcd.get("chat_type"),
                    thread_id=srcd.get("thread_id"),
                )
                event = MessageEvent(
                    text=evd.get("text", ""),
                    message_type=MessageType(evd.get("message_type", "text")),
                    source=source,
                    message_id=evd.get("message_id"),
                    media_urls=list(evd.get("media_urls") or []),
                    media_types=list(evd.get("media_types") or []),
                    reply_to_message_id=evd.get("reply_to_message_id"),
                    reply_to_text=evd.get("reply_to_text"),
                    channel_prompt=evd.get("channel_prompt"),
                    auto_skill=evd.get("auto_skill"),
                )
                bucket.append(
                    {
                        "event": event,
                        "priority": item.get("priority", "P1"),
                        "created_at": float(item.get("created_at", time.time())),
                        "fingerprint": item.get("fingerprint") or self._fingerprint_fn(event),
                    }
                )
            if bucket:
                loaded[session_key] = bucket
        return loaded
