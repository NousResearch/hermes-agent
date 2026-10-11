"""The identity of messages removed by an authenticated platform notification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gateway.config import Platform

if TYPE_CHECKING:
    from gateway.platforms.event import MessageEvent


@dataclass(frozen=True)
class NativeMessageDeletion:
    platform: Platform
    scope_id: str | None
    chat_id: str
    message_ids: tuple[str, ...]
    thread_id: str | None = None

    def matches(self, event: MessageEvent) -> bool:
        source = event.source
        if event.internal or source is None:
            return False
        origin_chat_id = (
            source.parent_chat_id
            if source.platform == Platform.DISCORD and source.auto_thread_created
            else source.chat_id
        )
        return (
            bool(self.chat_id) and bool(self.message_ids)
            and source.platform == self.platform and source.scope_id == self.scope_id
            and origin_chat_id == self.chat_id and event.message_id in self.message_ids
            and (self.thread_id is None or source.thread_id == self.thread_id)
        )
