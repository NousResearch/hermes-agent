"""The identity of messages removed by an authenticated platform notification."""

from dataclasses import dataclass

from gateway.config import Platform
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
        return (
            bool(self.chat_id) and bool(self.message_ids)
            and not event.internal and source is not None
            and source.platform == self.platform and source.scope_id == self.scope_id
            and source.chat_id == self.chat_id and event.message_id in self.message_ids
            and (self.thread_id is None or source.thread_id == self.thread_id)
        )
