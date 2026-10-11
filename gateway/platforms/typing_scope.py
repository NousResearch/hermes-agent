"""Topic-local typing state with compatibility for older platform adapters."""
import inspect

from gateway.config import Platform


def typing_scope(platform, chat_id, metadata=None):
    thread_id = (metadata or {}).get("thread_id")
    if platform == Platform.TELEGRAM and thread_id is not None:
        return (str(chat_id), str(thread_id))
    return str(chat_id)


def set_typing_paused(adapter, chat_id, *, paused, metadata=None):
    """Pass topic metadata only to adapters that accept it (third-party overrides)."""
    method = adapter.pause_typing_for_chat if paused else adapter.resume_typing_for_chat
    params = inspect.signature(method).parameters
    supports_metadata = "metadata" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
    method(chat_id, **({"metadata": metadata} if supports_metadata else {}))


class TypingPauseMixin:
    def pause_typing_for_chat(self, chat_id: str, metadata=None) -> None:
        """Pause a Telegram topic (or a whole chat when no topic is supplied)."""
        self._typing_paused.add(typing_scope(self.platform, chat_id, metadata))

    def resume_typing_for_chat(self, chat_id: str, metadata=None) -> None:
        """Resume the matching scope without unpausing another task's prompt."""
        self._typing_paused.discard(typing_scope(self.platform, chat_id, metadata))

