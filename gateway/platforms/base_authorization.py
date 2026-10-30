"""Authorization callback invocation shared by platform adapters."""
from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional

from gateway.session import SessionSource

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)


def check_sender_authorization(
    adapter: BasePlatformAdapter,
    user_id: Optional[str], chat_type: Optional[str], chat_id: Optional[str], *,
    is_bot: bool = False, thread_id: Optional[str] = None,
    source: Optional[SessionSource] = None,
) -> Optional[bool]:
    """Return literal booleans or unknown; forward full sources only to compatible callbacks."""
    callback = adapter._authorization_check
    if not user_id or callback is None:
        return None
    extra: dict[str, Any] = {}
    if is_bot:
        extra["is_bot"] = True
    if thread_id is not None:
        extra["thread_id"] = thread_id
    try:
        if source is not None:
            parameters = inspect.signature(callback).parameters
            if "source" in parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                extra["source"] = source
        result = callback(user_id, chat_type, chat_id, **extra)
    except Exception:
        logger.warning("[%s] Authorization check raised for user %s; treating as unknown",
                       adapter.name, user_id, exc_info=True)
        return None
    if result is True or result is False:
        return result
    logger.warning("[%s] Authorization check returned %s for user %s; treating as unknown",
                   adapter.name, type(result).__name__, user_id)
    return None
