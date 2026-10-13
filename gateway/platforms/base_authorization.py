"""Adapter authorization callbacks and runner-owned native slash policy."""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("gateway.platforms.base")


class BaseAuthorizationMixin:
    def set_slash_access_check(
        self,
        check: Optional[Callable[[Any, str], Optional[str]]],
    ) -> None:
        """Install the runner-owned slash-policy checker for native callbacks.

        The checker receives an actor-bearing ``SessionSource`` and a canonical
        command name.  It returns ``None`` when allowed and a denial string
        otherwise, matching ``GatewayRunner._check_slash_access``.
        """
        self._slash_access_check = check

    def set_authorization_check(
        self, callback: Optional[Callable[[str, Optional[str], Optional[str]], bool]]) -> None:
        """Register ``(user_id, chat_type, chat_id) -> bool``; adapters pulling external context
        (Slack thread replies) use it to flag non-allowlisted senders as unverified background."""
        self._authorization_check = callback

    def _is_sender_authorized(self, user_id: Optional[str], chat_type: Optional[str] = None,
                              chat_id: Optional[str] = None, *, is_bot: bool = False,
                              thread_id: Optional[str] = None) -> Optional[bool]:
        """True/False from the registered check, or None when no check exists ("trust unknown",
        legacy). ``is_bot``/``thread_id`` are forwarded as keywords only when set so legacy
        three-positional callbacks keep working. Only literal booleans propagate: a truthy
        non-boolean is "unknown", never an authorization that gates a credentialed side effect."""
        if not user_id or self._authorization_check is None:
            return None
        extra: dict[str, Any] = {}
        if is_bot:
            extra["is_bot"] = True
        if thread_id is not None:
            extra["thread_id"] = thread_id
        try:
            result = self._authorization_check(user_id, chat_type, chat_id, **extra)
        except Exception:
            logger.warning("[%s] Authorization check raised for user %s; treating as unknown",
                           self.name, user_id, exc_info=True)
            return None
        if result is True or result is False:
            return result
        logger.warning("[%s] Authorization check returned %s for user %s; treating as unknown",
                       self.name, type(result).__name__, user_id)
        return None
