"""``gateway.failure_reply``: one plain message in place of Hermes's own failure copy in chats.

Hermes words a failed turn for the person who can fix it: the provider, the attempt count, ``/retry``
and ``/model``. A bot that talks to a team (a front desk, a support room) must not show that to the
team. ``gateway.failure_reply.text`` swaps every failure message the gateway would post to a
messaging chat for that one line; the original goes to the gateway log at WARNING.

    gateway:
      failure_reply:
        text: "Sorry, something went wrong on my side. I'll get back to you here."
        scope: non_dm        # non_dm (default) = every chat except a DM; all = DMs too

Unset (the default) changes nothing. Local and programmatic surfaces (CLI, TUI, desktop, ``api_server``,
webhooks) always keep the raw copy: their reader is the operator or a program.

The failure paths call :func:`apply_failure_reply` with the copy they were about to send; a normal model
reply never reaches it. Terminal-failure *status* lines never need it: the chat already drops them
(``run_status_filter.is_terminal_failure_status``) because the failed turn's final reply carries the
message, and that reply is what gets replaced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from gateway.platforms.helpers import hash_for_log
from gateway.run_status_filter import status_chat_type

logger = logging.getLogger(__name__)

SCOPE_NON_DM = "non_dm"
SCOPE_ALL = "all"
# An unclassified chat type ("") is NOT a DM here: the option exists to keep internals out of shared
# chats, so a source nothing classified is hidden from, not trusted.
_DM_CHAT_TYPES = frozenset({"dm", "private", "direct"})


class ReplacedFailureText(str):
    """The configured plain message standing in for failure copy.

    ``run_turn`` checks it right after the reply is shaped, so it can drop what only belongs next to
    the real failure copy: the failed-turn notice, the runtime footer, the usage-limit model offer.
    """


@dataclass(frozen=True)
class FailureReplyPolicy:
    text: str
    all_chats: bool


def parse_policy(config: Any) -> Optional[FailureReplyPolicy]:
    """The policy in a raw config mapping, or None when ``gateway.failure_reply.text`` is unset/blank."""
    gateway = config.get("gateway") if isinstance(config, dict) else None
    section = gateway.get("failure_reply") if isinstance(gateway, dict) else None
    if not isinstance(section, dict):
        return None
    text = section.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    scope = str(section.get("scope") or SCOPE_NON_DM).strip().lower()
    if scope not in (SCOPE_NON_DM, SCOPE_ALL):
        logger.warning("gateway.failure_reply.scope %r is not %r or %r; using %r",
                       scope, SCOPE_NON_DM, SCOPE_ALL, SCOPE_NON_DM)
        scope = SCOPE_NON_DM
    return FailureReplyPolicy(text=text.strip(), all_chats=scope == SCOPE_ALL)


def failure_reply_for(source: Any) -> Optional[str]:
    """The plain message to send instead of failure copy to ``source``'s chat, else None."""
    from gateway.run import _gateway_surface_passes_raw_text, _load_gateway_config

    if source is None or _gateway_surface_passes_raw_text(getattr(source, "platform", None)):
        return None
    policy = parse_policy(_load_gateway_config())
    if policy is None:
        return None
    chat_type = status_chat_type(source)
    kind = str(getattr(chat_type, "value", chat_type) or "").strip().lower()
    if not policy.all_chats and kind in _DM_CHAT_TYPES:
        return None
    return policy.text


def apply_failure_reply(source: Any, failure_text: str, *, original: Optional[str] = None) -> str:
    """``failure_text`` unchanged, or a :class:`ReplacedFailureText` when the chat is configured for one.

    Only for text Hermes wrote about a failure: callers never pass a model's own reply. ``original`` is
    what the WARNING log records when it differs from ``failure_text`` (the raw provider text a category
    reply stands for). No ``source`` (a path that cannot name its chat) leaves the text alone.
    """
    if not failure_text or isinstance(failure_text, ReplacedFailureText):
        return failure_text
    replacement = failure_reply_for(source)
    if replacement is None:
        return failure_text
    from gateway.run import _redact_gateway_user_facing_secrets

    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", None))
    logger.warning(
        "Failure copy replaced by gateway.failure_reply: platform=%s chat=%s original=%s",
        platform, hash_for_log(getattr(source, "chat_id", "")),
        _redact_gateway_user_facing_secrets(str(original if original is not None else failure_text))[:500],
    )
    return ReplacedFailureText(replacement)
