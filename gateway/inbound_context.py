"""Bounded owner for inbound reply-context projection (#101866).

Reply-target identity is known on the inbound event, but the busy/queued
projection used to drop it before the next mutation seam: a Telegram
reply-quote arriving mid-turn reached the agent as bare text, the agent
resolved "the same" to the most recent draft, and sent to the wrong
recipient. The `[Replying to ...]` pointer existed only on the idle path.

This module is the single owner of reply-context projection + its audit
log evidence. Both the idle path and the queued/steered follow-up paths
consume it through :func:`build_reply_to_prefix`; nothing else constructs
the prefix. gateway/run.py only delegates (#54962 monolith policy:
run.py must not grow new safety-critical mutation seams).
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPLY_SNIPPET_MAX_CHARS = 500


def build_reply_to_prefix(event: Any) -> str:
    """Build the ``[Replying to: ...]`` disambiguation prefix, or ``''``.

    Always inject the reply-to pointer when the event carries BOTH
    ``reply_to_text`` and ``reply_to_message_id`` — even when the quoted
    text already appears in history. The prefix isn't deduplication, it's
    disambiguation: it tells the agent *which* prior message the user is
    referencing. History can contain the same or similar text multiple
    times, and without an explicit pointer the agent has to guess (or
    answer for both subjects). Token overhead is minimal.

    Contracts preserved from the original idle-path implementation:

    - no prefix when either ``reply_to_text`` or ``reply_to_message_id``
      is absent (a lone orphan quote must NOT produce a pointer)
    - own-message form: ``[Replying to your previous message: "..."]``
      when ``reply_to_is_own_message``; foreign form otherwise
    - snippet truncated to 500 chars
    """
    reply_to_text = getattr(event, "reply_to_text", None)
    reply_to_message_id = getattr(event, "reply_to_message_id", None)
    if not reply_to_text or not reply_to_message_id:
        return ""
    reply_snippet = str(reply_to_text)[:_REPLY_SNIPPET_MAX_CHARS]
    if getattr(event, "reply_to_is_own_message", False):
        return f'[Replying to your previous message: "{reply_snippet}"]\n\n'
    return f'[Replying to: "{reply_snippet}"]\n\n'


def apply_reply_to_prefix(message_text: str, event: Any) -> str:
    """Prepend the reply-to pointer to *message_text* (no-op without context).

    Shared by the idle inbound path and the queued/steered follow-up
    path — exactly the two consumers that previously diverged (#101866).
    Injecting exactly once is guaranteed by construction: the prefix is
    prepended here and nowhere else.
    """
    prefix = build_reply_to_prefix(event)
    if not prefix:
        return message_text
    return f"{prefix}{message_text}"


def log_inbound_reply_context(
    *,
    source: Any,
    message_text: str,
    event: Optional[Any] = None,
    queued: bool = False,
) -> None:
    """Emit the inbound-message audit line, including reply-target identity.

    The idle path always logged this; queued/steered follow-ups used to be
    invisible in gateway.log exactly when an operator needed to trace
    "did the mid-turn message arrive, and with what reply context?" — the
    incident-report debugging blind spot in #101866. One implementation,
    consumed by both paths; the ``queued`` marker distinguishes them.
    """
    reply_to_id = getattr(event, "reply_to_message_id", None) if event else None
    reply_to_text = ""
    if event:
        reply_to_text = (
            (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " ")
        )
    logger.info(
        "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r queued=%s",
        getattr(source, "platform", None),
        getattr(source, "user_id", None),
        getattr(source, "chat_id", None),
        (message_text or "")[:80],
        reply_to_id,
        reply_to_text,
        queued,
    )
