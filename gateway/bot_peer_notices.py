"""What the runtime itself may send to the author of an inbound message (#137027).

A bot peer reads every message in its chat as a new inbound turn, so a runtime-written notice (busy
ack, redirect, status, progress, warnings, failure copy, delegation failures) sent to it starts a
turn there, whose reply starts one here. The model's own reply is the only thing a bot peer gets.
Every automatic-notice egress asks ``automatic_notices_allowed``; none re-derives the rule.
"""

from __future__ import annotations

from typing import Any


def automatic_notices_allowed(source: Any) -> bool:
    """False when the message came from a known bot (``SessionSource.is_bot``, a bool: the platform's
    bot flag or the adapter's ``bot_users`` opt-in). Anything but a real ``True`` is not a known bot."""
    return getattr(source, "is_bot", False) is not True


def setup_notices_allowed(source: Any, config: Any = None) -> bool:
    """May a setup/operational notice this runtime writes reach ``source``'s author?

    A setup notice — the first-contact onboarding note and the one-time "no home channel" prompt —
    is the operator's business, never a member's: on a served instance the human who writes in is
    not its operator, and the notice both announces the platform's own setup and invites the model
    to explain it back. The audience is declared, not hardcoded: ``setup_notices`` on the platform
    block reads ``author`` (the default: today's behaviour, the notice goes to whoever wrote),
    ``operator`` (only the authors listed in ``operator_ids``) or ``off`` (never).

    An empty operator list names nobody, so ``operator`` with no ids declared keeps the default
    instead of silently muting the operator's own notice (``#137027``'s shape: a rule nobody
    declared is not a rule).
    """
    if source is None:
        return True
    get_setup_notices = getattr(config, "get_setup_notices", None)
    mode = get_setup_notices(getattr(source, "platform", None)) if callable(get_setup_notices) else "author"
    if mode == "off":
        return False
    if mode != "operator":
        return True
    get_operator_ids = getattr(config, "get_operator_ids", None)
    declared: Any = get_operator_ids(getattr(source, "platform", None)) if callable(get_operator_ids) else ()
    if not isinstance(declared, (list, tuple, set, frozenset)) or not declared:
        return True
    author = str(getattr(source, "user_id", "") or "").strip()
    return bool(author) and author in {str(value).strip() for value in declared}
