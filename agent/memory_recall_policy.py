"""Identity-derived policy for auto-injected memory recall (#40170).

Auto-recalled memory — the ``<memory-context>`` block that
``build_turn_context()`` prefetches once per turn and injects into the user
message — carries operator-side observations (the operator, their contacts,
infra, and prior sessions). On a customer-facing channel that recall must not
be injected, or operator memory leaks to end users as a textbook indirect
prompt-injection surface (Greshake et al. 2023).

The boundary is derived from *who is asking*, not from a hardcoded platform
name list. Recall is operator-scoped (and therefore safe to inject) only when:

* the turn happens on one of the operator's own private local surfaces
  (CLI / TUI / desktop), where a single local principal both writes and reads
  memory; or
* the turn is a 1:1 gateway DM from an account the identity owner (the gateway)
  has positively confirmed to be the operator.

Everything else — group/channel/thread traffic, and DMs from anyone we cannot
confirm is the operator — suppresses auto-recall. The default is safe: an
operator opts back into gateway recall by identifying their own account, rather
than recall leaking by default. This replaces the earlier ad-hoc
``{telegram, discord, ...}`` platform set, which both missed platforms outside
the list and wrongly blocked the operator on their own channel.
"""

from __future__ import annotations

from typing import Optional

# The operator's own private surfaces. A single local principal writes and
# reads memory here, so auto-recall never crosses an operator/customer line.
# An empty platform covers internal/non-gateway agents (subagents, cron) that
# behaved as operator-scoped before this gate existed.
LOCAL_OPERATOR_PLATFORMS = frozenset({"cli", "tui", "desktop", "acp", "local", ""})

# Chat types that denote a 1:1 conversation with a single, stable counterpart.
_DM_CHAT_TYPES = frozenset({"dm", "direct", "private", ""})


def is_local_operator_surface(platform: Optional[str]) -> bool:
    """True for the operator's own local surfaces (never a shared channel)."""
    return (platform or "").strip().lower() in LOCAL_OPERATOR_PLATFORMS


def operator_scoped_recall(
    *,
    platform: Optional[str],
    chat_type: Optional[str],
    requester_is_operator: bool,
) -> bool:
    """Return True when auto-recall is operator-scoped and safe to inject.

    ``requester_is_operator`` is the identity decision made by the caller that
    owns identity (the gateway): whether the account driving this turn is the
    operator whose memory would be recalled. It is ignored on local surfaces.
    """
    if is_local_operator_surface(platform):
        return True
    # Shared multi-user contexts (groups, channels, forum threads) can never be
    # operator-private, regardless of who sent the current message.
    if (chat_type or "").strip().lower() not in _DM_CHAT_TYPES:
        return False
    return bool(requester_is_operator)


def suppress_memory_recall(
    *,
    platform: Optional[str],
    chat_type: Optional[str],
    requester_is_operator: bool,
) -> bool:
    """Inverse of :func:`operator_scoped_recall`, for setting the agent flag."""
    return not operator_scoped_recall(
        platform=platform,
        chat_type=chat_type,
        requester_is_operator=requester_is_operator,
    )


__all__ = [
    "LOCAL_OPERATOR_PLATFORMS",
    "is_local_operator_surface",
    "operator_scoped_recall",
    "suppress_memory_recall",
]
