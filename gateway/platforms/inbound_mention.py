"""Pure, transport-neutral admission policy for inbound group messages.

Adapters retain their own configuration/env precedence and turn parsed transport
payloads into these facts.  Keeping the final decision here makes their defaults
and accepted addressing forms directly comparable without coupling this module to
any SDK.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_REQUIRE_MENTION: dict[str, bool] = {
    "telegram": False,
    "whatsapp": False,
    "dingtalk": False,
    "discord": True,
    "slack": True,
    "matrix": True,
    "mattermost": True,
    "feishu": True,
    "signal": False,
}
"""Current adapter defaults, documented beside the shared decision contract."""


@dataclass(frozen=True)
class InboundMentionFacts:
    """Facts already derived by an adapter from one inbound message."""

    is_dm: bool = False
    is_mentioned: bool = False
    matches_custom_pattern: bool = False
    command_addresses_bot: bool = False
    is_reply_to_bot: bool = False
    is_free_response_scope: bool = False
    is_participating_thread: bool = False


def resolve_inbound_mention_decision(
    facts: InboundMentionFacts, *, require_mention: bool,
) -> bool:
    """Return whether mention policy admits an inbound message.

    Access-control, bot-loop, allow-list, and adapter-specific thread-strictness
    checks intentionally remain outside this narrow policy decision.
    """
    return (
        facts.is_dm
        or facts.is_free_response_scope
        or facts.is_participating_thread
        or not require_mention
        or facts.is_mentioned
        or facts.matches_custom_pattern
        or facts.command_addresses_bot
        or facts.is_reply_to_bot
    )
