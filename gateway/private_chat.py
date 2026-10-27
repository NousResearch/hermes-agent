"""Is this gateway chat a confirmed one-to-one conversation with its sender?

``SessionSource.chat_type == "dm"`` cannot answer that: it is the dataclass default, the relay
fills it in when the connector omits the type, Slack labels group DMs (mpim) "dm" so they get
DM-style session and thread scoping, and a home-channel handoff without a thread is built as
"dm". Anything that treats the owner's private state as safe to use (channel memory decides
whether MEMORY.md / USER.md writes are the owner's) needs positive evidence instead, and
falls back to "not private" when there is none.

Evidence, in order:

* the adapter saw platform proof and set ``source.private_chat`` (Discord DMChannel,
  Mattermost channel type ``D``, Teams ``personal``, Matrix room with at most two members, ...);
* the platform addresses a one-to-one chat by the peer's own id, so ``chat_id`` equal to the
  sender's id cannot be a shared room (Telegram private chats, Signal, SMS, email, IRC, ...);
* the conversation id itself encodes the kind (Slack ``D…`` channels are 1:1 IMs; group DMs
  are ``G…``/``C…``). These two read only ``chat_id``/``user_id``, so they also hold for
  sources rebuilt from the relay wire.
"""

from __future__ import annotations

from typing import Any, Callable

DM_LABELS = frozenset({"dm", "private"})

# Adapters that build a DM source with ``chat_id`` = the sender's own id.
_PEER_ADDRESSED_PLATFORMS = frozenset({
    "telegram", "signal", "sms", "email", "irc", "line", "ntfy", "weixin", "whatsapp_cloud"})


def _slack_im(source: Any) -> bool:
    return str(source.chat_id or "").startswith("D")


_CHAT_ID_RULES: dict[str, Callable[[Any], bool]] = {"slack": _slack_im}


def _peer_addressed(source: Any) -> bool:
    chat = str(source.chat_id or "")
    return bool(chat) and chat in {str(source.user_id or ""), str(source.user_id_alt or "")}


def member_count_proves_private(count: Any) -> bool:
    """A known member count of at most two (the agent and one person) is 1:1 proof."""
    return isinstance(count, int) and not isinstance(count, bool) and 0 < count <= 2


def is_confirmed_private_chat(source: Any) -> bool:
    """True only on positive evidence that ``source`` is a 1:1 chat with its sender."""
    if str(source.chat_type or "").lower() not in DM_LABELS:
        return False
    if getattr(source, "private_chat", False) is True:
        return True
    platform = str(getattr(source.platform, "value", source.platform) or "")
    if platform in _PEER_ADDRESSED_PLATFORMS:
        return _peer_addressed(source)
    rule = _CHAT_ID_RULES.get(platform)
    return bool(rule and rule(source))
