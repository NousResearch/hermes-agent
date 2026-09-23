"""Signal outbound egress allowlist — the one decision shared by the gateway adapter
(``SignalAdapter._with_target``) and the standalone JSON-RPC sender
(``tools/send_message_senders.py::_send_signal``), so neither path can bypass the other.

Env reads go through ``get_scoped_secret`` at call time so the owning profile's scope
applies (cron ticker, ``hermes send``, gateway turn)."""

from typing import AbstractSet, FrozenSet, List

from gateway.platforms._shared import get_scoped_secret

BLOCKED_ERROR = "outbound blocked: target not on allowlist"


def parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def group_allowlist() -> FrozenSet[str]:
    return frozenset(parse_comma_list(get_scoped_secret("SIGNAL_GROUP_ALLOWED_USERS", "")))


def dm_send_allowlist() -> FrozenSet[str]:
    """Explicit SIGNAL_SEND_ALLOWED_USERS wins; otherwise fall back to the inbound
    SIGNAL_ALLOWED_USERS, and if that is open ("*") or unset fail closed (no DM sends)."""
    explicit = get_scoped_secret("SIGNAL_SEND_ALLOWED_USERS", "")
    if explicit:
        return frozenset(parse_comma_list(explicit))
    inbound = frozenset(parse_comma_list(get_scoped_secret("SIGNAL_ALLOWED_USERS", "*")))
    return frozenset() if "*" in inbound else inbound


def outbound_allowed(chat_id: str, dm_send_allow: AbstractSet[str], group_allow: AbstractSet[str]) -> bool:
    """Groups: allowed only when on the group allowlist (unset = no group sends). DMs: allowed
    only when on the outbound DM allowlist."""
    if chat_id.startswith("group:"):
        return chat_id[6:] in group_allow
    return chat_id in dm_send_allow
