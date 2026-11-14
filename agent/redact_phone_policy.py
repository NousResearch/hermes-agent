"""Chat-egress phone-number policy: ``security.redact_phone_numbers`` (default true, #68911).

Phone numbers stay masked in the agent's replies and in text the gateway sends to chat platforms
unless the profile's config.yaml sets ``security.redact_phone_numbers: false``. That opt-in only
skips the E.164 pass of those two scrubs (the stored assistant reply and the forced chat-egress
scrub): API keys, bearer tokens and every other secret pattern stay masked, and logs and tool output
keep masking phone numbers.
"""

from __future__ import annotations

import logging

from agent.redact import redact_for_egress, redact_sensitive_text

logger = logging.getLogger(__name__)
_WARNED: set[str] = set()


def chat_keeps_phone_numbers() -> bool:
    """True only when the active profile's config.yaml sets ``security.redact_phone_numbers: false``.
    Any other value, or a config that cannot be read, keeps phone numbers masked (fails closed)."""
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly().get("security") or {}).get("redact_phone_numbers", True)
    except Exception:  # health: allow BLE001 -- an unreadable config keeps the masked default (fails closed)
        return False
    if isinstance(raw, bool):
        return not raw
    if repr(raw) not in _WARNED:
        _WARNED.add(repr(raw))
        logger.warning("security.redact_phone_numbers must be true or false, got %r; phone numbers stay masked", raw)
    return False


def redact_for_chat_egress(text: str) -> str:
    """``redact_for_egress`` for text sent to a chat platform, honouring the phone-number opt-in."""
    return redact_for_egress(text, keep_phone_numbers=chat_keeps_phone_numbers())


def redact_assistant_reply(text: str) -> str:
    """The stored assistant reply's scrub (follows ``security.redact_secrets``), with the same opt-in."""
    return redact_sensitive_text(text, keep_phone_numbers=chat_keeps_phone_numbers())
