"""Disposable WhatsApp identity views for diagnostics; never use for routing or storage."""

import re


_PHONE_IDENTITY = re.compile(
    r"(?:\+?[1-9]\d{6,14}|\+?\d{2,3}\*{4}\d{2,4})(?:@(?:s\.whatsapp\.net|c\.us))?"
)


def _session_platform(session_key: object) -> str:
    parts = str(session_key).split(":", 3)
    return parts[2] if len(parts) >= 3 else ""


def session_key_for_log(session_key: object) -> str:
    """Return a log-safe routing key without changing its runtime identity.

    WhatsApp and WhatsApp Cloud DM keys contain phone-derived identifiers.
    Other platforms can legitimately use comparable numeric IDs, so the
    opt-in bare-phone redactor is restricted to the structured platform slot.
    """
    value = str(session_key)
    if _session_platform(value) not in {"whatsapp", "whatsapp_cloud"}:
        return value

    parts = value.split(":", 4)
    if len(parts) < 5:
        return value
    return ":".join(parts[:4]) + ":" + log_safe_gateway_identity(parts[2], parts[4])


def log_safe_gateway_identity(platform: object, value: object) -> str:
    """Return an identity suitable for a diagnostic log argument.

    WhatsApp and WhatsApp Cloud use phone-derived identifiers for users and
    chats.  Keep the platform scoping at this boundary: numeric identifiers
    on other platforms remain useful diagnostics, while WhatsApp values are
    either partially masked phone-shaped text or presence-only metadata.
    This helper is for rendered logger arguments only; callers must continue
    using the original value for routing, authorization, persistence, and
    provider calls.
    """
    if value is None:
        return "absent"
    text = str(value)
    platform_value = getattr(platform, "value", platform)
    if str(platform_value or "").lower() not in {"whatsapp", "whatsapp_cloud"}:
        return text

    # A display name or opaque LID must not become visible merely because a
    # phone substring can be masked. Recognize only a complete phone identity
    # (or its already masked form); all other values are presence metadata.
    if not _PHONE_IDENTITY.fullmatch(text):
        return "present" if text else "absent"

    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(
        text,
        force=True,
        redact_bare_phone_numbers=True,
    )


def log_safe_gateway_error(platform: object, error: object) -> str:
    """WhatsApp error bodies can echo content; retain other platforms' diagnostic detail."""
    name = str(getattr(platform, "value", platform) or "").lower()
    if name not in {"whatsapp", "whatsapp_cloud"}:
        return str(error or "")
    if isinstance(error, BaseException):
        return type(error).__name__
    return "present" if error else "absent"


def session_error_for_log(session_key: object, error: object) -> str:
    """Select the error projection from the session key's structured platform slot."""
    return log_safe_gateway_error(_session_platform(session_key), error)
