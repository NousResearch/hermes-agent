"""API-only assembly of observed group context around the prepared current turn."""
from typing import Any, Optional

from gateway.session import defang_verified_sender_claims, wrap_with_verified_sender_note


def _wrap_observed_text(text: str, prefix: str) -> str:
    # Inbound preparation has already defanged the current body. Only its leading
    # gateway-authored note may survive; never extract a note from observed rows.
    note, separator, body = text.partition("\n\n")
    if separator and note.startswith("[Gateway-verified sender: ") and note.endswith("]") and "\n" not in note:
        return wrap_with_verified_sender_note(prefix + body, note)
    return prefix + text


def wrap_current_message_with_observed_context(message: Any, observed_context: Optional[str]) -> Any:
    """Keep the prepared sender note outside observed context; leave stored rows alone."""
    if not observed_context:
        return message
    from gateway.run import _CURRENT_ADDRESSED_MESSAGE_HEADER, _OBSERVED_GROUP_CONTEXT_HEADER

    observed_context = defang_verified_sender_claims(observed_context)
    prefix = f"{_OBSERVED_GROUP_CONTEXT_HEADER}\n{observed_context}\n\n{_CURRENT_ADDRESSED_MESSAGE_HEADER}\n"
    if isinstance(message, str):
        return _wrap_observed_text(message, prefix)
    if isinstance(message, list):
        wrapped = [dict(part) if isinstance(part, dict) else part for part in message]
        for part in wrapped:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = _wrap_observed_text(part.get("text", ""), prefix)
                return wrapped
        return [{"type": "text", "text": prefix.rstrip()}] + wrapped
    return message
