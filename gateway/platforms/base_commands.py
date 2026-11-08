"""Exact plaintext gateway control grammar, separate from adapter/event handling."""
import re


_RESTART_PATTERNS = (
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?hermes\s+gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+hermes[.!?\s]*$", re.IGNORECASE))


def plaintext_gateway_command(text, *, archive_allowed, is_dm):
    text = text.strip()
    if archive_allowed and text.lower() in {'archive', 'archive this', 'archive the chat'}:
        return '/archive'
    if is_dm and any(pattern.match(text) for pattern in _RESTART_PATTERNS):
        return '/restart'
    return None
