"""Credential masking for anything that outlives the process that received it.

A stored transcript and a plugin's copy of the conversation are both read back long after the
turn that produced them, so a user's ``security.redact_secrets`` preference must not decide
whether they were masked. Every caller uses the redactor the log path and the request-dump path
already run, and nothing here re-implements a pattern: a second list would drift from the first,
which is the failure this module exists to prevent.

Why the helpers are not in ``agent/redact.py``: that module and the transcript write funnel
(``persistence/sqlite/conversation/_store/messages.py``) both sat within twenty lines of the
repository's 2000-line file cap when this landed (``scripts/code_health``), so neither could grow
for a security fix without an exception to the ratchet or a package split of a core module. This
module is the smallest home that keeps one implementation and needs neither.
"""

from __future__ import annotations

from typing import Any


def redact_message_text(text: str, *, force: bool = True) -> str:
    """``text`` masked for a boundary that must never persist or replay a raw credential.

    The same redactor every other boundary uses. ``force`` is on by default here because these
    callers hand text to something that outlives the process that received it — a stored
    transcript, or a plugin's own store — so a user's ``security.redact_secrets`` preference must
    not quietly reopen the hole. Late import: ``agent.redact`` loads the vault registry, which
    must not run at package-import time.
    """
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(text, force=force)


def redact_message_content(content: Any, *, force: bool = True) -> Any:
    """Redact the string leaves of a message ``content`` payload, structure intact.

    Accepts plain text or a list of multimodal parts: only ``text`` values are rewritten, so an
    image part, a ``type`` discriminator and every other payload key survive a round-trip. A
    false positive here corrupts a transcript silently, which is why the masking is the shared
    redactor's — a caller's own pattern list would drift from it.
    """
    if isinstance(content, str):
        return redact_message_text(content, force=force)
    if isinstance(content, list):
        return [
            {**part, "text": redact_message_text(part["text"], force=force)}
            if isinstance(part, dict) and isinstance(part.get("text"), str) else part
            for part in content
        ]
    return content
