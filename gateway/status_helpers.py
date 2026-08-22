"""Compression-status helpers extracted from gateway/run.py (#54962).

Fourth slice of the gateway god-file unpacking: the status-template-to-regex
compiler and the opt-in compression-progress matcher. Pure derivation — the
regex is built from the SAME template constants the emit sites format, so
wording drift between agent/conversation_compression.py and the gateway
matcher cannot silently diverge.
"""

from __future__ import annotations

import re

from agent.conversation_compression import (
    COMPACTION_STATUS,
    COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE,
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE,
    IDLE_COMPACTION_STATUS_TEMPLATE,
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
    PRE_API_COMPRESSION_STATUS_TEMPLATE,
)


def _status_template_to_regex(template: str) -> str:
    """Compile a compression status template constant into a regex source.

    Literal text is escaped verbatim (so wording drift in
    agent/conversation_compression.py cannot silently diverge from this
    matcher — the constants ARE the wording) and each ``{field}`` format
    placeholder is replaced with a numeric-ish pattern covering every value
    the emit sites format in (ints, ``{:,}`` thousands separators).
    """
    parts = re.split(r"\{[^{}]*\}", template)
    return r"[\d,]+".join(re.escape(part) for part in parts)


# ROUTINE compression progress statuses, derived from the SAME template
# constants the emit sites format (agent/conversation_compression.py, #69550)
# — never re-inlined wording. Used ONLY by the opt-in
# ``compression.progress_notices`` gate below (#52995) to decide which of the
# noisy statuses matched by _TELEGRAM_NOISY_STATUS_RE are compression
# progress (deliverable when the user opted in) versus unrelated aux/retry
# chatter (always suppressed on chat surfaces). Failure notices and manual
# /compress feedback never match _TELEGRAM_NOISY_STATUS_RE in the first
# place, so they are unaffected by this gate.
_COMPRESSION_PROGRESS_STATUS_RE = re.compile(
    "|".join(
        _status_template_to_regex(_template)
        for _template in (
            COMPACTION_STATUS,
            PRE_API_COMPRESSION_STATUS_TEMPLATE,
            PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
            IDLE_COMPACTION_STATUS_TEMPLATE,
            COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE,
            COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE,
            COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
            COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE,
        )
    ),
    re.IGNORECASE,
)
