"""Shared real-user vs runtime-scaffolding predicates.

Catalog extraction must use the same real-user semantics as conversation
compression, plus the extra synthetic prefixes/metadata that must never
enter catalog story/topics. Lives here so ``catalog_residual`` can reuse
the predicates without importing ``conversation_compression`` (that
module imports ``context_compressor``, which already imports the catalog).
"""

from __future__ import annotations

from typing import Any

from agent.conversation_compression_scaffolding import (
    _SYNTHETIC_USER_FLAGS,
    _SYNTHETIC_USER_PREFIXES,
    _message_text,
    is_user_role_scaffolding,
)

# Flags and prefixes are main's scaffolding definitions (agent/conversation_compression_scaffolding.py,
# a leaf module), so catalog extraction and conversation compression cannot drift apart.
SYNTHETIC_USER_FLAGS = _SYNTHETIC_USER_FLAGS
SYNTHETIC_USER_PREFIXES = _SYNTHETIC_USER_PREFIXES

# Additional scaffolding catalog extraction must ignore. Broader than
# conversation compression so todo snapshots, cron echoes, and handoff
# stems never become story/topics.
CATALOG_SYNTHETIC_USER_PREFIXES = SYNTHETIC_USER_PREFIXES + (
    "[System:",
    "[CONTEXT COMPACTION",
    "[CONTEXT SUMMARY]",
    "[PRIOR CONTEXT",
    "[Planning state preserved",
    "[ASYNC DELEGATION",
    "[OUT-OF-BAND",
    "Cronjob Response:",
)


def user_message_text(message: Any) -> str:
    """Best-effort text view of a message's content."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def is_real_user_turn(message: Any) -> bool:
    """Same real-user semantics as conversation compression's ``_is_real_user_message``."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if is_user_role_scaffolding(message) or not _message_text(message).strip():
        return False
    from agent.context_compressor import ContextCompressor

    return not ContextCompressor._is_synthetic_compression_user_turn(message)


def is_catalog_user_turn(message: Any) -> bool:
    """Human ask that may enter catalog story/topics."""
    if not is_real_user_turn(message):
        return False
    if message.get("_compressed_summary"):
        return False
    text = user_message_text(message).strip()
    return not text.startswith(CATALOG_SYNTHETIC_USER_PREFIXES)
