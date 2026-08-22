"""Shared real-user vs runtime-scaffolding predicates.

Catalog extraction must use the same real-user semantics as conversation
compression, plus the extra synthetic prefixes/metadata that must never
enter catalog story/topics. Lives here so ``catalog_residual`` can reuse
the predicates without importing ``conversation_compression`` (that
module imports ``context_compressor``, which already imports the catalog).
"""

from __future__ import annotations

from typing import Any

# Metadata flags that mark user-role scaffolding. SessionDB projection drops
# underscore-prefixed keys, so content prefixes below remain authoritative
# after persistence. Keep this tuple identical to conversation compression.
SYNTHETIC_USER_FLAGS = (
    "_todo_snapshot_synthetic",
    "_empty_recovery_synthetic",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_dropped_toolcall_nudge",
)

# Canonical prefixes used by conversation compression's real-user predicate.
SYNTHETIC_USER_PREFIXES = (
    "[System: Your previous response was truncated",
    "[System: The previous response was cut off",
    "[System: Your previous tool call",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
)

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
    """Same real-user semantics as conversation compression."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if any(message.get(flag) for flag in SYNTHETIC_USER_FLAGS):
        return False
    text = user_message_text(message).strip()
    if not text:
        return False
    if text.startswith(SYNTHETIC_USER_PREFIXES):
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
