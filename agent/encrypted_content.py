"""Responses encrypted fields are opaque context, never truncatable text."""

from typing import Any

# Provider field limit in characters, unrelated to the model's token window.
MAX_ENCRYPTED_CONTENT_CHARS = 20_971_520


def has_oversized_latest_checkpoint(agent: Any, messages: Any) -> bool:
    """Check only checkpoints selected by the current request's replay policy.

    Reuse native eligibility, issuer filtering, deduplication and newest-checkpoint
    pruning before compression can replace context needed for an actual replay.
    Ineligible routes retain their existing omission/no-pruning behavior.
    """
    from agent.codex_responses_adapter import _native_responses_replay_items

    items = _native_responses_replay_items(agent, messages)
    return any(
        item.get("type") == "compaction"
        and isinstance(item.get("encrypted_content"), str)
        and len(item["encrypted_content"]) > MAX_ENCRYPTED_CONTENT_CHARS
        for item in items or []
    )


class EncryptedContentTooLarge(ValueError):
    """Replay cannot proceed safely; the retained context must remain intact."""

    def __init__(self, *, item_type: str, index: int, length: int) -> None:
        super().__init__(
            f"Responses input[{index}].encrypted_content ({item_type}, {length} chars) "
            f"exceeds the {MAX_ENCRYPTED_CONTENT_CHARS}-character limit. "
            "Encrypted context cannot be truncated or silently removed."
        )
