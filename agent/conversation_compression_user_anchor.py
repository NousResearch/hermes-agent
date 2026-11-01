"""Merge a real user anchor with compaction scaffolding without losing metadata."""

from typing import Any


def _merge_anchor_into_user_message(target: dict, anchor: dict) -> None:
    """Fold the human anchor into an existing user-role scaffolding turn.
    Used only when any insertion would create consecutive user turns. Anchor text leads, scaffolding follows,
    and synthetic flags are cleared."""
    from agent.conversation_compression import _replace_message_content, _SYNTHETIC_USER_FLAGS
    from agent.conversation_compression_ledger import ledger_message_kind
    has_ledger = bool(ledger_message_kind(target))
    anchor_content = anchor.get("content")
    target_content = target.get("content")
    if isinstance(anchor_content, list) or isinstance(target_content, list):

        def _parts(content: Any) -> list:
            return list(content) if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]

        _replace_message_content(target, _parts(anchor_content) + _parts(target_content))
    else:
        merged = f"{anchor_content or ''}\n\n{target_content or ''}".strip()
        _replace_message_content(target, merged)
    if has_ledger:
        target["_decision_ledger_embedded"] = True
        target.setdefault("display_metadata", {})["decision_ledger"] = "embedded"
    for flag in _SYNTHETIC_USER_FLAGS:
        target.pop(flag, None)
    # The anchor's text leads the composite, so the fold keeps the anchor's uid and records the
    # scaffolding turn's (merge witness).
    from agent.message_metadata import record_absorbed_message

    record_absorbed_message(target, anchor, dropped_leads=True)
