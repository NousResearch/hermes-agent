"""Token price of encrypted reasoning that is replayed on later requests.

Codex/xAI ``encrypted_content`` and Claude thinking ``signature`` are ciphertext, so their bytes are
not their cost; local estimates used to price them at 0 (#100611: a 5M-char native compaction
checkpoint read as 1.29M tokens). But the provider bills each replayed reasoning item as INPUT at the
reasoning count it reported for the turn that minted it (live A/B, same request with and without the
item: gpt-5.6-sol reported 2,765 -> +2,767 billed; claude-opus-5.5 reported 3,959 -> +3,959 billed,
while its readable summary was ~1,016). Pricing it at 0 hid most of a long Codex/xAI session from the
compaction cut, so compression found "nothing to cut" while the bill sat over the window (#125920).

A turn's reasoning is priced at the count stamped on it (``message_metadata.stamp_reasoning_tokens``).
Compaction checkpoints stay at 0: they are cumulative state, priced only by real usage.
"""

from __future__ import annotations

from typing import Any

from hermes_agent.agent.message_metadata import stamped_reasoning_tokens

# Turns persisted before the stamp existed: ciphertext characters per billed reasoning token, measured
# live per issuer (OpenAI/Codex ~13.5, xAI grok-4.7 ~1.6). An estimate, used only when no stamp exists.
_LEGACY_CHARS_PER_TOKEN = {"xai_responses": 1.6}
_DEFAULT_LEGACY_CHARS_PER_TOKEN = 13.5


def strip_opaque_replay_items(items: Any) -> Any:
    """``codex_reasoning_items`` with ``encrypted_content`` blanked; the ciphertext is priced separately
    (:func:`codex_reasoning_replay_tokens`), never by its bytes as text."""
    if not isinstance(items, list):
        return items
    return [
        {k: ("" if k == "encrypted_content" else v) for k, v in item.items()} if isinstance(item, dict) else item
        for item in items
    ]


def codex_reasoning_replay_tokens(msg: Any) -> int:
    """Billed input cost of the encrypted reasoning an assistant row (or a converted Responses
    ``reasoning`` input item) replays. Compaction checkpoints are excluded."""
    if isinstance(msg, dict) and msg.get("type") == "reasoning":
        return _legacy_tokens([msg])
    items = msg.get("codex_reasoning_items") if isinstance(msg, dict) and msg.get("role") == "assistant" else None
    if not isinstance(items, list):
        return 0
    reasoning = [item for item in items if isinstance(item, dict) and item.get("type") != "compaction"]
    if not _legacy_tokens(reasoning):
        return 0
    return stamped_reasoning_tokens(msg) or _legacy_tokens(reasoning)


def _legacy_tokens(items: list[Any]) -> int:
    return sum(
        int(len(item["encrypted_content"]) / _LEGACY_CHARS_PER_TOKEN.get(
            str(item.get("_issuer_kind") or ""), _DEFAULT_LEGACY_CHARS_PER_TOKEN))
        for item in items
        if isinstance(item, dict) and isinstance(item.get("encrypted_content"), str) and item["encrypted_content"]
    )


def signed_thinking_replay_tokens(msg: Any, readable_tokens: int) -> int:
    """Billed input cost of an Anthropic-signed thinking carrier a kept-thinking route replays: the
    provider-reported count when stamped (the server re-expands the FULL thinking from the
    signature), else the readable text's estimate (summary, or 0 for ``redacted_thinking``)."""
    return stamped_reasoning_tokens(msg) or readable_tokens
