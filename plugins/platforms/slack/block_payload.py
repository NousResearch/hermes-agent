"""Agent-facing serialization of *inbound* Block Kit payloads.

The reverse direction — agent markdown rendered *into* blocks — lives in :mod:`block_kit`.
Here the blocks are the sender's, so the projection is an allowlist: ``type``-style scalars are
copied, the keys in ``_BLOCK_RECURSIVE_KEYS`` are recursed into, and everything else (``url`` and
friends) is dropped on purpose.

Nesting is sender-controlled, so the dump walk is depth-bounded (``_BLOCK_MAX_DEPTH``): past the
budget a subtree is dropped rather than walked, and the dump never falls back to serializing the
unfiltered blocks.
"""

from __future__ import annotations

import json
from typing import Any

from agent.compression_marker import elide


# Block Kit keys kept in the agent-facing payload dump (scalars copied; containers recursed).
_BLOCK_SCALAR_KEYS = frozenset(
    "type block_id action_id style dispatch_action optional multiple emoji".split())
_BLOCK_RECURSIVE_KEYS = frozenset(
    "text title description label placeholder accessory fields elements options "
    "option_groups confirm submit close hint".split())
#: Past this nesting depth the dump stops descending. Sender-authored trees only need a
#: handful of levels, and an unbounded walk lets a deep payload raise ``RecursionError``
#: inside ``_sanitize`` — which used to fall through to ``repr(inspectable)``, i.e. the
#: *unfiltered* blocks carrying the ``url`` fields the allowlist exists to strip.
_BLOCK_MAX_DEPTH = 100


def _serialize_slack_blocks_for_agent(blocks: list, max_chars: int = 6000) -> str:
    """Compact, redacted JSON view of non-``rich_text`` Block Kit blocks.
    ``rich_text`` is already rendered into the message text; dumping it here would repeat the
    author's words with every ``url`` stripped by the allowlist. ``table`` is rendered by
    :func:`_render_slack_table_block`; the allowlist drops ``rows`` so it would dump as a husk."""
    inspectable = [
        block for block in (blocks or []) if (block or {}).get("type") not in ("rich_text", "table")]
    if not inspectable:
        return ""
    def _sanitize(value, depth: int = 0):
        if depth > _BLOCK_MAX_DEPTH:
            return None
        if isinstance(value, list):
            return [
                item for item in (_sanitize(v, depth + 1) for v in value) if item not in (None, {}, [], "")]
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if key in _BLOCK_SCALAR_KEYS:
                    sanitized[key] = item
                elif key in _BLOCK_RECURSIVE_KEYS:
                    cleaned = _sanitize(item, depth + 1)
                    if cleaned not in (None, {}, [], ""):
                        sanitized[key] = cleaned
            return sanitized
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    try:
        payload = json.dumps(_sanitize(inspectable), ensure_ascii=False, indent=2)
    except Exception:
        # Never fall back to ``repr(inspectable)``: that is the pre-filter list, so it would
        # hand the agent every field the allowlist above strips.
        payload = "[Slack Block Kit payload unavailable]"
    payload = elide(payload, max_chars)
    return f"[Slack Block Kit payload for this message]\n```json\n{payload}\n```"


def _extract_urls_from_slack_blocks(blocks: list) -> list[str]:
    """Return deduped URLs from a Block Kit tree in discovery order.
    Targeted opt-in for alert links; ``_serialize_slack_blocks_for_agent`` deliberately strips
    ``url`` from the generic payload dump."""
    if not blocks:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("url", "image_url", "external_url"):
                value = node.get(key)
                is_url = isinstance(value, str) and value.startswith(("http://", "https://"))
                if is_url and value not in seen:
                    seen.add(value)
                    found.append(value)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(blocks)
    return found
