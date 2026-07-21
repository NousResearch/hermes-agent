"""Prompt-cache prefix-shape diagnostics (#68489).

Providers that support prompt caching (DeepSeek, OpenAI, Anthropic, Kimi,
Qwen, ...) bill cached prefix tokens at a steep discount, but only when the
request prefix is byte-stable across turns.  Hermes already works hard to
keep that prefix stable (byte-stable system prompt, ``api_content`` replay
sidecars, whitespace/tool-call normalization in ``conversation_loop``), and
already *surfaces* per-call hit rates.  What it could not do is explain a
sudden miss: when the hit rate collapses mid-session, the user has no way to
tell whether the system prompt changed, the toolset changed, history was
rewritten (compaction), or the provider simply evicted the cache.

This module turns that guessing into data.  Before each API call the loop
captures a :class:`PrefixShape` — content hashes of the system message, the
serialized tool schemas, and each conversation message.  When the provider
then reports a poor cache hit rate, :func:`diagnose_cache_miss` compares the
previous call's shape against the current one and names exactly what
changed.  It is observability only: nothing here mutates the request, so the
"prompt caching is sacred" invariant is untouched.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hashlib
import json


# Below this hit rate a shape *change* is reported as the likely cause of
# the miss.  Appending a large tool result legitimately lowers the per-call
# hit rate with a warm cache (the new suffix is uncached), so shape changes
# on high-hit-rate turns are not worth reporting.
LOW_HIT_RATE_PCT = 40.0

# Length of the hex digest kept per component. 12 hex chars (48 bits) is
# plenty for change *detection* within one session and keeps log lines short.
_DIGEST_CHARS = 12


def _stable_hash(value: Any) -> str:
    """Deterministic content hash for a JSON-ish payload fragment."""
    try:
        serialized = json.dumps(
            value, sort_keys=True, ensure_ascii=False, default=str
        )
    except (TypeError, ValueError):
        serialized = repr(value)
    return hashlib.sha256(serialized.encode("utf-8", "replace")).hexdigest()[
        :_DIGEST_CHARS
    ]


@dataclass(frozen=True)
class PrefixShape:
    """Fingerprint of one API request's cache-relevant prefix components."""

    system_hash: str
    tools_hash: str
    message_hashes: Tuple[str, ...]
    tool_count: int


def capture_prefix_shape(
    api_messages: Sequence[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> PrefixShape:
    """Fingerprint the request the agent is about to send.

    ``api_messages`` is the final per-call message list (system message
    first when present); ``tools`` is the schema list passed to the
    provider.  Hashes cover the full message dicts, so tool_calls,
    reasoning echo-back, and cache_control markers all participate in
    change detection — anything that alters wire bytes alters the hash.
    """
    system_hash = ""
    body = list(api_messages)
    if body and body[0].get("role") == "system":
        system_hash = _stable_hash(body[0])
        body = body[1:]
    return PrefixShape(
        system_hash=system_hash,
        tools_hash=_stable_hash(tools) if tools else "",
        message_hashes=tuple(_stable_hash(msg) for msg in body),
        tool_count=len(tools or []),
    )


def prefix_changes(prev: PrefixShape, cur: PrefixShape) -> List[str]:
    """Name every prefix component that changed between two requests.

    Returns an empty list when the current request is a pure append-only
    extension of the previous one (the cache-friendly case).
    """
    changes: List[str] = []
    if prev.system_hash != cur.system_hash:
        changes.append("system prompt changed")
    if prev.tools_hash != cur.tools_hash:
        if prev.tool_count != cur.tool_count:
            changes.append(
                f"tool schemas changed ({prev.tool_count} → {cur.tool_count} tools)"
            )
        else:
            changes.append("tool schemas changed")

    prev_msgs, cur_msgs = prev.message_hashes, cur.message_hashes
    common = min(len(prev_msgs), len(cur_msgs))
    divergence = next(
        (i for i in range(common) if prev_msgs[i] != cur_msgs[i]), None
    )
    if divergence is not None:
        changes.append(
            "conversation history rewritten at message "
            f"#{divergence + 1} of {len(cur_msgs)} (compaction or edit)"
        )
    elif len(cur_msgs) < len(prev_msgs):
        changes.append(
            f"conversation history shrank ({len(prev_msgs)} → {len(cur_msgs)} "
            "messages; compaction or truncation)"
        )
    return changes


def diagnose_cache_miss(
    prev: Optional[PrefixShape],
    cur: Optional[PrefixShape],
    *,
    cache_read_tokens: int,
    prompt_tokens: int,
) -> Optional[str]:
    """Explain a poor cache hit rate, or return None when there is nothing
    interesting to say.

    Reported cases:

    - Hit rate below :data:`LOW_HIT_RATE_PCT` AND the prefix shape changed
      → name the changed component(s).
    - Zero cache hits despite a stable, append-only prefix → the miss is on
      the provider side (cache TTL/eviction), which is worth knowing because
      no client-side tuning can fix it.

    A shape change on a high-hit-rate turn, or a merely *partial* hit with a
    stable prefix (normal append-only growth), returns None so healthy turns
    stay quiet.
    """
    if prev is None or cur is None or prompt_tokens <= 0:
        return None
    changes = prefix_changes(prev, cur)
    hit_pct = cache_read_tokens / prompt_tokens * 100.0
    if changes:
        if hit_pct < LOW_HIT_RATE_PCT:
            return "; ".join(changes)
        return None
    if cache_read_tokens == 0:
        return (
            "request prefix unchanged and append-only — the miss is "
            "provider-side (cache TTL or eviction)"
        )
    return None
