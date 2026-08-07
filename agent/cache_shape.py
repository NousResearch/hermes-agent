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

This module turns that guessing into data.  For each API attempt the loop
captures a :class:`PrefixShape` — content hashes of the system prompt, the
serialized tool schemas, and each conversation message, plus the backend the
request is routed to.  The capture happens once the request payload is
final (after reasoning re-application, prompt-cache re-decoration, transport
preflight, and request middleware), so the fingerprint describes the bytes
the provider actually received rather than an earlier draft.  When the
provider then reports a poor cache hit rate, :func:`diagnose_cache_miss`
compares the previous attempt's shape against the current one and names
exactly what changed — or, when the request moved to a different backend,
says so instead of blaming a cache the new backend never held.

It is observability only: nothing here mutates the request, so the
"prompt caching is sacred" invariant is untouched.
"""

from dataclasses import dataclass, replace
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
    """Fingerprint of one API request's cache-relevant prefix components.

    ``scope`` identifies the backend the prefix was sent to.  Prompt caches
    live per provider/model/endpoint, so two shapes are only comparable when
    their scopes match — see :func:`diagnose_cache_miss`.
    """

    system_hash: str
    tools_hash: str
    message_hashes: Tuple[str, ...]
    tool_count: int
    scope: str = ""


def cache_scope(provider: Any, model: Any, base_url: Any) -> str:
    """Identity of the backend serving (and keying) the prompt cache."""
    return "|".join(
        str(part or "").strip().lower() for part in (provider, model, base_url)
    )


def capture_prefix_shape(
    api_messages: Sequence[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    *,
    scope: str = "",
) -> PrefixShape:
    """Fingerprint the request the agent is about to send.

    ``api_messages`` is the final per-call message list (system message
    first when present); ``tools`` is the schema list passed to the
    provider.  Hashes cover the full message dicts, so tool_calls,
    reasoning echo-back, and cache_control markers all participate in
    change detection — anything that alters wire bytes alters the hash.
    """
    system_hash = ""
    body = [msg for msg in api_messages if isinstance(msg, dict)]
    if body and body[0].get("role") == "system":
        system_hash = _stable_hash(body[0])
        body = body[1:]
    return PrefixShape(
        system_hash=system_hash,
        tools_hash=_stable_hash(tools) if tools else "",
        message_hashes=tuple(_stable_hash(msg) for msg in body),
        tool_count=len(tools or []),
        scope=scope,
    )


def capture_request_shape(
    api_kwargs: Dict[str, Any],
    *,
    provider: Any = "",
    model: Any = "",
    base_url: Any = "",
) -> PrefixShape:
    """Fingerprint the *effective* request payload.

    Takes the finalized ``api_kwargs`` — after provider-specific reasoning
    re-application, prompt-cache re-decoration, ``_build_api_kwargs``,
    transport preflight, and request middleware — so the diagnosis describes
    the bytes the provider actually received rather than an earlier draft of
    them.

    Handles both wire shapes: Chat Completions (``messages``, with the
    system prompt as the leading message) and Codex Responses
    (``input`` plus a top-level ``instructions`` string).
    """
    messages = api_kwargs.get("messages")
    if not isinstance(messages, list):
        messages = api_kwargs.get("input")
    if not isinstance(messages, list):
        messages = []
    tools = api_kwargs.get("tools")
    if not isinstance(tools, list):
        tools = None

    shape = capture_prefix_shape(
        messages, tools, scope=cache_scope(provider, model, base_url),
    )
    # Codex Responses carries the system prompt outside the message list;
    # fold it into system_hash so a changed instruction block is still
    # attributed to the system prompt rather than showing up as nothing.
    instructions = api_kwargs.get("instructions")
    if not shape.system_hash and isinstance(instructions, str) and instructions:
        shape = replace(shape, system_hash=_stable_hash(instructions))
    return shape


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

    - The request went to a different backend than the previous one → the
      cold prefix is expected, because prompt caches are keyed per
      provider/model/endpoint.
    - Hit rate below :data:`LOW_HIT_RATE_PCT` AND the prefix shape changed
      → name the changed component(s).
    - Zero cache hits despite a stable, append-only prefix **on the same
      backend** → the miss is on the provider side (cache TTL/eviction),
      which is worth knowing because no client-side tuning can fix it.

    A shape change on a high-hit-rate turn, or a merely *partial* hit with a
    stable prefix (normal append-only growth), returns None so healthy turns
    stay quiet.
    """
    if prev is None or cur is None or prompt_tokens <= 0:
        return None
    hit_pct = cache_read_tokens / prompt_tokens * 100.0

    # Routing changed (fallback activation, /model switch, credential-pool
    # rotation to a different endpoint). The previous shape describes a
    # different cache namespace entirely, so neither a prefix comparison nor
    # a provider-side TTL conclusion would be sound — the cold read is just
    # what switching backends costs.
    if prev.scope != cur.scope:
        if hit_pct < LOW_HIT_RATE_PCT:
            return (
                "request routed to a different backend since the last call — "
                "prompt caches are per provider/model/endpoint, so this "
                "prefix starts cold"
            )
        return None

    changes = prefix_changes(prev, cur)
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
