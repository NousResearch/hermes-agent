"""Anthropic prompt caching strategy.

The default layout uses 4 cache_control breakpoints: the static system
prefix, the end of the system prompt, and the last 2 non-system messages.
When a static system prefix is unavailable, it falls back to one system
breakpoint plus the last 3 messages. All markers use the same TTL (5m or 1h).
This preserves intra-session caching while allowing new sessions to reuse the
stable system-prompt prefix.

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool" and native_anthropic:
        # Native Anthropic layout: top-level marker; the adapter moves it
        # inside the tool_result block.
        msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        if role == "tool" and not native_anthropic:
            # OpenRouter rejects top-level cache_control on role:tool (silent
            # hang) and an empty message has no content part to carry the
            # marker — skip. Non-empty tool content falls through below and
            # gets the marker on a content part, which OpenRouter honors.
            return
        if role == "assistant" and not native_anthropic:
            # Empty assistant turns are pure tool_calls. A top-level marker
            # here is ignored on the envelope layout, so skip.
            return
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def _can_carry_marker(msg: dict, native_anthropic: bool) -> bool:
    """True if a marker on this message is actually honored by the provider.

    On the native Anthropic layout every message works (top-level markers are
    relocated by the adapter). On the envelope layout (OpenRouter et al.) only
    markers inside content parts are honored: empty-content messages (e.g.
    assistant turns that are pure tool_calls) and empty tool messages would
    receive a top-level marker the provider ignores — wasting one of the four
    breakpoints. Skip those so the breakpoints land on messages that count.
    """
    if native_anthropic:
        return True
    content = msg.get("content")
    if content is None or content == "":
        return False
    if isinstance(content, list):
        # _apply_cache_marker only marks the LAST content part, so the carrier
        # predicate must agree: a list whose last element isn't a dict cannot
        # actually receive a marker and would waste a breakpoint. Mirror the
        # `content` truthiness + last-element-dict check in _apply_cache_marker.
        return bool(content) and isinstance(content[-1], dict)
    return isinstance(content, str)


def _build_marker(ttl: str) -> Dict[str, str]:
    """Build a cache_control marker dict for the given TTL ('5m' or '1h')."""
    marker: Dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def _apply_system_cache_markers(
    message: dict,
    cache_marker: dict,
    static_system_prefix: str | None,
    *,
    native_anthropic: bool,
) -> int:
    """Mark the static system prefix and full prompt when they can be split.

    The system prompt remains one stored string. Splitting it only in the
    outgoing request keeps session persistence and non-Anthropic transports
    unchanged while making the stable prefix independently cacheable.
    """
    content = message.get("content")
    if (
        isinstance(static_system_prefix, str)
        and static_system_prefix
        and isinstance(content, str)
        and content.startswith(static_system_prefix)
    ):
        suffix = content[len(static_system_prefix):]
        if suffix:
            message["content"] = [
                {
                    "type": "text",
                    "text": static_system_prefix,
                    "cache_control": cache_marker,
                },
                {"type": "text", "text": suffix, "cache_control": cache_marker},
            ]
            return 2

    _apply_cache_marker(message, cache_marker, native_anthropic=native_anthropic)
    return 1


def strip_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove ``cache_control`` markers and undo decoration-produced list shapes.

    Used before re-applying decoration after a mid-turn provider failover so
    the mutated, undecorated shape (image shrink / ASCII cleanup / etc.) is
    preserved while markers match the *new* provider's cache policy (#72626).

    Flattening back to a plain string is restricted to the exact shapes
    :func:`apply_anthropic_cache_control` produces from string content —
    a single ``{"type": "text"}`` part, or the two-part ``[static, volatile]``
    system split — so the ``""``-join is provably byte-exact. Organic
    multi-part text (merged user turns, imported transcripts) and parts
    carrying extra keys (``citations`` etc.) keep their structure; only
    per-part markers are removed. Marker removal is copy-on-write on the
    part dicts: content parts may alias the persistent conversation history
    (the per-call copy is shallow), and stripping must never rewrite the
    stored transcript.

    Mutates the top-level message dicts of ``api_messages`` in place and
    returns the same list.
    """
    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        msg.pop("cache_control", None)
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if any(isinstance(part, dict) and "cache_control" in part for part in content):
            content = [
                {k: v for k, v in part.items() if k != "cache_control"}
                if isinstance(part, dict) and "cache_control" in part
                else part
                for part in content
            ]
            msg["content"] = content
        decoration_shape = content and all(
            isinstance(part, dict)
            and part.get("type", "text") == "text"
            and isinstance(part.get("text"), str)
            and set(part.keys()) <= {"type", "text"}
            for part in content
        ) and (
            len(content) == 1
            or (msg.get("role") == "system" and len(content) == 2)
        )
        if decoration_shape:
            msg["content"] = "".join(part["text"] for part in content)
    return api_messages


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    static_system_prefix: str | None = None,
) -> List[Dict[str, Any]]:
    """Apply Anthropic cache-control markers to API messages.

    When ``static_system_prefix`` exactly matches the beginning of a string
    system prompt, it receives an early marker and the full system prompt gets
    a trailing marker. The remaining two markers target the latest cacheable
    non-system messages. Without that prefix, the legacy system-and-3 layout
    is retained.

    Returns:
        Shallow copy of message list with selective deep copies of modified messages.
    """
    if not api_messages:
        return api_messages

    messages = list(api_messages)
    marker = _build_marker(cache_ttl)

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        messages[0] = copy.deepcopy(messages[0])
        breakpoints_used = _apply_system_cache_markers(
            messages[0],
            marker,
            static_system_prefix,
            native_anthropic=native_anthropic,
        )

    remaining = 4 - breakpoints_used
    non_sys = [
        i
        for i in range(len(messages))
        if messages[i].get("role") != "system"
        and _can_carry_marker(messages[i], native_anthropic=native_anthropic)
    ]
    for idx in non_sys[-remaining:]:
        messages[idx] = copy.deepcopy(messages[idx])
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
