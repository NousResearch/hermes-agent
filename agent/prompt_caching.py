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
from dataclasses import dataclass
from typing import Any, Dict, List

from agent.prompt_cache_boundary import find_stable_prefix


@dataclass(frozen=True)
class PromptCachePlan:
    """Request-local message and tool sections with their cache markers."""

    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]

    @property
    def marker_count(self) -> int:
        """Wire-visible cache markers in this plan (computed on demand).

        Only tests consume this; keeping it lazy avoids walking every
        message part and tool schema on the per-request hot path.
        """
        return _count_cache_markers(self.messages, self.tools)


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
        if role == "user":
            stable_prefix = find_stable_prefix(content)
            if stable_prefix is not None:
                # Builder-declared boundary (#81867): the scaffold carries the
                # breakpoint, the volatile invocation tail rides unmarked so a
                # changed ticket ID or timestamp no longer invalidates the
                # whole skill body. Request-local only — the canonical session
                # message stays a plain string.
                msg["content"] = [
                    {
                        "type": "text",
                        "text": stable_prefix,
                        "cache_control": cache_marker,
                    },
                    {"type": "text", "text": content[len(stable_prefix):]},
                ]
                return
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
    mark_suffix: bool = True,
    fallback_to_whole: bool = True,
) -> int:
    """Mark the static system prefix (and optionally the full prompt).

    The system prompt remains one stored string. Splitting it only in the
    outgoing request keeps session persistence and non-Anthropic transports
    unchanged while making the stable prefix independently cacheable.

    ``mark_suffix=False`` is the tool-cache-plan layout: only the static
    prefix carries a marker, the volatile suffix rides unmarked (its
    breakpoint budget is spent on the tools array instead).

    ``fallback_to_whole=False`` skips marking entirely when the prefix
    split is not possible (no prefix, mismatched prefix, non-string
    content) instead of marking the whole message.

    When the prompt IS exactly the static prefix (empty suffix), the whole
    message is marked as a single block — never a two-part split with an
    empty text block, which Anthropic rejects.

    Returns the number of markers applied (0, 1, or 2).
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
            suffix_part: dict = {"type": "text", "text": suffix}
            if mark_suffix:
                suffix_part["cache_control"] = cache_marker
            message["content"] = [
                {
                    "type": "text",
                    "text": static_system_prefix,
                    "cache_control": cache_marker,
                },
                suffix_part,
            ]
            return 2 if mark_suffix else 1
        # Empty suffix: the stored prompt IS the static prefix. Mark it as
        # one whole block — a [marked-prefix, ""] split would put an empty
        # text block on the wire (HTTP 400 on native Anthropic).
        _apply_cache_marker(message, cache_marker, native_anthropic=native_anthropic)
        return 1

    if not fallback_to_whole:
        return 0
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
    a single ``{"type": "text"}`` part, the two-part ``[static, volatile]``
    system split, or the two-part builder-declared skill split (recognised
    by its marker-on-the-first-part shape, so flattening never depends on
    the prefix registry still holding the entry) — so the ``""``-join is
    provably byte-exact. Organic
    multi-part text (merged user turns, imported transcripts) and parts
    carrying extra keys (``citations`` etc.) keep their structure; only
    per-part markers are removed. Marker removal is copy-on-write on the
    part dicts: content parts can alias caller-held message lists (the main
    send path now hands structurally-cloned copies via
    _clone_message_for_send, but other callers may pass shallow copies),
    and stripping must never rewrite the stored transcript.

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
        # Two-part skill-invocation split (#81867). The builder-declared
        # boundary is the only decoration that marks the *first* part of a
        # user message: list content otherwise receives its marker on the
        # last part, and the two-part [static, volatile] split is role-gated
        # to system. So the shape alone identifies it, and flattening stays
        # correct even when the prefix registry has since evicted the entry
        # (failover re-decorates a request built many messages ago, #72626).
        skill_split_shape = (
            msg.get("role") == "user"
            and len(content) == 2
            and isinstance(content[0], dict)
            and isinstance(content[1], dict)
            and "cache_control" in content[0]
            and "cache_control" not in content[1]
        )
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
            or skill_split_shape
        )
        if decoration_shape:
            msg["content"] = "".join(part["text"] for part in content)
    return api_messages


def strip_anthropic_tool_cache_control(tools: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Return copied tools without request-local Anthropic cache markers."""
    cleaned = copy.deepcopy(tools or [])
    for tool in cleaned:
        if isinstance(tool, dict):
            tool.pop("cache_control", None)
    return cleaned


def _count_cache_markers(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> int:
    """Count the wire-visible cache markers in a request-local plan."""
    count = sum(
        1
        for message in messages
        if isinstance(message, dict) and "cache_control" in message
    )
    count += sum(
        1
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and "cache_control" in part
    )
    return count + sum(
        1 for tool in tools if isinstance(tool, dict) and "cache_control" in tool
    )


def _completed_transaction_endpoint_indexes(
    messages: List[Dict[str, Any]], *, native_anthropic: bool,
) -> List[int]:
    """Select legal ends of completed tool runs and ordinary turns."""
    endpoints: List[int] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") == "system":
            index += 1
            continue

        if message.get("role") == "assistant" and message.get("tool_calls"):
            result_start = index + 1
            result_end = result_start
            while result_end < len(messages):
                result = messages[result_end]
                if not isinstance(result, dict) or result.get("role") != "tool":
                    break
                result_end += 1
            if result_end > result_start:
                endpoint = result_end - 1
                if _can_carry_marker(messages[endpoint], native_anthropic):
                    endpoints.append(endpoint)
            index = result_end
            continue

        if message.get("role") == "tool":
            while index < len(messages):
                result = messages[index]
                if not isinstance(result, dict) or result.get("role") != "tool":
                    break
                index += 1
            continue

        if message.get("role") == "user" and index + 1 < len(messages):
            index += 1
            continue

        if (
            message.get("role") == "assistant"
            and message.get("content") in (None, "")
        ):
            index += 1
            continue

        if _can_carry_marker(message, native_anthropic):
            endpoints.append(index)
        index += 1
    return endpoints


def build_prompt_cache_plan(
    api_messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
    *,
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    static_system_prefix: str | None = None,
    direct_native_tool_cache: bool = False,
) -> PromptCachePlan:
    """Build isolated cache sections for one resolved request destination."""
    messages = copy.deepcopy(api_messages or [])
    strip_anthropic_cache_control(messages)
    planned_tools = strip_anthropic_tool_cache_control(tools)

    if not direct_native_tool_cache or not planned_tools:
        planned_messages = apply_anthropic_cache_control(
            messages,
            cache_ttl=cache_ttl,
            native_anthropic=native_anthropic,
            static_system_prefix=static_system_prefix,
        )
        return PromptCachePlan(messages=planned_messages, tools=planned_tools)

    marker = _build_marker(cache_ttl)
    if (
        messages
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    ):
        # Tool-cache layout: only the static prefix carries a system-side
        # marker; the volatile suffix's budget is spent on the tools array.
        _apply_system_cache_markers(
            messages[0],
            marker,
            static_system_prefix,
            native_anthropic=True,
            mark_suffix=False,
            fallback_to_whole=False,
        )
    planned_tools[-1]["cache_control"] = dict(marker)
    for endpoint in _completed_transaction_endpoint_indexes(
        messages,
        native_anthropic=True,
    )[-2:]:
        _apply_cache_marker(messages[endpoint], marker, native_anthropic=True)

    return PromptCachePlan(messages=messages, tools=planned_tools)


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
