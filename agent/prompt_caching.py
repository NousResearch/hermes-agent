"""Anthropic prompt caching strategy — pure functions, no AIAgent dependency.

Default layout: 4 cache_control breakpoints — the static system prefix, the end of the
system prompt, and the last 2 non-system messages (without a static prefix: one system
breakpoint plus the last 3 messages). All markers share one TTL (5m or 1h).
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


def envelope_tool_part_cache_markers_supported(provider: str | None, base_url: str | None) -> bool:
    """Whether the envelope-layout route honors part-level markers on role:tool.

    OpenRouter/Nous Portal relocate a part-level ``cache_control`` onto the ``tool_result``
    block; LiteLLM-style proxies copy parts verbatim, so it lands at ``tool_result.content[0]``
    (non-retryable 400). There, role:tool carries no part markers and the budget reallocates.
    """
    from agent.agent_runtime_helpers import _is_litellm_route

    return not _is_litellm_route((provider or "").strip().lower(), base_url or "")


def envelope_tool_part_cache_markers_supported(
    provider: str | None, base_url: str | None
) -> bool:
    """Whether the envelope-layout route honors part-level markers on role:tool.

    OpenRouter (and Nous Portal, which proxies to it) relocate a
    ``cache_control`` sitting on a tool message's content part onto the
    ``tool_result`` block during their OpenAI→Anthropic translation, so the
    marker is honored there. LiteLLM-style OpenAI-wire proxies instead map
    content parts verbatim: the part-level marker lands at
    ``tool_result.content[0]``, which the Anthropic Messages schema forbids —
    a non-retryable HTTP 400 that kills the whole turn (#89886). On those
    routes tool messages must not carry part-level markers at all; the
    breakpoint budget reallocates to the nearest eligible message instead.
    """
    from agent.agent_runtime_helpers import _is_litellm_route

    return not _is_litellm_route((provider or "").strip().lower(), base_url or "")


def _apply_cache_marker(
    msg: dict,
    cache_marker: dict,
    native_anthropic: bool = False,
    tool_part_markers: bool = True,
) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool" and not native_anthropic and not tool_part_markers:
        # LiteLLM-style envelope: a part marker → tool_result.content[0] → non-retryable 400.
        return

    if role == "tool" and not tool_part_markers:
        # Envelope route whose OpenAI→Anthropic translation copies content
        # parts verbatim (LiteLLM et al.): a part-level marker becomes
        # tool_result.content[0].cache_control → non-retryable 400 (#89886).
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
                suffix = content[len(stable_prefix):]
                if suffix.strip():
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
                        {"type": "text", "text": suffix},
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


def _can_carry_marker(
    msg: dict, native_anthropic: bool, tool_part_markers: bool = True
) -> bool:
    """True if a marker on this message is actually honored by the provider.

    On the native Anthropic layout every message works (top-level markers are
    relocated by the adapter). On the envelope layout (OpenRouter et al.) only
    markers inside content parts are honored: empty-content messages (e.g.
    assistant turns that are pure tool_calls) and empty tool messages would
    receive a top-level marker the provider ignores — wasting one of the four
    breakpoints. Skip those so the breakpoints land on messages that count.

    ``tool_part_markers=False`` (LiteLLM-style envelope routes, #89886)
    additionally excludes ALL role:tool messages: their part-level marker
    would be forwarded verbatim into ``tool_result.content[]`` and rejected
    with a non-retryable 400, so the breakpoint must reallocate instead.
    """
    if native_anthropic:
        return True
    if msg.get("role") == "tool" and not tool_part_markers:
        return False
    content = msg.get("content")
    if content is None or content == "":
        return False
    content = msg.get("content")
    return isinstance(content[-1], dict) if isinstance(content, list) and content else isinstance(content, str) and content != ""


def _build_marker(ttl: str) -> Dict[str, str]:
    """Build a cache_control marker dict for the given TTL ('5m' or '1h')."""
    return {"type": "ephemeral", "ttl": "1h"} if ttl == "1h" else {"type": "ephemeral"}


# Alibaba-family providers (Qwen routes): five-minute context cache, 1h tier rejected. Shared
# with agent_runtime_helpers.anthropic_prompt_cache_policy so the opt-in and the TTL clamp
# never desync. Do NOT narrow this set to extend a TTL — narrowing DISABLES caching.
ALIBABA_FAMILY_PROVIDERS = frozenset({"opencode", "opencode-go", "opencode-zen", "alibaba"})

# 1h-tier ALLOW-list: only routes wire-measured to retain a 1h marker. Other opencode routes
# are UNMEASURED, not known-bad (opencode-go's `ephemeral_5m_input_tokens` label is not
# evidence of the retention window).
MEASURED_1H_PROVIDERS = frozenset({"opencode-go"})

# Models measured to ignore the 1h tier on a MEASURED_1H_PROVIDERS route; consulted only
# there (the same model on its own endpoint is a separate route).
NO_1H_TIER_MODELS = frozenset({"minimax-m2.5"})


def _flat_model(model: str) -> str:
    """Bare model id, tolerating aggregator prefixes (``vendor/model``)."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


# --- 1h-tier membership: an ALLOW-list, deliberately minimal ----------------
#
# #84733 clamped 1h -> 5m for the whole alibaba/opencode family, reasoning from
# Alibaba's PUBLISHED Qwen docs. Wire measurement on the opencode-go route
# contradicts the docs. Controlled run: identical request, only the ttl flag
# varying, read back after 11 minutes with no intervening call (a read renews
# the window and would mask expiry):
#
#   qwen3.8-max   ttl=1h -> cache_read 2122  SURVIVED
#   qwen3.8-max   ttl=-  -> cache_read    0  EXPIRED    <- control
#   glm-5.2       ttl=1h -> cache_read 2092  SURVIVED
#   minimax-m2.5  ttl=1h -> cache_read    0  EXPIRED
#
# Read the two non-qwen rows for what they are: evidence about the ROUTE, not
# about traffic Hermes sends today. anthropic_prompt_cache_policy currently
# opts opencode-go in only for qwen models, so glm-5.2 and minimax-m2.5 on
# that route receive no cache_control marker at all and never reach this
# clamp in production. They constrain the route-level rule; they are not
# live paths.
#
# Only opencode-go is listed: it is the only route measured. Other opencode
# routes stay clamped because they were NOT measured, not because they are
# known bad. opencode-zen returns cache_creation.ephemeral_1h_input_tokens for
# Claude models, so it is a candidate -- but qwen on zen is unmeasured, so
# adding the provider wholesale would outrun the evidence.
#
# WARNING: opencode-go labels EVERY write `ephemeral_5m_input_tokens` whatever
# ttl was requested. That label is NOT evidence of the retention window -- it
# is what made the original docs-based reasoning look confirmed. Verify only
# with a delayed read past 5 minutes and no intervening call.
#
# NOTE: kept separate from ALIBABA_FAMILY_PROVIDERS on purpose. That set also
# drives the cache-marker-layout OPT-IN in
# agent_runtime_helpers.anthropic_prompt_cache_policy; narrowing it would
# silently DISABLE caching for qwen on opencode-go rather than extend its TTL.
MEASURED_1H_PROVIDERS = frozenset({
    "opencode-go",
})

# Models measured to ignore the 1h tier even on a 1h-capable route.
#
# SCOPE: consulted only for providers already in MEASURED_1H_PROVIDERS. The
# measurement was taken on the opencode-go route, so it says nothing about the
# same model reached some other way -- and MiniMax on its own
# Anthropic-compatible endpoint IS a separate, cache-eligible route
# (anthropic_prompt_cache_policy opts it in by provider id / host match).
# Checking this set globally would have silently regressed that unrelated
# route's configured 1h to 5m off the back of an opencode-go observation.
NO_1H_TIER_MODELS = frozenset({
    "minimax-m2.5",
})


def _flat_model(model: str) -> str:
    """Bare model id, tolerating aggregator prefixes (``vendor/model``)."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def is_qwen_model(model: str) -> bool:
    """True when ``model`` names a Qwen-family model (shared with anthropic_prompt_cache_policy)."""
    return "qwen" in (model or "").lower()


def effective_cache_ttl(ttl: str | None, *, model: str = "", provider: str = "") -> str:
    """Clamp a requested cache TTL to what the destination route supports (``None`` → ``5m``).

    Qwen/Alibaba context caching documents an explicit five-minute window
    (renewed on hit); the Anthropic ``1h`` tier is ignored/rejected there,
    so a configured ``1h`` regresses to ``5m`` instead of shipping a marker
    the provider drops and creating a false 1h-cache expectation (#84733).
    Exception: routes in ``MEASURED_1H_PROVIDERS`` were wire-measured to
    honour the tier (delayed read past 5 minutes) and keep ``1h`` — minus
    any model in ``NO_1H_TIER_MODELS`` measured to ignore it on that route.
    All other caching routes keep the requested TTL.

    ``None`` (caching active with no explicit tier) resolves to ``5m``.
    """
    if ttl != "1h":
        return ttl or "5m"
    if (provider or "").lower() in MEASURED_1H_PROVIDERS:
        # Route measured to honour the tier -- checked BEFORE the generic
        # is_qwen_model clamp below, which would otherwise swallow every Qwen
        # model on it. Within the route, a model measured to ignore the tier
        # still wins; the denial stays nested here so an opencode-go
        # observation cannot leak out and reclamp the same model on an
        # unrelated route.
        return "5m" if _flat_model(model) in NO_1H_TIER_MODELS else "1h"
    if is_qwen_model(model):
        return "5m"
    if (provider or "").lower() in ALIBABA_FAMILY_PROVIDERS:
        return "5m"
    return "1h"


def _apply_system_cache_markers(
    message: dict, cache_marker: dict, static_system_prefix: str | None, *,
    native_anthropic: bool, mark_suffix: bool = True, fallback_to_whole: bool = True,
) -> int:
    """Mark the static system prefix (and optionally the full prompt); returns markers applied.

    The stored system prompt stays one string, split only in the request. ``mark_suffix=False``
    is the tool-cache-plan layout (suffix budget spent on the tools array); ``fallback_to_whole=
    False`` marks nothing when the split is impossible. When the prompt IS the prefix the whole
    message is one block — never an empty text block (400).
    """
    content = message.get("content")
    if isinstance(static_system_prefix, str) and static_system_prefix and isinstance(content, str) and content.startswith(static_system_prefix):
        suffix = content[len(static_system_prefix):]
        if suffix.strip():
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
        # Empty/whitespace-only suffix: the stored prompt IS the static prefix. Mark it as
        # one whole block — a [marked-prefix, ""] split would put an empty
        # text block on the wire (HTTP 400 on native Anthropic).
        _apply_cache_marker(message, cache_marker, native_anthropic=native_anthropic)
        return 1

    if not fallback_to_whole:
        return 0
    _apply_cache_marker(message, cache_marker, native_anthropic=native_anthropic)
    return 1


def _has_part_marker(content: Any) -> bool:
    return isinstance(content, list) and any(isinstance(part, dict) and "cache_control" in part for part in content)


def strip_anthropic_cache_control(api_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove ``cache_control`` markers and undo decoration-produced list shapes (in place).

    Used before re-decorating after a mid-turn failover. Flattening to a string is restricted
    to the exact shapes :func:`apply_anthropic_cache_control` produces from string content
    (single text part, two-part system split, two-part skill split) so the ``""``-join is
    byte-exact. Marker removal is copy-on-write on part dicts: parts can alias caller-held
    lists and stripping must never rewrite the stored transcript.
    """
    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        msg.pop("cache_control", None)
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        role = msg.get("role")
        # The skill split is the only decoration marking the FIRST part of a user message,
        # so the shape alone identifies it even after the prefix registry evicted the entry.
        skill_split_shape = (role == "user" and len(content) == 2 and all(isinstance(p, dict) for p in content)
                             and "cache_control" in content[0] and "cache_control" not in content[1])
        if _has_part_marker(content):
            content = msg["content"] = [
                {k: v for k, v in part.items() if k != "cache_control"}
                if isinstance(part, dict) and "cache_control" in part else part for part in content
            ]
        plain_text_parts = content and all(
            isinstance(part, dict) and part.get("type", "text") == "text"
            and isinstance(part.get("text"), str) and set(part.keys()) <= {"type", "text"} for part in content
        )
        if plain_text_parts and (len(content) == 1 or (role == "system" and len(content) == 2) or skill_split_shape):
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
    parts = [p for m in messages if isinstance(m, dict) and isinstance(m.get("content"), list) for p in m["content"]]
    return sum(1 for item in [*messages, *parts, *tools] if isinstance(item, dict) and "cache_control" in item)


def _completed_transaction_endpoint_indexes(messages: List[Dict[str, Any]], *, native_anthropic: bool) -> List[int]:
    """Select legal ends of completed tool runs and ordinary turns."""

    def _tool_run_end(start: int) -> int:
        end = start
        while end < len(messages) and isinstance(messages[end], dict) and messages[end].get("role") == "tool":
            end += 1
        return end

    endpoints: List[int] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") == "system":
            index += 1
            continue
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            result_end = _tool_run_end(index + 1)
            if result_end > index + 1 and _can_carry_marker(messages[result_end - 1], native_anthropic):
                endpoints.append(result_end - 1)
            index = result_end
            continue

        if role == "tool":
            index = _tool_run_end(index)
            continue

        open_turn = (role == "user" and index + 1 < len(messages)) or (
            role == "assistant" and message.get("content") in (None, ""))
        if not open_turn and _can_carry_marker(message, native_anthropic):
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
    tool_part_markers: bool = True,
) -> PromptCachePlan:
    """Build isolated cache sections for one resolved request destination.

    ``tool_part_markers=False`` (LiteLLM-style envelope routes, #89886)
    keeps ``cache_control`` off role:tool content parts; breakpoints
    reallocate to the nearest eligible non-tool message.
    """
    messages = copy.deepcopy(api_messages or [])
    strip_anthropic_cache_control(messages)
    planned_tools = strip_anthropic_tool_cache_control(tools)

    if not direct_native_tool_cache or not planned_tools:
        planned_messages = apply_anthropic_cache_control(
            messages,
            cache_ttl=cache_ttl,
            native_anthropic=native_anthropic,
            static_system_prefix=static_system_prefix,
            tool_part_markers=tool_part_markers,
        )
        return PromptCachePlan(messages=planned_messages, tools=planned_tools)

    marker = _build_marker(cache_ttl)
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        # Tool-cache layout: only the static prefix carries a system-side marker; the
        # volatile suffix's budget is spent on the tools array.
        _apply_system_cache_markers(messages[0], marker, static_system_prefix,
                                    native_anthropic=True, mark_suffix=False, fallback_to_whole=False)
    planned_tools[-1]["cache_control"] = dict(marker)
    for endpoint in _completed_transaction_endpoint_indexes(messages, native_anthropic=True)[-2:]:
        _apply_cache_marker(messages[endpoint], marker, native_anthropic=True)

    return PromptCachePlan(messages=messages, tools=planned_tools)


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    static_system_prefix: str | None = None,
    tool_part_markers: bool = True,
) -> List[Dict[str, Any]]:
    """Apply Anthropic cache-control markers to API messages.

    With a matching ``static_system_prefix`` the prefix and full system prompt each get a
    marker and the remaining two go to the latest cacheable non-system messages; otherwise
    the legacy system-and-3 layout applies. Idempotent: pre-existing markers are stripped from
    a per-message copy first. Returns a shallow list copy with deep copies of modified messages.

    Idempotent: pre-existing ``cache_control`` markers are stripped from a
    per-message copy before new ones are placed, so calling this twice (or
    handing it messages a prior call already marked) can never accumulate
    past 4 markers. Only messages that already carry a marker pay the copy
    cost — a shallow top-level copy suffices because
    :func:`strip_anthropic_cache_control` is copy-on-write on content parts —
    and the rest of the copy-on-write contract is unchanged (#90971).

    ``tool_part_markers=False`` (LiteLLM-style envelope routes, #89886)
    keeps markers off role:tool messages entirely; the breakpoint budget
    reallocates to the nearest eligible non-tool message.

    Returns:
        Shallow copy of message list with selective deep copies of modified messages.
    """
    if not api_messages:
        return api_messages

    messages = list(api_messages)
    marker = _build_marker(cache_ttl)

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        has_marker = "cache_control" in msg or (
            isinstance(content, list)
            and any(isinstance(part, dict) and "cache_control" in part for part in content)
        )
        if has_marker:
            # Shallow top-level copy is enough: strip pops the top-level key
            # and rebuilds content lists/part dicts copy-on-write, so the
            # caller's message (and any aliased parts) are never mutated.
            messages[i] = strip_anthropic_cache_control([dict(msg)])[0]

    breakpoints_used = 0

    breakpoints_used = 0
    if messages[0].get("role") == "system":
        messages[0] = copy.deepcopy(messages[0])
        breakpoints_used = _apply_system_cache_markers(messages[0], marker, static_system_prefix,
                                                       native_anthropic=native_anthropic)

    remaining = 4 - breakpoints_used
    non_sys = [
        i
        for i in range(len(messages))
        if messages[i].get("role") != "system"
        and _can_carry_marker(
            messages[i],
            native_anthropic=native_anthropic,
            tool_part_markers=tool_part_markers,
        )
    ]
    for idx in non_sys[-remaining:]:
        messages[idx] = copy.deepcopy(messages[idx])
        _apply_cache_marker(
            messages[idx],
            marker,
            native_anthropic=native_anthropic,
            tool_part_markers=tool_part_markers,
        )

    return messages
