"""Gemini long decode: one response spread over several generateContent hops.

Gemini 4 caps each request (32,768 tokens unary, 65,536 streamed) but lets ``maxOutputTokens`` reach
1,048,576 minus the input. A hop that hits the per-request cap with budget left ends with
``finishReason: CONTINUATION`` plus a ``continuationToken``; the client resends the SAME request with
that token (never the partial output) until any other finish reason. Each hop re-reads the prompt plus
everything generated so far, so per-hop ``promptTokenCount`` grows: usage is summed for billing (what
the google-genai SDK reports and Google bills) while the first hop's prompt is kept as the context size,
because compaction fed the summed prompt would fire far too early. A dropped stream resumes from its
last checkpoint token rather than restarting (Google bills a dropped stream up to that checkpoint).
Docs: https://ai.google.dev/gemini-api/docs/generate-content/long-decode
"""

from __future__ import annotations

from typing import Any, Optional

# Effort above ``high`` (Gemini 4's top thinkingLevel) buys a larger total thinking+output budget.
# 262,144 is Google's starting point for hard reasoning; the API caps 1,048,576 at the window minus input.
_LONG_DECODE_BUDGETS = {"xhigh": 262_144, "max": 1_048_576, "ultra": 1_048_576}
_USAGE_COUNT_KEYS = (
    "promptTokenCount", "cachedContentTokenCount", "candidatesTokenCount", "thoughtsTokenCount",
    "toolUsePromptTokenCount", "totalTokenCount",
)


def long_decode_max_output_tokens(model: str, reasoning_config: Any) -> Optional[int]:
    """Total output budget a Gemini 4 request gets for its reasoning effort, or None (per-request cap)."""
    name = (model or "").strip().lower().removeprefix("google/")
    if not name.startswith("gemini-4") or not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
        return None
    return _LONG_DECODE_BUDGETS.get(str(reasoning_config.get("effort") or "").strip().lower())


def _candidate(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    first = candidates[0] if isinstance(candidates, list) and candidates else None
    return first if isinstance(first, dict) else {}


def continuation_token(payload: dict[str, Any]) -> Optional[str]:
    """The token to resend with when this hop ended on CONTINUATION; None on any other finish."""
    cand = _candidate(payload)
    token = cand.get("continuationToken")
    return token if str(cand.get("finishReason") or "").upper() == "CONTINUATION" and isinstance(token, str) and token else None


def merge_hop_usage(hops: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum every count across hops; ``initialPromptTokenCount`` (the google-genai name) is hop 1's prompt."""
    merged = {key: sum(int(h.get(key) or 0) for h in hops) for key in _USAGE_COUNT_KEYS if any(key in h for h in hops)}
    merged["initialPromptTokenCount"] = int(hops[0].get("promptTokenCount") or 0) if hops else 0
    return merged


def merge_hop_responses(hops: list[dict[str, Any]]) -> dict[str, Any]:
    """One generateContent response whose model turn joins every hop's parts, ending as the last hop did."""
    parts = [part for hop in hops for part in ((_candidate(hop).get("content") or {}).get("parts") or [])]
    last = _candidate(hops[-1])
    candidate = {**last, "content": {**(last.get("content") or {}), "role": "model", "parts": parts}}
    usage = merge_hop_usage([hop.get("usageMetadata") or {} for hop in hops])
    return {**hops[-1], "candidates": [candidate], "usageMetadata": usage}


def split_stream_continuation(event: dict[str, Any], hop_usage: list[dict[str, Any]]) -> tuple[dict[str, Any], Optional[str]]:
    """``(event to translate, token to resend with if the stream ends here)`` for one SSE event.

    A CONTINUATION finish is not the end: its finish reason and (per-stream cumulative) usage are
    held back in ``hop_usage`` and the token returned for the next hop. The real final event gets
    the usage summed over every hop. A token on an event with no finish is a mid-stream checkpoint
    (every ~5K tokens past 32K): it commits that event and everything before it, so the caller resumes
    a dropped stream by resending the request with it; a drop with no clean checkpoint restarts.
    """
    token = continuation_token(event)
    if token:
        hop_usage.append(event.get("usageMetadata") or {})
        cand = {k: v for k, v in _candidate(event).items() if k not in ("finishReason", "continuationToken")}
        return {k: v for k, v in event.items() if k != "usageMetadata"} | {"candidates": [cand]}, token
    cand = _candidate(event)
    if hop_usage and cand.get("finishReason"):
        return {**event, "usageMetadata": merge_hop_usage([*hop_usage, event.get("usageMetadata") or {}])}, None
    checkpoint = None if cand.get("finishReason") else cand.get("continuationToken")
    return event, checkpoint if isinstance(checkpoint, str) and checkpoint else None
