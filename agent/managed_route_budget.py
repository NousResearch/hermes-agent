"""Conservative text-only accounting at managed request boundaries.

UTF-8 bytes bound byte-fallback text tokens without a provider/catalog guess.
JSON framing and per-message slack deliberately overcount. Attachments require
route-specific accounting; their encoded URL/base64 length is not a token count.
"""
from __future__ import annotations

import json

from agent.model_selection_store import get_receipt
from agent.model_selection_types import RoutingBlocked


def estimate_text_input(messages, tools=None) -> int:
    if not isinstance(messages, list):
        raise RoutingBlocked("missing_input_estimate", "assembled messages are unavailable")
    for message in messages:
        if not isinstance(message, dict):
            raise RoutingBlocked("schema_invalid", "each message must be an object")
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(part, dict) or part.get("type") not in ("text", "input_text")
                   for part in content):
                raise RoutingBlocked("missing_input_estimate", "multimodal input needs verified route-specific accounting")
        elif content is not None and not isinstance(content, str):
            raise RoutingBlocked("missing_input_estimate", "unsupported message content accounting")
    if not messages and not tools:
        return 0
    try:
        encoded = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False, allow_nan=False)
        return len(encoded.encode("utf-8")) + 16 * len(messages)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RoutingBlocked("schema_invalid", "assembled input is not serializable text") from exc


def enforce_input_budget(home, receipt_id, messages, tools=None, max_tokens=None) -> None:
    decision = get_receipt(home, receipt_id)
    if decision is None:
        raise RoutingBlocked("stale_or_revoked_decision", "input budget receipt is missing")
    capacity = decision["selected"].get("verified_input_budget")
    reserve = decision["requirements"].get("reserve_tokens")
    if type(capacity) is not int or capacity <= 0:
        raise RoutingBlocked("missing_input_estimate", "receipt lacks verified route capacity; reselect before dispatch")
    if type(reserve) is not int or reserve <= 0:
        raise RoutingBlocked("missing_input_estimate", "receipt lacks output/tool-growth reserve; reselect before dispatch")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise RoutingBlocked("schema_invalid", "max_tokens must be a positive integer")
    # Reserve for output AND a tool round even when the caller omitted a limit.
    reserve = max(reserve, (max_tokens or 8192) + 8192)
    if estimate_text_input(messages, tools) + reserve > capacity:
        raise RoutingBlocked("input_too_large", "assembled prompt/context/tools plus output/tool-growth reserve exceed verified capacity; prepare smaller context or start a newly routed attempt")
