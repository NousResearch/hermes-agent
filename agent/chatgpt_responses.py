"""Wire contract for Responses requests authorized by a ChatGPT plan."""

from copy import deepcopy
from typing import Any


_UNSUPPORTED_FIELDS = frozenset({
    "background", "conversation", "max_output_tokens", "max_tool_calls", "metadata",
    "moderation", "multi_agent", "prompt", "prompt_cache_retention", "safety_identifier",
    "temperature", "top_logprobs", "top_p", "truncation", "user", "previous_response_id",
})
_TOOL_NAMESPACE = "hermes"
_TERMINAL_CODES = frozenset({
    "subscription_sharing_user_not_eligible", "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_unsupported_capability", "subscription_sharing_route_not_supported",
    "subscription_sharing_invalid_user", "chatpass_v2_scope_not_authorized",
    "chatpass_v2_invalid_authorization_context",
    "chatgpt_session_changed",
})
_TEMPORARY_CODES = frozenset({
    "subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable",
})


def validate_chatgpt_base_url(base_url: Any) -> None:
    """The direct ChatGPT grant authorizes only the official public API route."""
    if str(base_url or "").rstrip("/") != "https://api.openai.com/v1":
        raise ValueError("ChatGPT plan requests require https://api.openai.com/v1.")


def classify_chatgpt_error(error: Exception, *, status_code=None, error_code=None,
                           message="", body=None, model="") -> dict[str, Any] | None:
    """Keep plan permission/quota refusals out of generic auth and billing recovery.

    The generic rate-limit and auth paths rotate accounts or activate a fallback even
    when retry hints are false. A terminal provider restriction instead stops the turn
    while preserving the selected registration. Transient service errors keep the
    existing bounded server-error retry policy.
    """
    error_code = error_code or getattr(error, "code", None)
    if error_code in _TERMINAL_CODES or status_code in {401, 403}:
        verdict = {"reason": "provider_policy_blocked", "retryable": False,
                   "should_rotate_credential": False, "should_fallback": False}
        if error_code == "subscription_sharing_usage_limit_exceeded":
            verdict["error_context"] = {"user_guidance": (
                "ChatGPT plan usage for this app has reached a limit. Requests have stopped; "
                "this may be an app-specific limit, not your entire plan. "
                "Check usage at https://chatgpt.com/settings/usage before trying again."
            )}
        return verdict
    if error_code in _TEMPORARY_CODES or status_code == 503:
        return {"reason": "server_error", "retryable": True,
                "should_rotate_credential": False, "should_fallback": False}
    return None


def prepare_chatgpt_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply SIWC constraints after middleware and overrides, without changing API-key requests.

    All Hermes functions share a stable namespace. Reconstruct it on replay because the
    internal chat-completions history stores the local function name, not its wire namespace.
    """
    request = deepcopy(kwargs)
    extra = request.pop("extra_body", None) or {}
    # These fields would otherwise override the normalized body during SDK serialization.
    for key in ("input", "instructions", "tools", "tool_choice", "store", "stream"):
        if key in extra:
            request[key] = extra.pop(key)
    for key in _UNSUPPORTED_FIELDS:
        request.pop(key, None)
        extra.pop(key, None)
    request.update(store=False, stream=True)
    if extra:
        request["extra_body"] = extra

    tools, namespaces = _namespace_tools(request.get("tools") or [])
    if "tools" in request:
        request["tools"] = tools
    items = request.get("input")
    if not isinstance(items, list):
        raise ValueError("ChatGPT plan requests require the complete input history as an array.")
    for item in items:
        if item.get("role") == "system":
            item["role"] = "developer"
        if item.get("type") in {"function_call", "custom_tool_call"}:
            item.setdefault("namespace", namespaces.get(item.get("name"), _TOOL_NAMESPACE))
    choice = request.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") in {"function", "custom"}:
        choice.setdefault("namespace", namespaces.get(choice.get("name"), _TOOL_NAMESPACE))
    return request


def _namespace_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: list[dict[str, Any]] = []
    result: list[dict[str, Any]] = []
    namespaces: dict[str, str] = {}
    for tool in tools:
        kind = tool.get("type")
        if kind in {"function", "custom"}:
            grouped.append(tool)
            namespaces[tool["name"]] = _TOOL_NAMESPACE
        elif kind == "namespace":
            result.append(tool)
            for child in tool.get("tools", []):
                if child.get("type") not in {"function", "custom"}:
                    raise ValueError("ChatGPT plan namespaces support only function and custom tools.")
                namespaces[child["name"]] = tool["name"]
        elif kind in {"web_search", "web_search_preview"}:
            result.append(tool)
        else:
            raise ValueError(f"ChatGPT plan does not support the {kind!r} tool.")
    if grouped:
        result.append({"type": "namespace", "name": _TOOL_NAMESPACE,
                       "description": "Hermes agent tools.", "tools": grouped})
    return result, namespaces
