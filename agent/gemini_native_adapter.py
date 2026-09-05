"""OpenAI-compatible facade over Google AI Studio's native Gemini API: the ``gemini``
provider keeps ``api_mode='chat_completions'`` so the agent loop stays OpenAI-shaped,
and this shim converts ``messages[]``/``tools[]`` into ``models/{model}:generateContent``
payloads and responses back. Google's OpenAI-compat endpoint is brittle for the
multi-turn tool loop (auth churn, tool-call replay, thought signatures); native is canonical."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import httpx

from agent.bounded_response import read_streaming_error_body
from agent.gemini_schema import sanitize_gemini_tool_parameters

logger = logging.getLogger(__name__)

try:
    import hermes_cli as _hermes_cli

    _HERMES_VERSION = str(_hermes_cli.__version__)
except Exception:
    _HERMES_VERSION = "0.0.0"
_API_CLIENT = f"hermes-agent/{_HERMES_VERSION}"  # client context per Gemini's partner-integration guidance

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Published max output-token ceiling shared by every current Gemini text model; used
# for max_tokens=None because the native API's low internal default truncates output.
GEMINI_DEFAULT_MAX_OUTPUT_TOKENS = 65535

_FREE_TIER_GUIDANCE = (
    "\n\nYour Google API key is on the free tier (a few hundred requests/day for Gemini Flash models). "
    "Hermes typically makes 3-10 API calls per user turn, so the free tier is exhausted in a handful of "
    "messages and cannot sustain an agent session. Enable billing on your Google Cloud project and "
    "regenerate the key in a billing-enabled project: https://aistudio.google.com/apikey"
)
_STANDARD_KEY_GUIDANCE = (
    "\n\nGoogle Gemini rejected this API key's type — you do NOT need OAuth. Google began rejecting legacy "
    "'Standard' Google Cloud keys for the Gemini API on June 19, 2026, and all Standard keys stop working in "
    "September 2026. Open https://aistudio.google.com/api-keys, check the key's type and status, and create a "
    "replacement Gemini API key (or, as a temporary bridge, restrict the Standard key to "
    "generativelanguage.googleapis.com). Then update GEMINI_API_KEY / GOOGLE_API_KEY in ~/.hermes/.env and "
    "restart your session. Details: https://ai.google.dev/gemini-api/docs/api-key"
)
# Stands in for a model turn that never arrived (stream failure / interrupt / quota
# fallback) when a human user text turn directly follows a tool-result turn, keeping
# the request alternation-valid while the user's message stays a turn of its own
# (mirrors gemini-cli's placeholder repair).
_INTERRUPTED_RESPONSE_PLACEHOLDER = "[The previous response was interrupted before it completed.]"
# Cross-provider tool_calls (e.g. fallback from xAI/Anthropic) carry no Gemini thoughtSignature;
# without this sentinel Gemini 3 thinking models reject replayed history with 400 INVALID_ARGUMENT.
_SKIP_SIGNATURE = "skip_thought_signature_validator"
_END = object()  # stream-exhausted marker for _advance_stream_iterator
_TOOL_CHOICE_MODES = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
_FINISH_REASON_MAP = {
    "STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter", "OTHER": "stop",
}
_HTTP_ERROR_CODES = {401: "gemini_unauthorized", 429: "gemini_rate_limited", 404: "gemini_model_not_found"}
_MISSING_KEY_ERROR = (
    "Gemini native client requires an API key, but none was provided. Set GOOGLE_API_KEY or GEMINI_API_KEY in your "
    "environment / ~/.hermes/.env (get one at https://aistudio.google.com/app/apikey), or run `hermes setup` to "
    "configure the Google provider."
)


def bare_gemini_model_id(model: str) -> str:
    """Strip Gemini's own provider prefix from an aggregator-style model id."""
    name = (model or "").strip()
    for prefix in ("google/", "gemini/"):
        if name.lower().startswith(prefix):
            return name[len(prefix):].strip() or name
    return name


def gemini_requires_tool_call_ids(model: str) -> bool:
    """Gemini 3+ needs explicit functionCall/functionResponse ids so replayed parallel tool calls
    pair with their responses; 2.x rejects the field.

    Gemini 3+ models require explicit tool call IDs in replayed history — without them, multi-tool turns can
    be rejected or mismatched. Older Gemini models (2.x) reject unexpected ``id`` fields, so this is gated
    on the major version. Mirrors earendil-works/pi#7494 (their fix for the same class of bug in the
    google-shared converter).
    """
    match = re.match(r"gemini-(\d+)", bare_gemini_model_id(model).lower())
    return match is not None and int(match.group(1)) >= 3


def is_native_gemini_base_url(base_url: str) -> bool:
    """True when the endpoint speaks Gemini's native REST API (not ``/openai``)."""
    normalized = str(base_url or "").strip().rstrip("/").lower()
    return "generativelanguage.googleapis.com" in normalized and not normalized.endswith("/openai")


def probe_gemini_tier(
    api_key: str, base_url: str = DEFAULT_GEMINI_BASE_URL, *, model: str = "gemini-3.7-flash", timeout: float = 10.0
) -> str:
    """Probe a Google AI Studio key → ``"free"`` | ``"paid"`` | ``"unknown"`` (probe failed; callers proceed without blocking)."""
    key = (api_key or "").strip()
    if not key:
        return "unknown"
    base = str(base_url or DEFAULT_GEMINI_BASE_URL).strip().rstrip("/") or DEFAULT_GEMINI_BASE_URL
    base = re.sub(r"/openai\Z", "", base, flags=re.IGNORECASE)
    payload = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1}}
    headers = {"Content-Type": "application/json", "X-Goog-Api-Client": _API_CLIENT}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base}/models/{model}:generateContent", params={"key": key}, json=payload, headers=headers)
    except Exception as exc:
        logger.debug("probe_gemini_tier: network error: %s", exc)
        return "unknown"
    rpd_header = {k.lower(): v for k, v in resp.headers.items()}.get("x-ratelimit-limit-requests-per-day")
    try:
        if rpd_header:  # free-tier daily caps top out at 1000 (flash-lite); Tier 1 starts ~1500+
            return "free" if int(rpd_header) <= 1000 else "paid"
    except (TypeError, ValueError):
        pass
    if resp.status_code == 429:
        return "free" if "free_tier" in _response_text(resp).lower() else "paid"
    return "paid" if 200 <= resp.status_code < 300 else "unknown"


def _response_text(response: Any) -> str:
    try:
        return response.text or ""
    except Exception:
        return ""


def is_free_tier_quota_error(error_message: str) -> bool:
    """True when a Gemini 429 message indicates free-tier exhaustion."""
    return bool(error_message) and "free_tier" in error_message.lower()


def is_standard_key_auth_error(status: int, error_message: str, reason: str = "") -> bool:
    """True when a Gemini 401 means Google rejected the key TYPE (legacy "Standard" Cloud key → misleading
    "expected OAuth 2 access token" / ErrorInfo ``ACCESS_TOKEN_TYPE_UNSUPPORTED``). Narrow so ``API_KEY_INVALID``
    keeps its message."""
    return status == 401 and (reason == "ACCESS_TOKEN_TYPE_UNSUPPORTED" or "expected oauth 2 access token" in (error_message or "").lower())


class GeminiAPIError(Exception):
    """Error shape compatible with Hermes retry/error classification."""

    def __init__(self, message: str, *, code: str = "gemini_api_error", status_code: Optional[int] = None,
                 response: Optional[httpx.Response] = None, retry_after: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code, self.status_code, self.response = code, status_code, response
        self.retry_after, self.details = retry_after, details or {}


# ── OpenAI → Gemini request translation ──────────────────────────────────────
def _text_of(item: Any) -> Optional[str]:
    """Text of an OpenAI content item (plain str or ``{"type": "text"}`` dict), else None."""
    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
        return item["text"]
    return item if isinstance(item, str) else None


def _coerce_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(t for t in map(_text_of, content) if t is not None)
    return "" if content is None else str(content)


def _inline_data_part(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``inlineData`` part for an ``image_url`` item carrying a ``data:`` URL; None otherwise."""
    url = (item.get("image_url") or {}).get("url") or ""
    if item.get("type") != "image_url" or not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        header, encoded = url.split(",", 1)
        data = base64.b64encode(base64.b64decode(encoded)).decode("ascii")
        return {"inlineData": {"mimeType": header.split(":", 1)[1].split(";", 1)[0], "data": data}}
    except Exception:
        return None


def _multimodal_part(item: Any) -> Optional[Dict[str, Any]]:
    text = _text_of(item)
    if text or isinstance(item, str):
        return {"text": text}
    return _inline_data_part(item) if isinstance(item, dict) else None


def _extract_multimodal_parts(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return [p for p in map(_multimodal_part, content) if p]
    text = _coerce_content_to_text(content)
    return [{"text": text}] if text else []


def _tool_call_extra_signature(tool_call: Dict[str, Any]) -> Optional[str]:
    """Replayed Gemini thoughtSignature from ``extra_content`` (``google.thought_signature`` or flat)."""
    extra = tool_call.get("extra_content") or {}
    sig = (extra.get("google") or extra.get("thought_signature")) if isinstance(extra, dict) else None
    if isinstance(sig, dict):
        sig = sig.get("thought_signature") or sig.get("thoughtSignature")
    return sig if isinstance(sig, str) and sig else None


def _tool_call_id(tool_call: Dict[str, Any]) -> str:
    return str(tool_call.get("id") or tool_call.get("call_id") or "")


def _translate_tool_call_to_gemini(tool_call: Dict[str, Any], include_ids: bool = False) -> Dict[str, Any]:
    fn = tool_call.get("function") or {}
    args_raw = fn.get("arguments", "")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else {}
    except json.JSONDecodeError:
        args = {"_raw": args_raw}
    call: Dict[str, Any] = {"name": str(fn.get("name") or ""), "args": args if isinstance(args, dict) else {"_value": args}}
    if include_ids and (call_id := _tool_call_id(tool_call)):
        call["id"] = call_id
    return {"functionCall": call, "thoughtSignature": _tool_call_extra_signature(tool_call) or _SKIP_SIGNATURE}


def _looks_like_json_schema(node: Any) -> bool:
    """True if a parsed value contains a JSON-Schema ``$ref`` pointer (``#/...``): Gemini 3 resolves
    ``$ref``/``$defs`` inside functionResponse.response and rejects unknown pointers with HTTP 400, so a
    tool result that is itself a JSON Schema (e.g. ``tool_describe`` output) is forwarded as opaque text.
    False positives only lose the structured shape, never the content."""
    if isinstance(node, dict):
        return any((k == "$ref" and isinstance(v, str) and v.startswith("#/")) or _looks_like_json_schema(v) for k, v in node.items())
    return isinstance(node, list) and any(_looks_like_json_schema(item) for item in node)


def _translate_tool_result_to_gemini(
    message: Dict[str, Any], tool_name_by_call_id: Optional[Dict[str, str]] = None, include_ids: bool = False, *, is_gemini3: bool = False,
) -> Dict[str, Any]:
    tool_call_id = str(message.get("tool_call_id") or "")
    # functionResponse.name must echo the matching functionCall.name, so the call-id
    # mapping beats the result's own name (may be an unwrapped MCP name via `tool_call`).
    name = str((tool_name_by_call_id or {}).get(tool_call_id) or message.get("name") or tool_call_id or "tool")
    raw_content = message.get("content")
    content = _coerce_content_to_text(raw_content)
    try:
        parsed = json.loads(content) if content.strip().startswith(("{", "[")) else None
    except json.JSONDecodeError:
        parsed = None
    # Gemini 3 resolves JSON-Schema ``$ref`` pointers inside a functionResponse.response payload and rejects
    # unknown references with HTTP 400 INVALID_ARGUMENT ("referenced name '#/$defs/...' does not match a
    # display_name"; see vercel/ai#14369). A tool result that is itself a JSON Schema (e.g. tool_describe
    # output for an MCP tool) must therefore be forwarded as opaque text, not as a structured response.
    structured = isinstance(parsed, dict) and not _looks_like_json_schema(parsed)
    function_response: Dict[str, Any] = {"name": name, "response": parsed if structured else {"output": content}}
    if include_ids and tool_call_id:
        function_response["id"] = tool_call_id
    # Gemini 3.x accepts images inside functionResponse.parts; 2.x rejects the field.
    if image_parts := [p for p in _extract_multimodal_parts(raw_content) if "inlineData" in p] if is_gemini3 else []:
        function_response["parts"] = image_parts
    return {"functionResponse": function_response}


def _has_function_response(content: Dict[str, Any]) -> bool:
    return any(isinstance(part, dict) and "functionResponse" in part for part in content.get("parts", []))


def _merge_alternating(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Alternation contract for generateContent: 1) adjacent same-role contents merge (else HTTP 400
    "multiturn requests [must] alternate"); 2) EXCEPT never fuse a human user text turn into a preceding
    user content that only carries functionResponse parts (or vice versa) — Gemini 3 accepts the fold but
    reads the text as a continuation of the tool result and returns an empty candidate (parallel
    functionResponse + functionResponse still merge); 3) the split pair stays API-valid via an interposed
    placeholder model turn."""
    merged: List[Dict[str, Any]] = []
    # Compatibility contract for native Gemini generateContent: 1) Same-role adjacent contents still merge
    # in general (strict user/model alternation for ordinary text turns and parallel tool-result grouping;
    # consecutive same-role contents are rejected with HTTP 400 "Please ensure that multiturn requests
    # alternate between user and model"). 2) Exception: do NOT fuse a human user text turn into a preceding
    # user content that only carries functionResponse parts (or vice versa). Gemini 3 accepts that fold with
    # HTTP 200 but then reads the trailing text as a continuation of the tool result — it returns an empty
    # candidate or "finishes the user's sentence" instead of answering (same defect gemini-cli fixed in
    # google-gemini/gemini-cli#28700). 3) Because rule 1's HTTP 400 makes two consecutive user contents
    # unsafe to emit (#55125 — the reason this merge exists), the split pair is kept API-valid by
    # interposing a placeholder model turn between the functionResponse content and the human text content,
    # mirroring gemini-cli's INTERRUPTED_RESPONSE_PLACEHOLDER repair.
    for content in contents:
        prev = merged[-1] if merged else None
        same_role = prev is not None and prev["role"] == content["role"]
        if same_role and content["role"] == "user" and _has_function_response(prev) != _has_function_response(content):
            same_role = False
            merged.append({"role": "model", "parts": [{"text": _INTERRUPTED_RESPONSE_PLACEHOLDER}]})
        if same_role:
            merged[-1]["parts"].extend(content["parts"])
        else:
            merged.append(content)
    return merged


def _build_gemini_contents(
    messages: List[Dict[str, Any]], include_tool_call_ids: bool = False, *, is_gemini3: bool = False
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    system_text_parts: List[str] = []
    contents: List[Dict[str, Any]] = []
    tool_name_by_call_id: Dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        if role == "system":
            system_text_parts.append(_coerce_content_to_text(msg.get("content")))
            continue
        if role in {"tool", "function"}:
            part = _translate_tool_result_to_gemini(msg, tool_name_by_call_id, include_tool_call_ids, is_gemini3=is_gemini3)
            contents.append({"role": "user", "parts": [part]})
            continue
        parts = _extract_multimodal_parts(msg.get("content"))
        tool_calls = msg.get("tool_calls") or []
        for tool_call in (tc for tc in tool_calls if isinstance(tc, dict)) if isinstance(tool_calls, list) else ():
            tool_name = str((tool_call.get("function") or {}).get("name") or "")
            if (call_id := _tool_call_id(tool_call)) and tool_name:
                tool_name_by_call_id[call_id] = tool_name
            parts.append(_translate_tool_call_to_gemini(tool_call, include_ids=include_tool_call_ids))
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    joined_system = "\n".join(part for part in system_text_parts if part).strip()
    return _merge_alternating(contents), ({"role": "system", "parts": [{"text": joined_system}]} if joined_system else None)


def _function_declaration(tool: Any) -> Optional[Dict[str, Any]]:
    fn = (tool.get("function") or {}) if isinstance(tool, dict) else None
    if not isinstance(fn, dict) or not (isinstance(fn.get("name"), str) and fn["name"]):
        return None
    decl: Dict[str, Any] = {"name": fn["name"]}
    if isinstance(fn.get("description"), str) and fn["description"]:
        decl["description"] = fn["description"]
    if isinstance(fn.get("parameters"), dict):
        decl["parameters"] = sanitize_gemini_tool_parameters(fn["parameters"])
    return decl


def _translate_tools_to_gemini(tools: Any) -> List[Dict[str, Any]]:
    declarations = [d for d in map(_function_declaration, tools if isinstance(tools, list) else []) if d]
    return [{"functionDeclarations": declarations}] if declarations else []


def _translate_tool_choice_to_gemini(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if isinstance(tool_choice, str) and tool_choice in _TOOL_CHOICE_MODES:
        return {"functionCallingConfig": {"mode": _TOOL_CHOICE_MODES[tool_choice]}}
    name = (tool_choice.get("function") or {}).get("name") if isinstance(tool_choice, dict) else None
    return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}} if isinstance(name, str) and name else None


# (camelCase key, snake_case alias, accepted types, normalizer)
_THINKING_KEYS = (
    ("thinkingBudget", "thinking_budget", (int, float), int),
    ("includeThoughts", "include_thoughts", bool, lambda v: v),
    ("thinkingLevel", "thinking_level", str, lambda v: v.strip().lower()),
)


def _normalize_thinking_config(config: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(config, dict):
        return None
    values = {key: config.get(key, config.get(alias)) for key, alias, _, _ in _THINKING_KEYS}
    normalized = {key: norm(values[key]) for key, _, types, norm in _THINKING_KEYS
                  if isinstance(values[key], types) and (values[key].strip() if isinstance(values[key], str) else True)}
    return normalized or None


def _thinking_requests_output_headroom(thinking_config: Any) -> bool:
    """True when Gemini will spend output tokens on thinking: thought tokens bill against ``maxOutputTokens``,
    so a global 4096/16384 cap can be consumed entirely by high thinking (``finishReason=MAX_TOKENS``, no answer)."""
    normalized = _normalize_thinking_config(thinking_config) or {}
    budget, has_level = normalized.get("thinkingBudget"), "thinkingLevel" in normalized
    if normalized.get("includeThoughts") is False:
        return has_level or bool(budget)
    return bool(normalized) and not (isinstance(budget, int) and budget <= 0 and not has_level)


def _effective_gemini_max_output_tokens(max_tokens: Optional[int], thinking_config: Any) -> int:
    """Native ``maxOutputTokens``: an omitted/invalid cap becomes the published ceiling (Gemini
    truncates on its low internal default); an explicit cap is raised to the ceiling when
    thinking is enabled so thoughts don't starve the answer."""
    try:
        requested = int(max_tokens)
    except (TypeError, ValueError):
        requested = 0
    if requested <= 0 or _thinking_requests_output_headroom(thinking_config):
        return max(requested, GEMINI_DEFAULT_MAX_OUTPUT_TOKENS)
    return requested


def build_gemini_request(
    *, messages: List[Dict[str, Any]], tools: Any = None, tool_choice: Any = None, temperature: Optional[float] = None,
    max_tokens: Optional[int] = None, top_p: Optional[float] = None, stop: Any = None, thinking_config: Any = None,
    model: str = "",
) -> Dict[str, Any]:
    # Gemini 3+ both requires tool-call ids and accepts multimodal functionResponse parts.
    is_gemini3 = gemini_requires_tool_call_ids(model)
    contents, system_instruction = _build_gemini_contents(messages, include_tool_call_ids=is_gemini3, is_gemini3=is_gemini3)
    optional = (
        ("systemInstruction", system_instruction), ("tools", _translate_tools_to_gemini(tools)),
        ("toolConfig", _translate_tool_choice_to_gemini(tool_choice)),
    )
    request: Dict[str, Any] = {"contents": contents, **{k: v for k, v in optional if v}}
    # Key order is part of the wire format (prompt-cache parity): temperature, maxOutputTokens, topP, stop, thinking.
    generation = (
        ("temperature", temperature), ("maxOutputTokens", _effective_gemini_max_output_tokens(max_tokens, thinking_config)),
        ("topP", top_p), ("stopSequences", (stop if isinstance(stop, list) else [str(stop)]) if stop else None),
        ("thinkingConfig", _normalize_thinking_config(thinking_config)),
    )
    request["generationConfig"] = {k: v for k, v in generation if v is not None}
    return request


# ── Gemini → OpenAI response translation ─────────────────────────────────────
def _tool_call_extra_from_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sig = part.get("thoughtSignature")
    return {"google": {"thought_signature": sig}} if isinstance(sig, str) and sig else None


def _new_call_id(fc: Dict[str, Any]) -> str:
    """Echo the functionCall/delta ``id`` when present, else mint an OpenAI-style one."""
    fc_id = fc.get("id")
    return fc_id if isinstance(fc_id, str) and fc_id else f"call_{uuid.uuid4().hex[:12]}"


def _dump_call_args(fc: Dict[str, Any], **kwargs: Any) -> str:
    try:
        return json.dumps(fc.get("args") or {}, ensure_ascii=False, **kwargs)
    except (TypeError, ValueError):
        return "{}"


def _usage_from_metadata(usage_meta: Dict[str, Any]) -> SimpleNamespace:
    count = lambda key: int(usage_meta.get(key) or 0)  # noqa: E731
    return SimpleNamespace(
        prompt_tokens=count("promptTokenCount"), completion_tokens=count("candidatesTokenCount"),
        total_tokens=count("totalTokenCount"), prompt_tokens_details=SimpleNamespace(cached_tokens=count("cachedContentTokenCount")),
    )


def _envelope(model: str, object_: str, choice: SimpleNamespace, usage: Any, cls: type = SimpleNamespace) -> Any:
    """OpenAI chat.completion / chat.completion.chunk envelope around one choice."""
    return cls(id=f"chatcmpl-{uuid.uuid4().hex[:12]}", object=object_, created=int(time.time()), model=model,
               choices=[choice], usage=usage)


def _tool_call_ns(name: str, arguments: str, index: int, call_id: str, extra_content: Any) -> SimpleNamespace:
    """OpenAI-shaped tool call; ``extra_content`` attached only when it is a dict."""
    extra = {"extra_content": extra_content} if isinstance(extra_content, dict) else {}
    return SimpleNamespace(id=call_id, type="function", index=index, function=SimpleNamespace(name=name, arguments=arguments), **extra)


def _part_text(part: Dict[str, Any]) -> tuple[Optional[str], bool]:
    """``(text, is_thought)`` for a candidate part; ``(None, False)`` when it carries no text."""
    text = part.get("text")
    return (text, part.get("thought") is True) if isinstance(text, str) else (None, False)


def _part_function_call(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fc = part.get("functionCall")
    return fc if isinstance(fc, dict) and fc.get("name") else None


def translate_gemini_response(resp: Dict[str, Any], model: str) -> SimpleNamespace:
    candidates = resp.get("candidates") or []
    cand = parts = None
    if isinstance(candidates, list) and candidates:
        cand = candidates[0] if isinstance(candidates[0], dict) else {}
        content_obj = cand.get("content")
        parts = content_obj.get("parts") if isinstance(content_obj, dict) else []
    pieces: Dict[bool, List[str]] = {False: [], True: []}  # is_thought → text pieces
    tool_calls: List[SimpleNamespace] = []
    for index, part in enumerate(parts or []):
        if not isinstance(part, dict):
            continue
        text, is_thought = _part_text(part)
        if text is not None:
            pieces[is_thought].append(text)
        elif fc := _part_function_call(part):
            tool_calls.append(_tool_call_ns(str(fc["name"]), _dump_call_args(fc), index, _new_call_id(fc), _tool_call_extra_from_part(part)))
    finish_reason = "tool_calls" if tool_calls else _FINISH_REASON_MAP.get(str((cand or {}).get("finishReason") or "").upper(), "stop")
    usage = _usage_from_metadata((resp.get("usageMetadata") or {}) if cand is not None else {})
    reasoning = "".join(pieces[True]) or None
    message = SimpleNamespace(role="assistant", content="".join(pieces[False]) if pieces[False] else ("" if cand is None else None),
                              tool_calls=tool_calls or None, reasoning=reasoning, reasoning_content=reasoning, reasoning_details=None)
    return _envelope(model, "chat.completion", SimpleNamespace(index=0, message=message, finish_reason=finish_reason), usage)


class _GeminiStreamChunk(SimpleNamespace): ...


def _make_stream_chunk(
    *, model: str, content: str = "", tool_call_delta: Optional[Dict[str, Any]] = None, finish_reason: Optional[str] = None, reasoning: str = "",
) -> _GeminiStreamChunk:
    d = tool_call_delta
    tool_calls = None if d is None else [
        _tool_call_ns(d.get("name") or "", d.get("arguments") or "", d.get("index", 0), _new_call_id(d), d.get("extra_content"))
    ]
    delta = SimpleNamespace(role="assistant", content=content or None, tool_calls=tool_calls, reasoning=reasoning or None,
                            reasoning_content=reasoning or None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return _envelope(model, "chat.completion.chunk", choice, None, cls=_GeminiStreamChunk)


def _iter_sse_events(response: httpx.Response) -> Iterator[Dict[str, Any]]:
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk or ""
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line.startswith("data: "):
                continue
            if (data := line[6:]) == "[DONE]":
                return
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Non-JSON Gemini SSE line: %s", data[:200])
                continue
            if isinstance(payload, dict):
                yield payload


def translate_stream_event(event: Dict[str, Any], model: str, tool_call_indices: Dict[str, Dict[str, Any]]) -> List[_GeminiStreamChunk]:
    candidates = event.get("candidates") or []
    if not candidates:
        return []
    cand = candidates[0] if isinstance(candidates[0], dict) else {}
    parts = (cand.get("content") or {}).get("parts") or []
    chunks: List[_GeminiStreamChunk] = []
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        text, is_thought = _part_text(part)
        if is_thought:
            chunks.append(_make_stream_chunk(model=model, reasoning=text))
            continue
        if text:
            chunks.append(_make_stream_chunk(model=model, content=text))
        if fc := _part_function_call(part):
            name = str(fc["name"])
            args_str = _dump_call_args(fc, sort_keys=True)
            thought_signature = part.get("thoughtSignature") if isinstance(part.get("thoughtSignature"), str) else ""
            call_key = json.dumps({"part_index": part_index, "name": name, "thought_signature": thought_signature}, sort_keys=True)
            if (slot := tool_call_indices.get(call_key)) is None:
                slot = tool_call_indices[call_key] = {"index": len(tool_call_indices), "id": _new_call_id(fc), "last_arguments": ""}
            # Gemini re-sends the full args each event; emit only the new suffix.
            last_arguments = str(slot.get("last_arguments") or "")
            slot["last_arguments"] = args_str
            delta = {"index": slot["index"], "id": slot["id"], "name": name, "extra_content": _tool_call_extra_from_part(part),
                     "arguments": args_str[len(last_arguments):] if args_str.startswith(last_arguments) else args_str}
            chunks.append(_make_stream_chunk(model=model, tool_call_delta=delta))
    if finish_reason_raw := str(cand.get("finishReason") or ""):
        finish_reason = "tool_calls" if tool_call_indices else _FINISH_REASON_MAP.get(finish_reason_raw.upper(), "stop")
        finish_chunk = _make_stream_chunk(model=model, finish_reason=finish_reason)
        if usage_meta := event.get("usageMetadata") or {}:  # rides on the finish chunk so the stream loop records tokens
            finish_chunk.usage = _usage_from_metadata(usage_meta)
        chunks.append(finish_chunk)
    return chunks


def _error_info(err_obj: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """``(reason, metadata)`` from the first google.rpc.ErrorInfo detail (later ones fill gaps until reason is set)."""
    reason, metadata = "", {}
    details = err_obj.get("details")
    for detail in details if isinstance(details, list) else []:
        if isinstance(detail, dict) and not reason and str(detail.get("@type") or "").endswith("/google.rpc.ErrorInfo"):
            reason = detail["reason"] if isinstance(detail.get("reason"), str) else reason
            metadata = detail["metadata"] if isinstance(detail.get("metadata"), dict) else metadata
    return reason, metadata


def _error_object(body_text: str) -> Dict[str, Any]:
    """The ``error`` object of a Google JSON error body, or ``{}``."""
    try:
        parsed = json.loads(body_text) if body_text else None
    except (ValueError, TypeError):
        return {}
    err_obj = parsed.get("error") if isinstance(parsed, dict) else None
    return err_obj if isinstance(err_obj, dict) else {}


def gemini_http_error(response: httpx.Response, *, body_text: Optional[str] = None) -> GeminiAPIError:
    status = response.status_code
    body_text = (_response_text(response) if body_text is None else body_text) or ""
    err_obj = _error_object(body_text)
    err_status, err_message = (str(err_obj.get(k) or "").strip() for k in ("status", "message"))
    reason, metadata = _error_info(err_obj)
    try:
        retry_after: Optional[float] = float(response.headers.get("Retry-After") or response.headers.get("retry-after"))
    except (TypeError, ValueError):
        retry_after = None
    message = (
        f"Gemini HTTP {status} ({err_status or 'error'}): {err_message}" if err_message
        else f"Gemini returned HTTP {status}: {body_text[:500]}"
    )
    # Users who bypassed the setup wizard (raw GOOGLE_API_KEY in .env) still need to learn the free
    # tier cannot sustain an agent session; a legacy "Standard" key gets the real fix (Google's raw 401 asks for OAuth).
    if status == 429 and is_free_tier_quota_error(err_message or body_text):
        message += _FREE_TIER_GUIDANCE
    if is_standard_key_auth_error(status, err_message or body_text, reason):
        message += _STANDARD_KEY_GUIDANCE
    return GeminiAPIError(
        message, code=_HTTP_ERROR_CODES.get(status, f"gemini_http_{status}"), status_code=status, response=response,
        retry_after=retry_after, details={"status": err_status, "reason": reason, "metadata": metadata, "message": err_message},
    )


class GeminiNativeClient:
    """Minimal OpenAI-SDK-compatible facade (``client.chat.completions.create(**kwargs)``) over Gemini's native REST API."""

    # For agent/auxiliary_client.py: a complete client, never re-dispatched through a wire adapter.
    # (No HERMES_SKIP_ASYNC_WRAP — the async path has a real conversion, AsyncGeminiNativeClient.)
    HERMES_SKIP_TRANSPORT_WRAP = True

    def __init__(
        self, *, api_key: str, base_url: Optional[str] = None, default_headers: Optional[Dict[str, str]] = None,
        timeout: Any = None, http_client: Optional[httpx.Client] = None, **_: Any,
    ) -> None:
        if not (api_key or "").strip():
            raise RuntimeError(_MISSING_KEY_ERROR)
        self.api_key, self.is_closed = api_key, False
        self.base_url = (base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/").removesuffix("/openai")
        self._default_headers = dict(default_headers or {})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self._http = http_client or httpx.Client(timeout=timeout or httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0))

    def close(self) -> None:
        self.is_closed = True
        with contextlib.suppress(Exception):
            self._http.close()

    # OpenAI-client duck-type surface: callers may use ``with client:``.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "Accept": "application/json", "x-goog-api-key": self.api_key,
                "User-Agent": f"{_API_CLIENT} (gemini-native)", "X-Goog-Api-Client": _API_CLIENT, **self._default_headers}

    @staticmethod
    def _advance_stream_iterator(iterator: Iterator[_GeminiStreamChunk]) -> tuple[bool, Optional[_GeminiStreamChunk]]:
        chunk = next(iterator, _END)
        return (True, None) if chunk is _END else (False, chunk)

    def _create_chat_completion(
        self, *, model: str = "gemini-3.7-flash", messages: Optional[List[Dict[str, Any]]] = None, stream: bool = False,
        tools: Any = None, tool_choice: Any = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        top_p: Optional[float] = None, stop: Any = None, extra_body: Optional[Dict[str, Any]] = None, timeout: Any = None, **_: Any,
    ) -> Any:
        extra = extra_body if isinstance(extra_body, dict) else {}
        request = build_gemini_request(
            messages=messages or [], tools=tools, tool_choice=tool_choice, temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, stop=stop, thinking_config=extra.get("thinking_config") or extra.get("thinkingConfig"), model=model,
        )
        model = bare_gemini_model_id(model)
        url = f"{self.base_url}/models/{model}:"
        if stream:
            return self._stream_completion(model, url + "streamGenerateContent?alt=sse", request, timeout)
        response = self._http.post(url + "generateContent", json=request, headers=self._headers(), timeout=timeout)
        if response.status_code != 200:
            raise gemini_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeminiAPIError(
                f"Invalid JSON from Gemini native API: {exc}", code="gemini_invalid_json", status_code=response.status_code, response=response,
            ) from exc
        return translate_gemini_response(payload, model=model)

    def _stream_completion(self, model: str, url: str, request: Dict[str, Any], timeout: Any) -> Iterator[_GeminiStreamChunk]:
        try:
            headers = {**self._headers(), "Accept": "text/event-stream"}
            with self._http.stream("POST", url, json=request, headers=headers, timeout=timeout) as response:
                if response.status_code != 200:
                    raise gemini_http_error(response, body_text=read_streaming_error_body(response))
                tool_call_indices: Dict[str, Dict[str, Any]] = {}
                for event in _iter_sse_events(response):
                    yield from translate_stream_event(event, model, tool_call_indices)
        except httpx.HTTPError as exc:
            raise GeminiAPIError(f"Gemini streaming request failed: {exc}", code="gemini_stream_error") from exc


class AsyncGeminiNativeClient:
    """Async wrapper used by auxiliary_client for native Gemini calls."""

    def __init__(self, sync_client: GeminiNativeClient):
        # ``_real_client``: the auxiliary cache evicts entries by leaf client; GeminiNativeClient is itself the leaf.
        self._sync = self._real_client = sync_client
        self.api_key, self.base_url = sync_client.api_key, sync_client.base_url
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    # Expose the underlying sync client as _real_client so the auxiliary cache's eviction-by-leaf-client
    # helper (#23482) can find and drop this async entry when the sync GeminiNativeClient is poisoned.
    # GeminiNativeClient is itself the leaf (no OpenAI client beneath it), so we point at the sync_client
    # directly.
    async def _create_chat_completion(self, **kwargs: Any) -> Any:
        result = await asyncio.to_thread(self._sync.chat.completions.create, **kwargs)
        return self._async_stream(result) if kwargs.get("stream") else result

    async def _async_stream(self, iterator: Iterator[_GeminiStreamChunk]) -> Any:
        while not (step := await asyncio.to_thread(self._sync._advance_stream_iterator, iterator))[0]:
            yield step[1]

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)
