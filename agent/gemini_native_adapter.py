"""OpenAI-compatible facade over Google AI Studio's native Gemini API.

Hermes keeps ``api_mode='chat_completions'`` for the ``gemini`` provider so the
main agent loop can keep using its existing OpenAI-shaped message flow.
This adapter is the transport shim that converts those OpenAI-style
``messages[]`` / ``tools[]`` requests into Gemini's native
``models/{model}:generateContent`` schema and converts the responses back.

Why this exists
---------------
Google's OpenAI-compatible endpoint has been brittle for Hermes's multi-turn
agent/tool loop (auth churn, tool-call replay quirks, thought-signature
requirements).  The native Gemini API is the canonical path and avoids the
OpenAI-compat layer entirely.
"""

from __future__ import annotations

import asyncio
import base64
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

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
VERTEX_EXPRESS_BASE_URL = "https://aiplatform.googleapis.com/v1beta1"

# Published max output-token ceiling shared by every current Gemini text model
# (2.5 + 3.x: flash, flash-lite, pro). Used as the default when the caller
# passes max_tokens=None, because Gemini's native API otherwise applies a low
# internal default and truncates output (unlike OpenAI-compat endpoints where
# an omitted limit means full budget).
GEMINI_DEFAULT_MAX_OUTPUT_TOKENS = 65535


def bare_gemini_model_id(model: str) -> str:
    """Strip Gemini's own provider prefix from an aggregator-style model id."""
    name = (model or "").strip()
    lowered = name.lower()
    for prefix in ("google/", "gemini/"):
        if lowered.startswith(prefix):
            return name[len(prefix):].strip() or name
    return name


def _gemini_major_version(model: str) -> Optional[int]:
    """Extract the major version from a Gemini model id (``gemini-3.6-flash`` → 3)."""
    name = bare_gemini_model_id(model).lower()
    match = re.match(r"gemini-(\d+)", name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def gemini_requires_tool_call_ids(model: str) -> bool:
    """Whether functionCall/functionResponse parts must carry explicit ids.

    Gemini 3+ models require explicit tool call IDs in replayed history —
    without them, multi-tool turns can be rejected or mismatched. Older
    Gemini models (2.x) reject unexpected ``id`` fields, so this is gated on
    the major version. Mirrors earendil-works/pi#7494 (their fix for the same
    class of bug in the google-shared converter).
    """
    version = _gemini_major_version(model)
    return version is not None and version >= 3


def is_native_gemini_base_url(base_url: str) -> bool:
    """Return True when the endpoint speaks Gemini's native REST API."""
    normalized = str(base_url or "").strip().rstrip("/").lower()
    if not normalized:
        return False
    if normalized.endswith("/openai"):
        return False
    return (
        "generativelanguage.googleapis.com" in normalized
        or "aiplatform.googleapis.com" in normalized
    )


def is_vertex_express_key(api_key: str) -> bool:
    """Vertex Express keys are ``AQ.…``; AI Studio keys are ``AIza…``."""
    return str(api_key or "").strip().startswith("AQ.")


def _is_vertex_express_base_url(base_url: str) -> bool:
    normalized = str(base_url or "").strip().rstrip("/").lower()
    return "aiplatform.googleapis.com" in normalized


def gemini_model_path(base_url: str, model: str) -> str:
    """Path under the native host: Studio ``models/…`` vs Express ``publishers/google/models/…``."""
    name = bare_gemini_model_id(model)
    if _is_vertex_express_base_url(base_url):
        return f"publishers/google/models/{name}"
    return f"models/{name}"


def probe_gemini_tier(
    api_key: str,
    base_url: str = DEFAULT_GEMINI_BASE_URL,
    *,
    model: str = "gemini-3.7-flash",
    timeout: float = 10.0,
) -> str:
    """Probe a Google AI Studio API key and return its tier.

    Returns one of:

    - ``"free"``    -- key is on the free tier (unusable with Hermes)
    - ``"paid"``    -- key is on a paid tier
    - ``"unknown"`` -- probe failed; callers should proceed without blocking.
    """
    key = (api_key or "").strip()
    if not key:
        return "unknown"

    normalized_base = str(base_url or DEFAULT_GEMINI_BASE_URL).strip().rstrip("/")
    if not normalized_base:
        normalized_base = DEFAULT_GEMINI_BASE_URL
    if normalized_base.lower().endswith("/openai"):
        normalized_base = normalized_base[: -len("/openai")]

    url = f"{normalized_base}/{gemini_model_path(normalized_base, model)}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"maxOutputTokens": 1},
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                params={"key": key},
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Client": f"hermes-agent/{_HERMES_VERSION}",
                },
            )
    except Exception as exc:
        logger.debug("probe_gemini_tier: network error: %s", exc)
        return "unknown"

    headers_lower = {k.lower(): v for k, v in resp.headers.items()}
    rpd_header = headers_lower.get("x-ratelimit-limit-requests-per-day")
    if rpd_header:
        try:
            rpd_val = int(rpd_header)
        except (TypeError, ValueError):
            rpd_val = None
        # Published free-tier daily caps (Dec 2025):
        #   gemini-2.5-pro: 100, gemini-2.5-flash: 250, flash-lite: 1000
        # Tier 1 starts at ~1500+ for Flash. We treat <= 1000 as free.
        if rpd_val is not None and rpd_val <= 1000:
            return "free"
        if rpd_val is not None and rpd_val > 1000:
            return "paid"

    if resp.status_code == 429:
        body_text = ""
        try:
            body_text = resp.text or ""
        except Exception:
            body_text = ""
        if "free_tier" in body_text.lower():
            return "free"
        return "paid"

    if 200 <= resp.status_code < 300:
        return "paid"

    return "unknown"


def is_free_tier_quota_error(error_message: str) -> bool:
    """Return True when a Gemini 429 message indicates free-tier exhaustion."""
    if not error_message:
        return False
    return "free_tier" in error_message.lower()


_FREE_TIER_GUIDANCE = (
    "\n\nYour Google API key is on the free tier (a few hundred requests/day "
    "for Gemini Flash models). Hermes typically makes 3-10 API calls per user turn, "
    "so the free tier is exhausted in a handful of messages and cannot sustain "
    "an agent session. Enable billing on your Google Cloud project and "
    "regenerate the key in a billing-enabled project: "
    "https://aistudio.google.com/apikey"
)


def is_standard_key_auth_error(
    status: int, error_message: str, reason: str = ""
) -> bool:
    """Return True when a Gemini 401 indicates Google rejected the key TYPE.

    Google began rejecting unrestricted legacy "Standard" Google Cloud API
    keys on the Gemini API on June 19, 2026, and ALL Standard keys stop
    working in September 2026. The rejection surfaces as a misleading 401
    telling the user to supply an OAuth 2 access token ("Request had invalid
    authentication credentials. Expected OAuth 2 access token, login cookie
    or other valid authentication credential."), optionally carrying
    ``google.rpc.ErrorInfo`` reason ``ACCESS_TOKEN_TYPE_UNSUPPORTED``.

    Scoped narrowly so a plain bad key (reason ``API_KEY_INVALID``,
    "API key not valid") keeps its existing message.
    """
    if status != 401:
        return False
    if reason == "ACCESS_TOKEN_TYPE_UNSUPPORTED":
        return True
    return "expected oauth 2 access token" in (error_message or "").lower()


_STANDARD_KEY_GUIDANCE = (
    "\n\nGoogle Gemini rejected this API key's type — you do NOT need OAuth. "
    "Google began rejecting legacy 'Standard' Google Cloud keys for the "
    "Gemini API on June 19, 2026, and all Standard keys stop working in "
    "September 2026. Open https://aistudio.google.com/api-keys, check the "
    "key's type and status, and create a replacement Gemini API key (or, as "
    "a temporary bridge, restrict the Standard key to "
    "generativelanguage.googleapis.com). Then update GEMINI_API_KEY / "
    "GOOGLE_API_KEY in ~/.hermes/.env and restart your session. "
    "Details: https://ai.google.dev/gemini-api/docs/api-key"
)


class GeminiAPIError(Exception):
    """Error shape compatible with Hermes retry/error classification."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "gemini_api_error",
        status_code: Optional[int] = None,
        response: Optional[httpx.Response] = None,
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.response = response
        self.retry_after = retry_after
        self.details = details or {}


def _coerce_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces)
    return str(content)


def _extract_multimodal_parts(content: Any) -> List[Dict[str, Any]]:
    if not isinstance(content, list):
        text = _coerce_content_to_text(content)
        return [{"text": text}] if text else []

    parts: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"text": item})
            continue
        if not isinstance(item, dict):
            continue
        ptype = item.get("type")
        if ptype == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append({"text": text})
        elif ptype == "image_url":
            url = ((item.get("image_url") or {}).get("url") or "")
            if not isinstance(url, str) or not url.startswith("data:"):
                continue
            try:
                header, encoded = url.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0]
                raw = base64.b64decode(encoded)
            except Exception:
                continue
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(raw).decode("ascii"),
                    }
                }
            )
    return parts


def _tool_call_extra_signature(tool_call: Dict[str, Any]) -> Optional[str]:
    extra = tool_call.get("extra_content") or {}
    if not isinstance(extra, dict):
        return None
    google = extra.get("google") or extra.get("thought_signature")
    if isinstance(google, dict):
        sig = google.get("thought_signature") or google.get("thoughtSignature")
        return str(sig) if isinstance(sig, str) and sig else None
    if isinstance(google, str) and google:
        return google
    return None


# Stands in for a model turn that never arrived (stream failure / interrupt /
# quota fallback) when history leaves a human user text turn directly after a
# tool-result turn. Interposed between the two user contents so the request
# stays alternation-valid while the user's message remains a turn of its own.
# Mirrors gemini-cli's INTERRUPTED_RESPONSE_PLACEHOLDER (gemini-cli#28700).
_INTERRUPTED_RESPONSE_PLACEHOLDER = (
    "[The previous response was interrupted before it completed.]"
)


def _translate_tool_call_to_gemini(
    tool_call: Dict[str, Any],
    include_ids: bool = False,
) -> Dict[str, Any]:
    fn = tool_call.get("function") or {}
    args_raw = fn.get("arguments", "")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else {}
    except json.JSONDecodeError:
        args = {"_raw": args_raw}
    if not isinstance(args, dict):
        args = {"_value": args}

    part: Dict[str, Any] = {
        "functionCall": {
            "name": str(fn.get("name") or ""),
            "args": args,
        }
    }
    if include_ids:
        # Gemini 3+ requires explicit tool call IDs so replayed parallel tool
        # calls pair with their functionResponses (earendil-works/pi#7494).
        tool_call_id = str(tool_call.get("id") or tool_call.get("call_id") or "")
        if tool_call_id:
            part["functionCall"]["id"] = tool_call_id
    thought_signature = _tool_call_extra_signature(tool_call)
    # Fallback sentinel for cross-provider tool_calls (e.g. fallback from
    # xAI/Anthropic to Gemini, where the original tool_call carries no
    # Gemini thoughtSignature). Mirrors gemini_cloudcode_adapter.py:106.
    # Without this, Gemini 3 thinking models reject replayed history with
    # 400 INVALID_ARGUMENT on the missing thoughtSignature.
    part["thoughtSignature"] = thought_signature or "skip_thought_signature_validator"
    return part


def _looks_like_json_schema(node: Any) -> bool:
    """True if a parsed value contains a JSON-Schema-style ``$ref`` pointer.

    Gemini 3 resolves ``$ref``/``$defs`` references inside a
    functionResponse.response payload and rejects unknown pointers with
    HTTP 400 INVALID_ARGUMENT. A tool result that is itself a JSON Schema
    (e.g. the output of ``tool_describe`` for an MCP tool) must therefore be
    forwarded as opaque text rather than as a structured response.

    Detection is deliberately structural, not semantic: any ``$ref`` value
    shaped like a JSON pointer (``#/...``) demotes the whole result. Non-schema
    data that happens to carry such a pointer is a false positive, but the raw
    content is preserved verbatim either way, so the cost is fidelity-free.
    The recursive walk is O(n) over the parsed value; tool-result payloads are
    small, so this is negligible per turn.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str) and value.startswith("#/"):
                return True
            if _looks_like_json_schema(value):
                return True
    elif isinstance(node, list):
        return any(_looks_like_json_schema(item) for item in node)
    return False


def _translate_tool_result_to_gemini(
    message: Dict[str, Any],
    tool_name_by_call_id: Optional[Dict[str, str]] = None,
    include_ids: bool = False,
    *,
    is_gemini3: bool = False,
) -> Dict[str, Any]:
    tool_name_by_call_id = tool_name_by_call_id or {}
    tool_call_id = str(message.get("tool_call_id") or "")
    # A tool result can carry the unwrapped internal tool name (for example,
    # an MCP tool invoked through the `tool_call` bridge). Gemini requires
    # functionResponse.name to echo the matching functionCall.name, so the
    # call-id mapping must take precedence over the internal result name.
    name = str(
        tool_name_by_call_id.get(tool_call_id)
        or message.get("name")
        or tool_call_id
        or "tool"
    )
    raw_content = message.get("content")
    content = _coerce_content_to_text(raw_content)
    try:
        parsed = json.loads(content) if content.strip().startswith(("{", "[")) else None
    except json.JSONDecodeError:
        parsed = None
    # Gemini 3 resolves JSON-Schema ``$ref`` pointers inside a
    # functionResponse.response payload and rejects unknown references with
    # HTTP 400 INVALID_ARGUMENT ("referenced name '#/$defs/...' does not match
    # a display_name"; see vercel/ai#14369). A tool result that is itself a
    # JSON Schema (e.g. tool_describe output for an MCP tool) must therefore
    # be forwarded as opaque text, not as a structured response.
    if isinstance(parsed, dict) and _looks_like_json_schema(parsed):
        parsed = None
    response = parsed if isinstance(parsed, dict) else {"output": content}
    function_response: Dict[str, Any] = {
        "name": name,
        "response": response,
    }
    if include_ids and tool_call_id:
        function_response["id"] = tool_call_id
    # Gemini 3.x supports embedding images directly inside
    # functionResponse.parts (Google's recommended shape for multimodal tool
    # results — see "Multimodal function responses" in the Gemini docs).
    # Gemini 2.x rejects the field, so only attach inlineData when the target
    # model supports it — otherwise the vision tool result is silently
    # downgraded to text-only.
    if is_gemini3:
        image_parts = [
            p for p in _extract_multimodal_parts(raw_content)
            if "inlineData" in p
        ]
        if image_parts:
            function_response["parts"] = image_parts
    return {"functionResponse": function_response}


def _build_gemini_contents(
    messages: List[Dict[str, Any]],
    include_tool_call_ids: bool = False,
    *,
    is_gemini3: bool = False,
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
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        _translate_tool_result_to_gemini(
                            msg,
                            tool_name_by_call_id=tool_name_by_call_id,
                            include_ids=include_tool_call_ids,
                            is_gemini3=is_gemini3,
                        )
                    ],
                }
            )
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts: List[Dict[str, Any]] = []

        content_parts = _extract_multimodal_parts(msg.get("content"))
        parts.extend(content_parts)

        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_call_id = str(tool_call.get("id") or tool_call.get("call_id") or "")
                    tool_name = str(((tool_call.get("function") or {}).get("name") or ""))
                    if tool_call_id and tool_name:
                        tool_name_by_call_id[tool_call_id] = tool_name
                    parts.append(
                        _translate_tool_call_to_gemini(
                            tool_call, include_ids=include_tool_call_ids
                        )
                    )

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    # Compatibility contract for native Gemini generateContent:
    # 1) Same-role adjacent contents still merge in general (strict user/model
    #    alternation for ordinary text turns and parallel tool-result grouping;
    #    consecutive same-role contents are rejected with HTTP 400 "Please
    #    ensure that multiturn requests alternate between user and model").
    # 2) Exception: do NOT fuse a human user text turn into a preceding user
    #    content that only carries functionResponse parts (or vice versa).
    #    Gemini 3 accepts that fold with HTTP 200 but then reads the trailing
    #    text as a continuation of the tool result — it returns an empty
    #    candidate or "finishes the user's sentence" instead of answering
    #    (same defect gemini-cli fixed in google-gemini/gemini-cli#28700).
    # 3) Because rule 1's HTTP 400 makes two consecutive user contents unsafe
    #    to emit (#55125 — the reason this merge exists), the split pair is
    #    kept API-valid by interposing a placeholder model turn between the
    #    functionResponse content and the human text content, mirroring
    #    gemini-cli's INTERRUPTED_RESPONSE_PLACEHOLDER repair.
    # 4) Parallel tool results (functionResponse + functionResponse) still
    #    merge into one user content — only mixed functionResponse/text is
    #    kept apart.
    merged_contents: List[Dict[str, Any]] = []
    for content in contents:
        same_role = bool(
            merged_contents and merged_contents[-1]["role"] == content["role"]
        )
        if same_role and content["role"] == "user":
            previous_has_function_response = any(
                isinstance(part, dict) and "functionResponse" in part
                for part in merged_contents[-1].get("parts", [])
            )
            current_has_function_response = any(
                isinstance(part, dict) and "functionResponse" in part
                for part in content.get("parts", [])
            )
            if previous_has_function_response != current_has_function_response:
                same_role = False
                merged_contents.append(
                    {
                        "role": "model",
                        "parts": [{"text": _INTERRUPTED_RESPONSE_PLACEHOLDER}],
                    }
                )

        if same_role:
            merged_contents[-1]["parts"].extend(content["parts"])
        else:
            merged_contents.append(content)
    contents = merged_contents

    system_instruction = None
    joined_system = "\n".join(part for part in system_text_parts if part).strip()
    if joined_system:
        system_instruction = {"role": "system", "parts": [{"text": joined_system}]}
    return contents, system_instruction


def _translate_tools_to_gemini(tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    declarations: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        decl: Dict[str, Any] = {"name": name}
        description = fn.get("description")
        if isinstance(description, str) and description:
            decl["description"] = description
        parameters = fn.get("parameters")
        if isinstance(parameters, dict):
            decl["parameters"] = sanitize_gemini_tool_parameters(parameters)
        declarations.append(decl)
    return [{"functionDeclarations": declarations}] if declarations else []


def _translate_tool_choice_to_gemini(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
        if tool_choice == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        if tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name")
        if isinstance(name, str) and name:
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return None


def _normalize_thinking_config(config: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(config, dict) or not config:
        return None
    budget = config.get("thinkingBudget", config.get("thinking_budget"))
    include = config.get("includeThoughts", config.get("include_thoughts"))
    level = config.get("thinkingLevel", config.get("thinking_level"))
    normalized: Dict[str, Any] = {}
    if isinstance(budget, (int, float)):
        normalized["thinkingBudget"] = int(budget)
    if isinstance(include, bool):
        normalized["includeThoughts"] = include
    if isinstance(level, str) and level.strip():
        normalized["thinkingLevel"] = level.strip().lower()
    return normalized or None


def _thinking_requests_output_headroom(thinking_config: Any) -> bool:
    """Return True when Gemini will spend output tokens on thinking.

    Gemini bills thought tokens against ``maxOutputTokens``. A global
    Hermes ``max_tokens`` of 4096/16384 is enough for visible text, but
    Ultra/high thinking can consume the entire budget and leave
    ``finishReason=MAX_TOKENS`` with no complete answer. Continuations
    then abort after 4 retries.
    """
    normalized = _normalize_thinking_config(thinking_config)
    if not normalized:
        return False
    if normalized.get("includeThoughts") is False:
        return "thinkingLevel" in normalized or bool(normalized.get("thinkingBudget"))
    budget = normalized.get("thinkingBudget")
    if isinstance(budget, int) and budget <= 0 and "thinkingLevel" not in normalized:
        return False
    return True


def _effective_gemini_max_output_tokens(
    max_tokens: Optional[int], thinking_config: Any
) -> int:
    """Resolve native ``maxOutputTokens``.

    Gemini's generateContent API does not treat an omitted cap as
    unlimited — it applies a low internal default and truncates. When
    thinking is enabled, also raise a too-small explicit cap to the
    published 65,535 ceiling so thought tokens do not starve the answer.
    """
    if max_tokens is None:
        return GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    try:
        requested = int(max_tokens)
    except (TypeError, ValueError):
        return GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    if requested <= 0:
        return GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    if _thinking_requests_output_headroom(thinking_config):
        return max(requested, GEMINI_DEFAULT_MAX_OUTPUT_TOKENS)
    return requested


def build_gemini_request(
    *,
    messages: List[Dict[str, Any]],
    tools: Any = None,
    tool_choice: Any = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Any = None,
    thinking_config: Any = None,
    model: str = "",
) -> Dict[str, Any]:
    version = _gemini_major_version(model)
    is_gemini3 = version is not None and version >= 3
    contents, system_instruction = _build_gemini_contents(
        messages,
        include_tool_call_ids=gemini_requires_tool_call_ids(model),
        is_gemini3=is_gemini3,
    )
    request: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        request["systemInstruction"] = system_instruction

    gemini_tools = _translate_tools_to_gemini(tools)
    if gemini_tools:
        request["tools"] = gemini_tools

    tool_config = _translate_tool_choice_to_gemini(tool_choice)
    if tool_config:
        request["toolConfig"] = tool_config

    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    generation_config["maxOutputTokens"] = _effective_gemini_max_output_tokens(
        max_tokens, thinking_config
    )
    if top_p is not None:
        generation_config["topP"] = top_p
    if stop:
        generation_config["stopSequences"] = stop if isinstance(stop, list) else [str(stop)]
    normalized_thinking = _normalize_thinking_config(thinking_config)
    if normalized_thinking:
        generation_config["thinkingConfig"] = normalized_thinking
    if generation_config:
        request["generationConfig"] = generation_config

    return request


def _map_gemini_finish_reason(reason: str) -> str:
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "stop",
    }
    return mapping.get(str(reason or "").upper(), "stop")


def _tool_call_extra_from_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sig = part.get("thoughtSignature")
    if isinstance(sig, str) and sig:
        return {"google": {"thought_signature": sig}}
    return None


def _empty_response(model: str) -> SimpleNamespace:
    message = SimpleNamespace(
        role="assistant",
        content="",
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


def translate_gemini_response(resp: Dict[str, Any], model: str) -> SimpleNamespace:
    candidates = resp.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return _empty_response(model)

    cand = candidates[0] if isinstance(candidates[0], dict) else {}
    content_obj = cand.get("content") if isinstance(cand, dict) else {}
    parts = content_obj.get("parts") if isinstance(content_obj, dict) else []

    text_pieces: List[str] = []
    reasoning_pieces: List[str] = []
    tool_calls: List[SimpleNamespace] = []

    for index, part in enumerate(parts or []):
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True and isinstance(part.get("text"), str):
            reasoning_pieces.append(part["text"])
            continue
        if isinstance(part.get("text"), str):
            text_pieces.append(part["text"])
            continue
        fc = part.get("functionCall")
        if isinstance(fc, dict) and fc.get("name"):
            try:
                args_str = json.dumps(fc.get("args") or {}, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = "{}"
            tool_call = SimpleNamespace(
                id=(
                    str(fc["id"])
                    if isinstance(fc.get("id"), str) and fc.get("id")
                    else f"call_{uuid.uuid4().hex[:12]}"
                ),
                type="function",
                index=index,
                function=SimpleNamespace(name=str(fc["name"]), arguments=args_str),
            )
            extra_content = _tool_call_extra_from_part(part)
            if extra_content:
                tool_call.extra_content = extra_content
            tool_calls.append(tool_call)

    finish_reason = "tool_calls" if tool_calls else _map_gemini_finish_reason(str(cand.get("finishReason") or ""))
    usage_meta = resp.get("usageMetadata") or {}
    usage = SimpleNamespace(
        prompt_tokens=int(usage_meta.get("promptTokenCount") or 0),
        completion_tokens=int(usage_meta.get("candidatesTokenCount") or 0),
        total_tokens=int(usage_meta.get("totalTokenCount") or 0),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=int(usage_meta.get("cachedContentTokenCount") or 0),
        ),
    )
    reasoning = "".join(reasoning_pieces) or None
    message = SimpleNamespace(
        role="assistant",
        content="".join(text_pieces) if text_pieces else None,
        tool_calls=tool_calls or None,
        reasoning=reasoning,
        reasoning_content=reasoning,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


class _GeminiStreamChunk(SimpleNamespace):
    pass


def _make_stream_chunk(
    *,
    model: str,
    content: str = "",
    tool_call_delta: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
    reasoning: str = "",
) -> _GeminiStreamChunk:
    delta_kwargs: Dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": None,
        "reasoning": None,
        "reasoning_content": None,
    }
    if content:
        delta_kwargs["content"] = content
    if tool_call_delta is not None:
        tool_delta = SimpleNamespace(
            index=tool_call_delta.get("index", 0),
            id=tool_call_delta.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function=SimpleNamespace(
                name=tool_call_delta.get("name") or "",
                arguments=tool_call_delta.get("arguments") or "",
            ),
        )
        extra_content = tool_call_delta.get("extra_content")
        if isinstance(extra_content, dict):
            tool_delta.extra_content = extra_content
        delta_kwargs["tool_calls"] = [tool_delta]
    if reasoning:
        delta_kwargs["reasoning"] = reasoning
        delta_kwargs["reasoning_content"] = reasoning
    delta = SimpleNamespace(**delta_kwargs)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return _GeminiStreamChunk(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=None,
    )


def _iter_sse_events(response: httpx.Response) -> Iterator[Dict[str, Any]]:
    buffer = ""
    for chunk in response.iter_text():
        if not chunk:
            continue
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
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
    parts = ((cand.get("content") or {}).get("parts") or []) if isinstance(cand, dict) else []
    chunks: List[_GeminiStreamChunk] = []

    for part_index, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True and isinstance(part.get("text"), str):
            chunks.append(_make_stream_chunk(model=model, reasoning=part["text"]))
            continue
        if isinstance(part.get("text"), str) and part["text"]:
            chunks.append(_make_stream_chunk(model=model, content=part["text"]))
        fc = part.get("functionCall")
        if isinstance(fc, dict) and fc.get("name"):
            name = str(fc["name"])
            try:
                args_str = json.dumps(fc.get("args") or {}, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                args_str = "{}"
            thought_signature = part.get("thoughtSignature") if isinstance(part.get("thoughtSignature"), str) else ""
            call_key = json.dumps(
                {
                    "part_index": part_index,
                    "name": name,
                    "thought_signature": thought_signature,
                },
                sort_keys=True,
            )
            slot = tool_call_indices.get(call_key)
            if slot is None:
                slot = {
                    "index": len(tool_call_indices),
                    "id": (
                        str(fc["id"])
                        if isinstance(fc.get("id"), str) and fc.get("id")
                        else f"call_{uuid.uuid4().hex[:12]}"
                    ),
                    "last_arguments": "",
                }
                tool_call_indices[call_key] = slot
            emitted_arguments = args_str
            last_arguments = str(slot.get("last_arguments") or "")
            if last_arguments:
                if args_str == last_arguments:
                    emitted_arguments = ""
                elif args_str.startswith(last_arguments):
                    emitted_arguments = args_str[len(last_arguments):]
            slot["last_arguments"] = args_str
            chunks.append(
                _make_stream_chunk(
                    model=model,
                    tool_call_delta={
                        "index": slot["index"],
                        "id": slot["id"],
                        "name": name,
                        "arguments": emitted_arguments,
                        "extra_content": _tool_call_extra_from_part(part),
                    },
                )
            )

    finish_reason_raw = str(cand.get("finishReason") or "")
    if finish_reason_raw:
        mapped = "tool_calls" if tool_call_indices else _map_gemini_finish_reason(finish_reason_raw)
        finish_chunk = _make_stream_chunk(model=model, finish_reason=mapped)
        # Attach usage from this event's usageMetadata so the streaming
        # loop in run_agent.py can record token counts (mirrors the
        # non-streaming path in translate_gemini_response).
        usage_meta = event.get("usageMetadata") or {}
        if usage_meta:
            finish_chunk.usage = SimpleNamespace(
                prompt_tokens=int(usage_meta.get("promptTokenCount") or 0),
                completion_tokens=int(usage_meta.get("candidatesTokenCount") or 0),
                total_tokens=int(usage_meta.get("totalTokenCount") or 0),
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=int(usage_meta.get("cachedContentTokenCount") or 0),
                ),
            )
        chunks.append(finish_chunk)
    return chunks


def gemini_http_error(
    response: httpx.Response, *, body_text: Optional[str] = None
) -> GeminiAPIError:
    status = response.status_code
    body_json: Dict[str, Any] = {}
    if body_text is None:
        try:
            body_text = response.text
        except Exception:
            body_text = ""
    body_text = body_text or ""
    if body_text:
        try:
            parsed = json.loads(body_text)
            if isinstance(parsed, dict):
                body_json = parsed
        except (ValueError, TypeError):
            body_json = {}

    err_obj = body_json.get("error") if isinstance(body_json, dict) else None
    if not isinstance(err_obj, dict):
        err_obj = {}
    err_status = str(err_obj.get("status") or "").strip()
    err_message = str(err_obj.get("message") or "").strip()
    _raw_details = err_obj.get("details")
    details_list = _raw_details if isinstance(_raw_details, list) else []

    reason = ""
    retry_after: Optional[float] = None
    metadata: Dict[str, Any] = {}
    for detail in details_list:
        if not isinstance(detail, dict):
            continue
        type_url = str(detail.get("@type") or "")
        if not reason and type_url.endswith("/google.rpc.ErrorInfo"):
            reason_value = detail.get("reason")
            if isinstance(reason_value, str):
                reason = reason_value
            md = detail.get("metadata")
            if isinstance(md, dict):
                metadata = md
    header_retry = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if header_retry:
        try:
            retry_after = float(header_retry)
        except (TypeError, ValueError):
            retry_after = None

    code = f"gemini_http_{status}"
    if status == 401:
        code = "gemini_unauthorized"
    elif status == 429:
        code = "gemini_rate_limited"
    elif status == 404:
        code = "gemini_model_not_found"

    if err_message:
        message = f"Gemini HTTP {status} ({err_status or 'error'}): {err_message}"
    else:
        message = f"Gemini returned HTTP {status}: {body_text[:500]}"

    # Free-tier quota exhaustion -> append actionable guidance so users who
    # bypassed the setup wizard (direct GOOGLE_API_KEY in .env) still learn
    # that the free tier cannot sustain an agent session.
    if status == 429 and is_free_tier_quota_error(err_message or body_text):
        message = message + _FREE_TIER_GUIDANCE

    # Legacy "Standard" Google Cloud key rejection (June 19, 2026 onward) ->
    # Google's raw 401 misleadingly tells the user to use OAuth. Append the
    # actual fix (mint a new Gemini API key in AI Studio).
    if is_standard_key_auth_error(status, err_message or body_text, reason):
        message = message + _STANDARD_KEY_GUIDANCE

    return GeminiAPIError(
        message,
        code=code,
        status_code=status,
        response=response,
        retry_after=retry_after,
        details={
            "status": err_status,
            "reason": reason,
            "metadata": metadata,
            "message": err_message,
        },
    )


class _GeminiChatCompletions:
    def __init__(self, client: "GeminiNativeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _AsyncGeminiChatCompletions:
    def __init__(self, client: "AsyncGeminiNativeClient"):
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return await self._client._create_chat_completion(**kwargs)


class _GeminiChatNamespace:
    def __init__(self, client: "GeminiNativeClient"):
        self.completions = _GeminiChatCompletions(client)


class _AsyncGeminiChatNamespace:
    def __init__(self, client: "AsyncGeminiNativeClient"):
        self.completions = _AsyncGeminiChatCompletions(client)


class GeminiNativeClient:
    """Minimal OpenAI-SDK-compatible facade over Gemini's native REST API."""

    # Declared for agent/auxiliary_client.py: already a complete client, so it
    # is never re-dispatched through a wire adapter. (No HERMES_SKIP_ASYNC_WRAP
    # — the async path has a real conversion, AsyncGeminiNativeClient.)
    HERMES_SKIP_TRANSPORT_WRAP = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
        http_client: Optional[httpx.Client] = None,
        **_: Any,
    ) -> None:
        if not (api_key or "").strip():
            raise RuntimeError(
                "Gemini native client requires an API key, but none was provided. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment / ~/.hermes/.env "
                "(get one at https://aistudio.google.com/app/apikey), or run `hermes setup` "
                "to configure the Google provider."
            )
        self.api_key = api_key
        normalized_base = (base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/")
        if normalized_base.endswith("/openai"):
            normalized_base = normalized_base[: -len("/openai")]
        if is_vertex_express_key(api_key) and not _is_vertex_express_base_url(normalized_base):
            normalized_base = VERTEX_EXPRESS_BASE_URL
        self.base_url = normalized_base
        self._default_headers = dict(default_headers or {})
        self.chat = _GeminiChatNamespace(self)
        self.is_closed = False
        self._http = http_client or httpx.Client(
            timeout=timeout or httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)
        )

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-goog-api-key": self.api_key,
            # Include Hermes client context following Gemini's partner
            # integration guidance.
            # See https://ai.google.dev/gemini-api/docs/partner-integration
            "User-Agent": f"hermes-agent/{_HERMES_VERSION} (gemini-native)",
            "X-Goog-Api-Client": f"hermes-agent/{_HERMES_VERSION}",
        }
        headers.update(self._default_headers)
        return headers

    @staticmethod
    def _advance_stream_iterator(iterator: Iterator[_GeminiStreamChunk]) -> tuple[bool, Optional[_GeminiStreamChunk]]:
        try:
            return False, next(iterator)
        except StopIteration:
            return True, None

    def _create_chat_completion(
        self,
        *,
        model: str = "gemini-3.7-flash",
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Any = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = extra_body.get("thinking_config") or extra_body.get("thinkingConfig")

        request = build_gemini_request(
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking_config=thinking_config,
            model=model,
        )

        model = bare_gemini_model_id(model)
        if stream:
            return self._stream_completion(model=model, request=request, timeout=timeout)

        url = f"{self.base_url}/{gemini_model_path(self.base_url, model)}:generateContent"
        response = self._http.post(url, json=request, headers=self._headers(), timeout=timeout)
        if response.status_code != 200:
            raise gemini_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeminiAPIError(
                f"Invalid JSON from Gemini native API: {exc}",
                code="gemini_invalid_json",
                status_code=response.status_code,
                response=response,
            ) from exc
        return translate_gemini_response(payload, model=model)

    def _stream_completion(self, *, model: str, request: Dict[str, Any], timeout: Any = None) -> Iterator[_GeminiStreamChunk]:
        url = f"{self.base_url}/{gemini_model_path(self.base_url, model)}:streamGenerateContent?alt=sse"
        stream_headers = dict(self._headers())
        stream_headers["Accept"] = "text/event-stream"

        def _generator() -> Iterator[_GeminiStreamChunk]:
            try:
                with self._http.stream("POST", url, json=request, headers=stream_headers, timeout=timeout) as response:
                    if response.status_code != 200:
                        body_text = read_streaming_error_body(response)
                        raise gemini_http_error(response, body_text=body_text)
                    tool_call_indices: Dict[str, Dict[str, Any]] = {}
                    for event in _iter_sse_events(response):
                        for chunk in translate_stream_event(event, model, tool_call_indices):
                            yield chunk
            except httpx.HTTPError as exc:
                raise GeminiAPIError(
                    f"Gemini streaming request failed: {exc}",
                    code="gemini_stream_error",
                ) from exc

        return _generator()


class AsyncGeminiNativeClient:
    """Async wrapper used by auxiliary_client for native Gemini calls."""

    def __init__(self, sync_client: GeminiNativeClient):
        self._sync = sync_client
        self.api_key = sync_client.api_key
        self.base_url = sync_client.base_url
        self.chat = _AsyncGeminiChatNamespace(self)
        # Expose the underlying sync client as _real_client so the auxiliary
        # cache's eviction-by-leaf-client helper (#23482) can find and drop
        # this async entry when the sync GeminiNativeClient is poisoned.
        # GeminiNativeClient is itself the leaf (no OpenAI client beneath
        # it), so we point at the sync_client directly.
        self._real_client = sync_client

    async def _create_chat_completion(self, **kwargs: Any) -> Any:
        stream = bool(kwargs.get("stream"))
        result = await asyncio.to_thread(self._sync.chat.completions.create, **kwargs)
        if not stream:
            return result

        async def _async_stream() -> Any:
            while True:
                done, chunk = await asyncio.to_thread(self._sync._advance_stream_iterator, result)
                if done:
                    break
                yield chunk

        return _async_stream()

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)
