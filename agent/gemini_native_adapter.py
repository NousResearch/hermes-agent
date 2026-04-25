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
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import httpx

from agent.gemini_schema import sanitize_gemini_tool_parameters

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def is_native_gemini_base_url(base_url: str) -> bool:
    """Return True when the endpoint speaks Gemini's native REST API."""
    normalized = str(base_url or "").strip().rstrip("/").lower()
    if not normalized:
        return False
    if "generativelanguage.googleapis.com" not in normalized:
        return False
    return not normalized.endswith("/openai")


def probe_gemini_tier(
    api_key: str,
    base_url: str = DEFAULT_GEMINI_BASE_URL,
    *,
    model: str = "gemini-2.5-flash",
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

    url = f"{normalized_base}/models/{model}:generateContent"
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
                headers={"Content-Type": "application/json"},
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
    "\n\nYour Google API key is on the free tier (<= 250 requests/day for "
    "gemini-2.5-flash). Hermes typically makes 3-10 API calls per user turn, "
    "so the free tier is exhausted in a handful of messages and cannot sustain "
    "an agent session. Enable billing on your Google Cloud project and "
    "regenerate the key in a billing-enabled project: "
    "https://aistudio.google.com/apikey"
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


def _translate_tool_call_to_gemini(tool_call: Dict[str, Any]) -> Dict[str, Any]:
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
    thought_signature = _tool_call_extra_signature(tool_call)
    if thought_signature:
        part["thoughtSignature"] = thought_signature
    return part


def _translate_tool_result_to_gemini(
    message: Dict[str, Any],
    tool_name_by_call_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    tool_name_by_call_id = tool_name_by_call_id or {}
    tool_call_id = str(message.get("tool_call_id") or "")
    name = str(
        message.get("name")
        or tool_name_by_call_id.get(tool_call_id)
        or tool_call_id
        or "tool"
    )
    content = _coerce_content_to_text(message.get("content"))
    try:
        parsed = json.loads(content) if content.strip().startswith(("{", "[")) else None
    except json.JSONDecodeError:
        parsed = None
    response = parsed if isinstance(parsed, dict) else {"output": content}
    return {
        "functionResponse": {
            "name": name,
            "response": response,
        }
    }


def _build_gemini_contents(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
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
                    parts.append(_translate_tool_call_to_gemini(tool_call))

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    system_instruction = None
    joined_system = "\n".join(part for part in system_text_parts if part).strip()
    if joined_system:
        system_instruction = {"parts": [{"text": joined_system}]}
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
) -> Dict[str, Any]:
    contents, system_instruction = _build_gemini_contents(messages)
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
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
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
                id=f"call_{uuid.uuid4().hex[:12]}",
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
                    "id": f"call_{uuid.uuid4().hex[:12]}",
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
        chunks.append(_make_stream_chunk(model=model, finish_reason=mapped))
    return chunks


def gemini_http_error(response: httpx.Response) -> GeminiAPIError:
    status = response.status_code
    body_text = ""
    body_json: Dict[str, Any] = {}
    try:
        body_text = response.text
    except Exception:
        body_text = ""
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
            "User-Agent": "hermes-agent (gemini-native)",
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
        model: str = "gemini-2.5-flash",
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
        )

        if stream:
            return self._stream_completion(model=model, request=request, timeout=timeout)

        url = f"{self.base_url}/models/{model}:generateContent"
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
        url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse"
        stream_headers = dict(self._headers())
        stream_headers["Accept"] = "text/event-stream"

        def _generator() -> Iterator[_GeminiStreamChunk]:
            try:
                with self._http.stream("POST", url, json=request, headers=stream_headers, timeout=timeout) as response:
                    if response.status_code != 200:
                        response.read()
                        raise gemini_http_error(response)
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
