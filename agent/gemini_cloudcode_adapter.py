"""OpenAI-compatible facade that talks to Google's Cloud Code Assist backend.

This adapter lets Hermes use the ``google-gemini-cli`` provider as if it were
a standard OpenAI-shaped chat completion endpoint, while the underlying HTTP
traffic goes to ``cloudcode-pa.googleapis.com/v1internal:{generateContent,
streamGenerateContent}`` with a Bearer access token obtained via OAuth PKCE.

Architecture
------------
- ``GeminiCloudCodeClient`` exposes ``.chat.completions.create(**kwargs)``
  mirroring the subset of the OpenAI SDK that ``run_agent.py`` uses.
- Incoming OpenAI ``messages[]`` / ``tools[]`` / ``tool_choice`` are translated
  to Gemini's native ``contents[]`` / ``tools[].functionDeclarations`` /
  ``toolConfig`` / ``systemInstruction`` shape.
- The request body is wrapped ``{project, model, user_prompt_id, request}``
  per Code Assist API expectations.
- Responses (``candidates[].content.parts[]``) are converted back to
  OpenAI ``choices[0].message`` shape with ``content`` + ``tool_calls``.
- Streaming uses SSE (``?alt=sse``) and yields OpenAI-shaped delta chunks.

Attribution
-----------
Translation semantics follow jenslys/opencode-gemini-auth (MIT) and the public
Gemini API docs. Request envelope shape
(``{project, model, user_prompt_id, request}``) is documented nowhere; it is
reverse-engineered from the opencode-gemini-auth and clawdbot implementations.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import httpx

from agent import google_oauth
from agent.google_code_assist import (
    CODE_ASSIST_ENDPOINT,
    FREE_TIER_ID,
    CodeAssistError,
    ProjectContext,
    resolve_project_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request translation: OpenAI → Gemini
# =============================================================================

_ROLE_MAP_OPENAI_TO_GEMINI = {
    "user": "user",
    "assistant": "model",
    "system": "user",   # handled separately via systemInstruction
    "tool": "user",     # functionResponse is wrapped in a user-role turn
    "function": "user",
}


def _coerce_content_to_text(content: Any) -> str:
    """OpenAI content may be str or a list of parts; reduce to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for p in content:
            if isinstance(p, str):
                pieces.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    pieces.append(p["text"])
                # Multimodal (image_url, etc.) — stub for now; log and skip
                elif p.get("type") in ("image_url", "input_audio"):
                    logger.debug("Dropping multimodal part (not yet supported): %s", p.get("type"))
        return "\n".join(pieces)
    return str(content)


def _translate_tool_call_to_gemini(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI tool_call -> Gemini functionCall part."""
    fn = tool_call.get("function") or {}
    args_raw = fn.get("arguments", "")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else {}
    except json.JSONDecodeError:
        args = {"_raw": args_raw}
    if not isinstance(args, dict):
        args = {"_value": args}
    return {
        "functionCall": {
            "name": fn.get("name") or "",
            "args": args,
        },
        # Sentinel signature — matches opencode-gemini-auth's approach.
        # Without this, Code Assist rejects function calls that originated
        # outside its own chain.
        "thoughtSignature": "skip_thought_signature_validator",
    }


def _translate_tool_result_to_gemini(message: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI tool-role message -> Gemini functionResponse part.

    The function name isn't in the OpenAI tool message directly; it must be
    passed via the assistant message that issued the call. For simplicity we
    look up ``name`` on the message (OpenAI SDK copies it there) or on the
    ``tool_call_id`` cross-reference.
    """
    name = str(message.get("name") or message.get("tool_call_id") or "tool")
    content = _coerce_content_to_text(message.get("content"))
    # Gemini expects the response as a dict under `response`. We wrap plain
    # text in {"output": "..."}.
    try:
        parsed = json.loads(content) if content.strip().startswith(("{", "[")) else None
    except json.JSONDecodeError:
        parsed = None
    response = parsed if isinstance(parsed, dict) else {"output": content}
    return {
        "functionResponse": {
            "name": name,
            "response": response,
        },
    }


def _build_gemini_contents(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Convert OpenAI messages[] to Gemini contents[] + systemInstruction."""
    system_text_parts: List[str] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")

        if role == "system":
            system_text_parts.append(_coerce_content_to_text(msg.get("content")))
            continue

        # Tool result message — emit a user-role turn with functionResponse
        if role == "tool" or role == "function":
            contents.append({
                "role": "user",
                "parts": [_translate_tool_result_to_gemini(msg)],
            })
            continue

        gemini_role = _ROLE_MAP_OPENAI_TO_GEMINI.get(role, "user")
        parts: List[Dict[str, Any]] = []

        text = _coerce_content_to_text(msg.get("content"))
        if text:
            parts.append({"text": text})

        # Assistant messages can carry tool_calls
        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    parts.append(_translate_tool_call_to_gemini(tc))

        if not parts:
            # Gemini rejects empty parts; skip the turn entirely
            continue

        contents.append({"role": gemini_role, "parts": parts})

    system_instruction: Optional[Dict[str, Any]] = None
    joined_system = "\n".join(p for p in system_text_parts if p).strip()
    if joined_system:
        system_instruction = {
            "role": "system",
            "parts": [{"text": joined_system}],
        }

    return contents, system_instruction


def _translate_tools_to_gemini(tools: Any) -> List[Dict[str, Any]]:
    """OpenAI tools[] -> Gemini tools[].functionDeclarations[]."""
    if not isinstance(tools, list) or not tools:
        return []
    declarations: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        decl = {"name": str(name)}
        if fn.get("description"):
            decl["description"] = str(fn["description"])
        params = fn.get("parameters")
        if isinstance(params, dict):
            decl["parameters"] = params
        declarations.append(decl)
    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


def _translate_tool_choice_to_gemini(tool_choice: Any) -> Optional[Dict[str, Any]]:
    """OpenAI tool_choice -> Gemini toolConfig.functionCallingConfig."""
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
        if name:
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [str(name)],
                },
            }
    return None


def _normalize_thinking_config(config: Any) -> Optional[Dict[str, Any]]:
    """Accept thinkingBudget / thinkingLevel / includeThoughts (+ snake_case)."""
    if not isinstance(config, dict) or not config:
        return None
    budget = config.get("thinkingBudget", config.get("thinking_budget"))
    level = config.get("thinkingLevel", config.get("thinking_level"))
    include = config.get("includeThoughts", config.get("include_thoughts"))
    normalized: Dict[str, Any] = {}
    if isinstance(budget, (int, float)):
        normalized["thinkingBudget"] = int(budget)
    if isinstance(level, str) and level.strip():
        normalized["thinkingLevel"] = level.strip().lower()
    if isinstance(include, bool):
        normalized["includeThoughts"] = include
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
    """Build the inner Gemini request body (goes inside ``request`` wrapper)."""
    contents, system_instruction = _build_gemini_contents(messages)

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction is not None:
        body["systemInstruction"] = system_instruction

    gemini_tools = _translate_tools_to_gemini(tools)
    if gemini_tools:
        body["tools"] = gemini_tools
    tool_cfg = _translate_tool_choice_to_gemini(tool_choice)
    if tool_cfg is not None:
        body["toolConfig"] = tool_cfg

    generation_config: Dict[str, Any] = {}
    if isinstance(temperature, (int, float)):
        generation_config["temperature"] = float(temperature)
    if isinstance(max_tokens, int) and max_tokens > 0:
        generation_config["maxOutputTokens"] = max_tokens
    if isinstance(top_p, (int, float)):
        generation_config["topP"] = float(top_p)
    if isinstance(stop, str) and stop:
        generation_config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        generation_config["stopSequences"] = [str(s) for s in stop if s]
    normalized_thinking = _normalize_thinking_config(thinking_config)
    if normalized_thinking:
        generation_config["thinkingConfig"] = normalized_thinking
    if generation_config:
        body["generationConfig"] = generation_config

    return body


def wrap_code_assist_request(
    *,
    project_id: str,
    model: str,
    inner_request: Dict[str, Any],
    user_prompt_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap the inner Gemini request in the Code Assist envelope."""
    return {
        "project": project_id,
        "model": model,
        "user_prompt_id": user_prompt_id or str(uuid.uuid4()),
        "request": inner_request,
    }


# =============================================================================
# Response translation: Gemini → OpenAI
# =============================================================================

def _translate_gemini_response(
    resp: Dict[str, Any],
    model: str,
) -> SimpleNamespace:
    """Non-streaming Gemini response -> OpenAI-shaped SimpleNamespace.

    Code Assist wraps the actual Gemini response inside ``response``, so we
    unwrap it first if present.
    """
    inner = resp.get("response") if isinstance(resp.get("response"), dict) else resp

    candidates = inner.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return _empty_response(model)

    cand = candidates[0]
    content_obj = cand.get("content") if isinstance(cand, dict) else {}
    parts = content_obj.get("parts") if isinstance(content_obj, dict) else []

    text_pieces: List[str] = []
    reasoning_pieces: List[str] = []
    tool_calls: List[SimpleNamespace] = []

    for i, part in enumerate(parts or []):
        if not isinstance(part, dict):
            continue
        # Thought parts are model's internal reasoning — surface as reasoning,
        # don't mix into content.
        if part.get("thought") is True:
            if isinstance(part.get("text"), str):
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
            tool_calls.append(SimpleNamespace(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                index=i,
                function=SimpleNamespace(name=str(fc["name"]), arguments=args_str),
            ))

    finish_reason = "tool_calls" if tool_calls else _map_gemini_finish_reason(
        str(cand.get("finishReason") or "")
    )

    usage_meta = inner.get("usageMetadata") or {}
    usage = SimpleNamespace(
        prompt_tokens=int(usage_meta.get("promptTokenCount") or 0),
        completion_tokens=int(usage_meta.get("candidatesTokenCount") or 0),
        total_tokens=int(usage_meta.get("totalTokenCount") or 0),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=int(usage_meta.get("cachedContentTokenCount") or 0),
        ),
    )

    message = SimpleNamespace(
        role="assistant",
        content="".join(text_pieces) if text_pieces else None,
        tool_calls=tool_calls or None,
        reasoning="".join(reasoning_pieces) or None,
        reasoning_content="".join(reasoning_pieces) or None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


def _empty_response(model: str) -> SimpleNamespace:
    message = SimpleNamespace(
        role="assistant", content="", tool_calls=None,
        reasoning=None, reasoning_content=None, reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
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


def _map_gemini_finish_reason(reason: str) -> str:
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "stop",
    }
    return mapping.get(reason.upper(), "stop")


# =============================================================================
# Streaming SSE iterator
# =============================================================================

class _GeminiStreamChunk(SimpleNamespace):
    """Mimics an OpenAI ChatCompletionChunk with .choices[0].delta."""
    pass


def _make_stream_chunk(
    *,
    model: str,
    content: str = "",
    tool_call_delta: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
    reasoning: str = "",
) -> _GeminiStreamChunk:
    delta_kwargs: Dict[str, Any] = {"role": "assistant"}
    if content:
        delta_kwargs["content"] = content
    if tool_call_delta is not None:
        delta_kwargs["tool_calls"] = [SimpleNamespace(
            index=tool_call_delta.get("index", 0),
            id=tool_call_delta.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function=SimpleNamespace(
                name=tool_call_delta.get("name") or "",
                arguments=tool_call_delta.get("arguments") or "",
            ),
        )]
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
    """Parse Server-Sent Events from an httpx streaming response."""
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
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON SSE line: %s", data[:200])


def _translate_stream_event(
    event: Dict[str, Any],
    model: str,
    tool_call_indices: Dict[str, int],
) -> List[_GeminiStreamChunk]:
    """Unwrap Code Assist envelope and emit OpenAI-shaped chunk(s)."""
    inner = event.get("response") if isinstance(event.get("response"), dict) else event
    candidates = inner.get("candidates") or []
    if not candidates:
        return []
    cand = candidates[0]
    if not isinstance(cand, dict):
        return []

    chunks: List[_GeminiStreamChunk] = []

    content = cand.get("content") or {}
    parts = content.get("parts") if isinstance(content, dict) else []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True and isinstance(part.get("text"), str):
            chunks.append(_make_stream_chunk(
                model=model, reasoning=part["text"],
            ))
            continue
        if isinstance(part.get("text"), str) and part["text"]:
            chunks.append(_make_stream_chunk(model=model, content=part["text"]))
        fc = part.get("functionCall")
        if isinstance(fc, dict) and fc.get("name"):
            name = str(fc["name"])
            idx = tool_call_indices.setdefault(name, len(tool_call_indices))
            try:
                args_str = json.dumps(fc.get("args") or {}, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = "{}"
            chunks.append(_make_stream_chunk(
                model=model,
                tool_call_delta={
                    "index": idx,
                    "name": name,
                    "arguments": args_str,
                },
            ))

    finish_reason_raw = str(cand.get("finishReason") or "")
    if finish_reason_raw:
        mapped = _map_gemini_finish_reason(finish_reason_raw)
        if tool_call_indices:
            mapped = "tool_calls"
        chunks.append(_make_stream_chunk(model=model, finish_reason=mapped))
    return chunks


# =============================================================================
# GeminiCloudCodeClient — OpenAI-compatible facade
# =============================================================================

MARKER_BASE_URL = "cloudcode-pa://google"


class _GeminiChatCompletions:
    def __init__(self, client: "GeminiCloudCodeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _GeminiChatNamespace:
    def __init__(self, client: "GeminiCloudCodeClient"):
        self.completions = _GeminiChatCompletions(client)


class GeminiCloudCodeClient:
    """Minimal OpenAI-SDK-compatible facade over Code Assist v1internal."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        project_id: str = "",
        **_: Any,
    ):
        # `api_key` here is a dummy — real auth is the OAuth access token
        # fetched on every call via agent.google_oauth.get_valid_access_token().
        # We accept the kwarg for openai.OpenAI interface parity.
        self.api_key = api_key or "google-oauth"
        self.base_url = base_url or MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._configured_project_id = project_id
        self._project_context: Optional[ProjectContext] = None
        self._project_context_lock = False  # simple single-thread guard
        self.chat = _GeminiChatNamespace(self)
        self.is_closed = False
        self._http = httpx.Client(timeout=httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0))

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    # Implement the OpenAI SDK's context-manager-ish closure check
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_project_context(self, access_token: str, model: str) -> ProjectContext:
        """Lazily resolve and cache the project context for this client."""
        if self._project_context is not None:
            return self._project_context

        env_project = google_oauth.resolve_project_id_from_env()
        creds = google_oauth.load_credentials()
        stored_project = creds.project_id if creds else ""

        # Prefer what's already baked into the creds
        if stored_project:
            self._project_context = ProjectContext(
                project_id=stored_project,
                managed_project_id=creds.managed_project_id if creds else "",
                tier_id="",
                source="stored",
            )
            return self._project_context

        ctx = resolve_project_context(
            access_token,
            configured_project_id=self._configured_project_id,
            env_project_id=env_project,
            user_agent_model=model,
        )
        # Persist discovered project back to the creds file so the next
        # session doesn't re-run the discovery.
        if ctx.project_id or ctx.managed_project_id:
            google_oauth.update_project_ids(
                project_id=ctx.project_id,
                managed_project_id=ctx.managed_project_id,
            )
        self._project_context = ctx
        return ctx

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
        access_token = google_oauth.get_valid_access_token()
        ctx = self._ensure_project_context(access_token, model)

        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = extra_body.get("thinking_config") or extra_body.get("thinkingConfig")

        inner = build_gemini_request(
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking_config=thinking_config,
        )
        wrapped = wrap_code_assist_request(
            project_id=ctx.project_id,
            model=model,
            inner_request=inner,
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "hermes-agent (gemini-cli-compat)",
            "X-Goog-Api-Client": "gl-python/hermes",
            "x-activity-request-id": str(uuid.uuid4()),
        }
        headers.update(self._default_headers)

        if stream:
            return self._stream_completion(model=model, wrapped=wrapped, headers=headers)

        url = f"{CODE_ASSIST_ENDPOINT}/v1internal:generateContent"
        response = self._http.post(url, json=wrapped, headers=headers)
        if response.status_code != 200:
            raise _gemini_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise CodeAssistError(
                f"Invalid JSON from Code Assist: {exc}",
                code="code_assist_invalid_json",
            ) from exc
        return _translate_gemini_response(payload, model=model)

    def _stream_completion(
        self,
        *,
        model: str,
        wrapped: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Iterator[_GeminiStreamChunk]:
        """Generator that yields OpenAI-shaped streaming chunks."""
        url = f"{CODE_ASSIST_ENDPOINT}/v1internal:streamGenerateContent?alt=sse"
        stream_headers = dict(headers)
        stream_headers["Accept"] = "text/event-stream"

        def _generator() -> Iterator[_GeminiStreamChunk]:
            try:
                with self._http.stream("POST", url, json=wrapped, headers=stream_headers) as response:
                    if response.status_code != 200:
                        # Materialize error body for better diagnostics
                        response.read()
                        raise _gemini_http_error(response)
                    tool_call_indices: Dict[str, int] = {}
                    for event in _iter_sse_events(response):
                        for chunk in _translate_stream_event(event, model, tool_call_indices):
                            yield chunk
            except httpx.HTTPError as exc:
                raise CodeAssistError(
                    f"Streaming request failed: {exc}",
                    code="code_assist_stream_error",
                ) from exc

        return _generator()


def _gemini_http_error(response: httpx.Response) -> CodeAssistError:
    status = response.status_code
    try:
        body = response.text[:500]
    except Exception:
        body = ""
    # Let run_agent's retry logic see auth errors as rotatable via `api_key`
    code = f"code_assist_http_{status}"
    if status == 401:
        code = "code_assist_unauthorized"
    elif status == 429:
        code = "code_assist_rate_limited"
    return CodeAssistError(
        f"Code Assist returned HTTP {status}: {body}",
        code=code,
    )
