"""Ollama native /api/chat adapter for Hermes Agent.

Provides integration with Ollama's native chat endpoint, following the same
adapter pattern as ``bedrock_adapter.py`` and ``anthropic_adapter.py``.

  - Converts Hermes/OpenAI-format messages to Ollama native format.
  - Converts OpenAI-format tool definitions to Ollama format.
  - Normalizes Ollama responses to OpenAI-compatible SimpleNamespace objects.
  - Supports both streaming (NDJSON) and non-streaming modes.
  - Provides graceful connection error handling for local Ollama instances.

Architecture:
  - All Ollama-specific logic is isolated in this module.
  - Messages/tools are converted between OpenAI format and Ollama format.
  - Responses are normalized back to OpenAI-compatible objects for the agent loop.

Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import logging
import os
from types import SimpleNamespace
from typing import Callable, Iterator, List, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger("ollama_adapter")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class OllamaAPIError(Exception):
    """Raised when the Ollama API returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OllamaAuthenticationError(Exception):
    """Raised when Ollama returns HTTP 401 (cloud/authenticated endpoints)."""

    def __init__(self, message: str = "Authentication failed (HTTP 401)"):
        super().__init__(message)
        self.message = message


class OllamaConnectionError(Exception):
    """Raised when the connection to Ollama is refused (Ollama not running)."""

    def __init__(self, message: str = "Cannot connect to Ollama — is it running?"):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def normalize_ollama_url(base_url: str) -> str:
    """Normalize a base URL to point at the Ollama /api/chat endpoint.

    Examples::

        normalize_ollama_url("http://localhost:11434")
        # -> "http://localhost:11434/api/chat"

        normalize_ollama_url("http://localhost:11434/api")
        # -> "http://localhost:11434/api/chat"

        normalize_ollama_url("http://localhost:11434/api/chat")
        # -> "http://localhost:11434/api/chat"
    """
    url = base_url.rstrip("/")
    if url.endswith("/api/chat"):
        return url
    if url.endswith("/api"):
        return url + "/chat"
    return url + "/api/chat"


# ---------------------------------------------------------------------------
# Message format conversion: OpenAI/Hermes → Ollama
# ---------------------------------------------------------------------------


def convert_messages_to_ollama(messages: list) -> list:
    """Convert Hermes/OpenAI-format messages to Ollama native /api/chat format.

    Handles:
      - System messages → ``{"role": "system", "content": "..."}``
      - User messages → ``{"role": "user", "content": "...", "images": [...]}``
      - Assistant messages with text → ``{"role": "assistant", "content": "..."}``
      - Assistant messages with tool calls → ``{"role": "assistant", "tool_calls": [...]}``
      - Tool result messages → ``{"role": "tool", "content": "..."}``

    Tool call ``function.arguments`` may arrive as a dict or a JSON string;
    this function always outputs a JSON string to match Ollama's expected format.
    """
    ollama_msgs = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                text = "\n".join(parts)
            ollama_msgs.append({"role": "system", "content": text})
            continue

        if role == "tool":
            # Tool result — Ollama uses role "tool" with content string
            result_content = content if isinstance(content, str) else json.dumps(content)
            entry: dict = {"role": "tool", "content": result_content}
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            ollama_msgs.append(entry)
            continue

        if role == "assistant":
            entry = {"role": "assistant"}

            # Handle tool calls
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                ollama_tool_calls = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    # Normalize arguments to a dict for Ollama
                    if isinstance(args, str):
                        try:
                            args_dict = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args_dict = {}
                    else:
                        args_dict = args if isinstance(args, dict) else {}
                    ollama_tool_calls.append({
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": args_dict,
                        }
                    })
                entry["tool_calls"] = ollama_tool_calls
                # Include content if both present
                if isinstance(content, str) and content.strip():
                    entry["content"] = content
                else:
                    entry["content"] = ""
            else:
                # Text-only assistant message
                if isinstance(content, str):
                    entry["content"] = content
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            parts.append(part)
                    entry["content"] = "\n".join(parts)
                else:
                    entry["content"] = ""

            ollama_msgs.append(entry)
            continue

        if role == "user":
            entry = {"role": "user"}

            # Extract images and text from content
            images = []
            text_parts = []

            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype == "text":
                            text_parts.append(part.get("text", ""))
                        elif ptype == "image_url":
                            # Extract base64 data or URL
                            image_url = part.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                            # Strip data URI prefix if present
                            if url.startswith("data:"):
                                # e.g. "data:image/png;base64,<data>"
                                try:
                                    b64_data = url.split(",", 1)[1]
                                    images.append(b64_data)
                                except IndexError:
                                    pass
                            else:
                                images.append(url)
                        elif ptype == "image":
                            # Anthropic-style image block
                            source = part.get("source", {})
                            if source.get("type") == "base64":
                                images.append(source.get("data", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
            elif content is not None:
                text_parts.append(str(content))

            entry["content"] = "\n".join(text_parts)
            # Also check top-level images field
            top_images = msg.get("images", [])
            if top_images:
                images.extend(top_images)
            if images:
                entry["images"] = images

            ollama_msgs.append(entry)
            continue

    return ollama_msgs


# ---------------------------------------------------------------------------
# Tool format conversion: OpenAI → Ollama
# ---------------------------------------------------------------------------


def convert_tools_to_ollama(tools: list) -> list:
    """Convert OpenAI-format tool definitions to Ollama format.

    Ollama's tool format is nearly identical to OpenAI's, but this function
    validates and cleans the input to ensure compatibility.

    Input format (OpenAI)::

        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]

    Output format (Ollama, same structure)::

        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    """
    ollama_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {})
        if not fn or not fn.get("name"):
            continue
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return ollama_tools


# ---------------------------------------------------------------------------
# Response normalization: Ollama → OpenAI-compatible SimpleNamespace
# ---------------------------------------------------------------------------


def normalize_ollama_response(data: dict) -> SimpleNamespace:
    """Convert an Ollama /api/chat JSON response to an OpenAI-compatible object.

    The agent loop in ``run_agent.py`` expects responses shaped like
    ``openai.ChatCompletion`` — this function bridges the gap.

    Returns a SimpleNamespace with:
      - ``.choices[0].message.content`` — text response (None if empty)
      - ``.choices[0].message.tool_calls`` — list of tool calls (None if absent)
      - ``.choices[0].message.reasoning`` — thinking/reasoning text (None if absent)
      - ``.choices[0].finish_reason`` — "stop" or "tool_calls"
      - ``.usage.prompt_tokens`` — input token count
      - ``.usage.completion_tokens`` — output token count
      - ``.model`` — model name from response
    """
    message = data.get("message", {})
    raw_content = message.get("content", "")
    content = raw_content if raw_content else None

    # Normalize tool calls — Ollama arguments may be dict or string
    raw_tool_calls = message.get("tool_calls") or []
    tool_calls = []
    for tc in raw_tool_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, dict):
            args_str = json.dumps(args)
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = json.dumps(args)
        tool_calls.append(SimpleNamespace(
            id=f"call_{uuid4().hex[:24]}",
            type="function",
            function=SimpleNamespace(
                name=fn.get("name", ""),
                arguments=args_str,
            ),
        ))

    reasoning = message.get("thinking") or None

    finish_reason = "tool_calls" if tool_calls else "stop"

    msg = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        reasoning=reasoning,
    )

    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    choice = SimpleNamespace(
        index=0,
        message=msg,
        finish_reason=finish_reason,
    )

    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=data.get("model", ""),
    )


# ---------------------------------------------------------------------------
# Request payload builder
# ---------------------------------------------------------------------------


def build_ollama_kwargs(
    model: str,
    messages: list,
    tools: Optional[list] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    options: Optional[dict] = None,
    keep_alive: Optional[str] = None,
    think: Optional[bool] = None,
) -> dict:
    """Build a request payload for Ollama /api/chat.

    Args:
        model: Ollama model name (e.g. "llama3.2", "qwen2.5-coder:7b").
        messages: Already-converted Ollama-format messages list.
        tools: Already-converted Ollama-format tools list (or None).
        max_tokens: Maps to ``options.num_predict`` in Ollama.
        temperature: Maps to ``options.temperature`` in Ollama.
        options: Raw Ollama options dict — merged with max_tokens/temperature.
            ``options.num_ctx`` is passed through for context window control.
        keep_alive: How long to keep the model loaded (e.g. "5m", "0", "-1").
        think: Whether to enable extended thinking (Ollama 0.9+ models).

    Returns:
        A dict ready to JSON-serialize and POST to /api/chat.
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    if tools:
        payload["tools"] = tools

    # Build options dict
    opts: dict = {}
    if options:
        opts.update(options)
    if max_tokens is not None:
        opts["num_predict"] = max_tokens
    if temperature is not None:
        opts["temperature"] = temperature
    if opts:
        payload["options"] = opts

    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    if think is not None:
        payload["think"] = think

    return payload


# ---------------------------------------------------------------------------
# HTTP layer — non-streaming
# ---------------------------------------------------------------------------


def call_ollama(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout: int = 300,
) -> dict:
    """Make a non-streaming POST request to Ollama /api/chat.

    Sets ``stream: false`` in the payload and returns the parsed JSON dict.

    Raises:
        OllamaConnectionError: If the connection is refused.
        OllamaAuthenticationError: On HTTP 401.
        OllamaAPIError: On other HTTP error responses.
    """
    url = normalize_ollama_url(base_url)
    payload = {**payload, "stream": False}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.ConnectError as exc:
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {base_url} — is it running?"
        ) from exc

    if resp.status_code == 401:
        raise OllamaAuthenticationError(
            f"Authentication failed (HTTP 401) for {url}"
        )
    if resp.status_code >= 400:
        raise OllamaAPIError(
            f"Ollama API error: HTTP {resp.status_code} — {resp.text}",
            status_code=resp.status_code,
        )

    return resp.json()


# ---------------------------------------------------------------------------
# HTTP layer — streaming
# ---------------------------------------------------------------------------


def call_ollama_stream(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout: int = 300,
):
    """Make a streaming POST request to Ollama /api/chat.

    Sets ``stream: true`` in the payload and returns the httpx streaming
    response. The caller is responsible for managing the response lifecycle
    (use as a context manager or call ``.aclose()``).

    Raises:
        OllamaConnectionError: If the connection is refused.
        OllamaAuthenticationError: On HTTP 401.
        OllamaAPIError: On other HTTP error responses.
    """
    url = normalize_ollama_url(base_url)
    payload = {**payload, "stream": True}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        client = httpx.Client(timeout=timeout)
        req = client.build_request("POST", url, json=payload, headers=headers)
        resp = client.send(req, stream=True)
    except httpx.ConnectError as exc:
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {base_url} — is it running?"
        ) from exc

    if resp.status_code == 401:
        resp.close()
        client.close()
        raise OllamaAuthenticationError(
            f"Authentication failed (HTTP 401) for {url}"
        )
    if resp.status_code >= 400:
        body = resp.text
        resp.close()
        client.close()
        raise OllamaAPIError(
            f"Ollama API error: HTTP {resp.status_code} — {body}",
            status_code=resp.status_code,
        )

    return resp


# ---------------------------------------------------------------------------
# Streaming response processor
# ---------------------------------------------------------------------------


def stream_ollama_with_callbacks(
    response_iter,
    on_text_delta: Optional[Callable[[str], None]] = None,
    on_tool_start: Optional[Callable[[str, dict], None]] = None,
    on_reasoning_delta: Optional[Callable[[str], None]] = None,
    on_interrupt_check: Optional[Callable[[], bool]] = None,
) -> SimpleNamespace:
    """Parse an Ollama NDJSON stream and fire real-time callbacks.

    Ollama streams newline-delimited JSON (NDJSON), not SSE. Each line is a
    complete JSON object. The final object has ``"done": true`` and carries
    token counts.

    Args:
        response_iter: An iterable of text lines (e.g. from
            ``httpx.Response.iter_lines()``).
        on_text_delta: Called with each text chunk as it arrives.
        on_tool_start: Called with ``(name, arguments_dict)`` for each tool
            call in a streaming chunk.
        on_reasoning_delta: Called with each thinking/reasoning chunk.
        on_interrupt_check: Called on each chunk; return True to stop early.

    Returns:
        An OpenAI-compatible SimpleNamespace response, identical in shape to
        ``normalize_ollama_response()``.
    """
    text_parts: list = []
    thinking_parts: list = []
    tool_calls: list = []
    prompt_tokens = 0
    completion_tokens = 0
    model_name = ""
    has_final_chunk = False

    for line in response_iter:
        if on_interrupt_check and on_interrupt_check():
            break

        if not line or not line.strip():
            continue

        try:
            chunk = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Skipping non-JSON line in Ollama stream: %r", line)
            continue

        if "error" in chunk:
            raise OllamaAPIError(str(chunk["error"]))

        message = chunk.get("message", {})

        # Text content delta
        text = message.get("content", "")
        if text:
            text_parts.append(text)
            if on_text_delta:
                on_text_delta(text)

        # Thinking/reasoning delta
        thinking = message.get("thinking", "")
        if thinking:
            thinking_parts.append(thinking)
            if on_reasoning_delta:
                on_reasoning_delta(thinking)

        # Tool calls in chunk
        chunk_tool_calls = message.get("tool_calls") or []
        for tc in chunk_tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args_dict = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args_dict = {}
            else:
                args_dict = args if isinstance(args, dict) else {}
            name = fn.get("name", "")
            tool_calls.append(SimpleNamespace(
                id=f"call_{uuid4().hex[:24]}",
                type="function",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args_dict),
                ),
            ))
            if on_tool_start:
                on_tool_start(name, args_dict)

        # Final chunk
        if chunk.get("done"):
            has_final_chunk = True
            model_name = chunk.get("model", "")
            prompt_tokens = chunk.get("prompt_eval_count", 0)
            completion_tokens = chunk.get("eval_count", 0)

    # Defaults if final chunk was missing
    if not has_final_chunk:
        logger.warning("Ollama stream ended without a 'done' chunk")

    full_text = "".join(text_parts) if text_parts else None
    full_thinking = "".join(thinking_parts) if thinking_parts else None
    finish_reason = "tool_calls" if tool_calls else "stop"

    msg = SimpleNamespace(
        role="assistant",
        content=full_text,
        tool_calls=tool_calls if tool_calls else None,
        reasoning=full_thinking,
    )

    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    choice = SimpleNamespace(
        index=0,
        message=msg,
        finish_reason=finish_reason,
    )

    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model_name,
    )


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


def list_ollama_models(base_url: str, api_key: str = "") -> list:
    """Fetch available models from the Ollama /api/tags endpoint.

    Returns a list of model dicts with normalized keys:
      - ``name``: model name (e.g. "llama3.2:latest")
      - ``size``: model size in bytes
      - ``parameter_size``: human-readable parameter count (e.g. "7B")
      - ``family``: model family (e.g. "llama")
      - ``quantization_level``: quantization (e.g. "Q4_K_M")

    Returns an empty list if Ollama is not reachable (graceful degradation).
    """
    url = base_url.rstrip("/")
    if url.endswith("/api/chat"):
        url = url[: -len("/chat")]
    elif not url.endswith("/api"):
        url = url + "/api"
    url = url + "/tags"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        logger.debug("Could not list Ollama models from %s: %s", url, exc)
        return []

    models = []
    for m in data.get("models", []):
        details = m.get("details", {})
        models.append({
            "name": m.get("name", ""),
            "size": m.get("size", 0),
            "parameter_size": details.get("parameter_size", ""),
            "family": details.get("family", ""),
            "quantization_level": details.get("quantization_level", ""),
        })
    models.sort(key=lambda m: m.get("name", ""))
    return models
