"""Command Code Alpha Adapter.

Translates Hermes agent chat completion requests into Command Code's proprietary
/alpha/generate NDJSON streaming protocol (used by Command Code CLI and OpenCodex for
browser OAuth and local CLI account authentication).
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import platform
import time
import urllib.error
import urllib.request
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.commandcode_alpha")

COMMANDCODE_GENERATE_URL = "https://api.commandcode.ai/alpha/generate"

# The wire protocol reports failures as an ``error`` event *inside a 200 stream*, and the
# status lives at the top level (``{"type":"error","statusCode":402,...}``). Carrying it on
# the exception is what lets ``agent.error_classifier`` map 402→billing, 401→auth,
# 429→rate_limit through the normal pipeline instead of a generic rejection.
class CommandCodeAPIError(RuntimeError):
    """Command Code transport/stream failure with an attached HTTP status code."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# The vendor registry marks these text-only: pixels sent to them are dropped server-side,
# so an attached image becomes a text placeholder instead of vanishing. Unknown ids stay
# image-capable (the registry's own fallback), which is why this is a deny-list.
_TEXT_ONLY_MODELS = frozenset({
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash-fast",
    "zai-org/GLM-5.3", "zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast", "zai-org/GLM-5.1", "zai-org/GLM-5",
    "MiniMaxAI/MiniMax-M2.7", "minimax/minimax-m2.7-free", "MiniMaxAI/MiniMax-M2.5",
    "xiaomi/mimo-v2.5-pro", "Qwen/Qwen3.6-Max-Preview", "Qwen/Qwen3.7-Max",
    "meituan/LongCat-2.0:free", "stepfun/Step-3.5-Flash", "tencent/hy4-preview", "tencent/Hy3",
    "tencent/hy3-paid", "nvidia/nemotron-3-ultra-550b-a55b", "poolside/laguna-s-2.1-free",
    "inclusionai/ling-3.0-flash-free", "inclusionai/ling-3.0-flash-sante:free",
})


def _media_url(part: Dict[str, Any]) -> tuple[str, str]:
    """``(kind, url)`` for any shape a caller sends — flat ``image``, OpenAI's nested
    ``image_url``, or an Anthropic ``source`` block."""
    ptype = str(part.get("type", ""))
    inner = part.get("image_url") or part.get("video_url") or {}
    is_video = "video" in ptype
    is_media = is_video or "image" in ptype or bool(inner) or bool(part.get("source")) or bool(part.get("image"))
    if not is_media:
        # Unknown non-text part (file, audio, …): do not misrepresent it as an image.
        return "", ""
    kind = "video" if is_video else "image"
    url = ""
    if isinstance(inner, dict):
        url = str(inner.get("url", "") or "")
    elif isinstance(inner, str):
        url = inner
    if not url:
        url = str(part.get("image") or part.get("url") or (part.get("source") or {}).get("data", "") or "")
        src = part.get("source") or {}
        if url and src.get("data"):
            media_type = src.get("media_type") or "image/png"
            url = f"data:{media_type};base64,{url}"
    return kind, url


_URL_EXT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _image_mime(url: str) -> str:
    """``image/png`` out of ``data:image/png;base64,…`` or an ``.png`` URL path; empty otherwise."""
    if url.startswith("data:") and "," in url:
        return url[5:].split(";", 1)[0].split(",", 1)[0]
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    for ext, mime in _URL_EXT_MIME.items():
        if path.endswith(ext):
            return mime
    return ""


def _normalize_media_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One non-text content part → the wire shape (``{"type": "image", "image": url}``).

    OpenAI's nested ``{"image_url": {"url": ...}}`` is rejected by the endpoint (400
    "expected string") and must be flattened. ``mimeType`` is what makes the endpoint
    actually process the pixels: the same part without it is accepted and silently
    ignored (verified against a solid-colour test image).
    """
    kind, url = _media_url(part)
    if not url:
        return None
    mime = str(part.get("mimeType") or part.get("mime_type") or "") or _image_mime(url)
    return {"type": kind, kind: url, "mimeType": mime} if mime else None


def _media_placeholder(part: Dict[str, Any]) -> str:
    kind, url = _media_url(part)
    return f"[{kind}: {_image_mime(url) or 'attached'}]"


def _format_messages_for_commandcode(
    messages: List[Dict[str, Any]], model: str = ""
) -> tuple[str, List[Dict[str, Any]]]:
    """Extract system prompt and convert messages to Command Code /alpha/generate wire format."""
    system_parts: List[str] = []
    wire_msgs: List[Dict[str, Any]] = []

    images_ok = (model or "").strip() not in _TEXT_ONLY_MODELS

    # Map assistant tool call IDs to names for tool results
    call_id_to_name: Dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
        elif role == "user":
            parts_user: List[Dict[str, Any]] = []
            if isinstance(content, str):
                if content:
                    parts_user.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            parts_user.append({"type": "text", "text": text})
                        continue
                    media = _normalize_media_part(part) if images_ok else None
                    if media is not None:
                        parts_user.append(media)
                    else:
                        p_kind, _p_url = _media_url(part)
                        if not p_kind:
                            # Unknown part type: skip rather than claim it was an image.
                            continue
                        # Keep the turn honest: the model should see that something was
                        # attached even when the endpoint would silently drop the pixels.
                        parts_user.append({"type": "text", "text": _media_placeholder(part)})
            if not parts_user:
                parts_user.append({"type": "text", "text": ""})
            wire_msgs.append({
                "role": "user",
                "content": parts_user,
            })
        elif role == "assistant":
            parts: List[Dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                call_id = tc.get("id") or str(uuid.uuid4())
                func = tc.get("function") or {}
                name = func.get("name", "tool")
                call_id_to_name[call_id] = name
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                parts.append({
                    "type": "tool-call",
                    "toolCallId": call_id,
                    "toolName": name,
                    "input": args,
                })
            if not parts:
                parts = [{"type": "text", "text": ""}]
            wire_msgs.append({
                "role": "assistant",
                "content": parts,
            })
        elif role == "tool":
            call_id = msg.get("tool_call_id") or ""
            tool_name = call_id_to_name.get(call_id, "tool")
            val = content if isinstance(content, str) else json.dumps(content)
            wire_msgs.append({
                "role": "tool",
                "content": [{
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "toolName": tool_name,
                    "output": {"type": "text", "value": val},
                }],
            })

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, wire_msgs


def _format_tools_for_commandcode(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool definitions to Command Code input_schema format."""
    wire_tools: List[Dict[str, Any]] = []
    if not tools:
        return wire_tools

    for tool in tools:
        func = tool.get("function", {}) if tool.get("type") == "function" else tool
        name = func.get("name")
        if not name:
            continue
        params = func.get("parameters") or {"type": "object", "properties": {}}
        wire_tools.append({
            "name": name,
            "description": func.get("description", ""),
            "input_schema": params,
        })
    return wire_tools


def _workspace_config(cwd: Optional[str] = None) -> Dict[str, Any]:
    working_dir = cwd or os.getcwd()
    return {
        "workingDir": working_dir,
        "isGitRepo": False,
        "currentBranch": "",
        "mainBranch": "",
        "gitStatus": "",
        "recentCommits": [],
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        # The official client reports the real host; a hardcoded "linux" is wrong on
        # macOS/Windows and can steer the model's shell assumptions.
        "environment": f"{platform.system().lower()}-{platform.machine()}, Python {platform.python_version()}",
        "structure": [],
    }


# Mapping from dashed slugs to canonical wire model IDs accepted by /alpha/generate
COMMAND_CODE_MODEL_ALIASES: Dict[str, str] = {
    "deepseek-deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "deepseek-deepseek-v4-flash-vision-exp": "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek-deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "meituan-LongCat-2.0:free": "meituan/LongCat-2.0:free",
    "meta-muse-spark-1.3-contributor": "meta/muse-spark-1.3-contributor",
    "MiniMaxAI-MiniMax-M3": "MiniMaxAI/MiniMax-M3",
    "moonshotai-Kimi-K3": "moonshotai/Kimi-K3",
    "poolside-laguna-s-2.1-free": "poolside/laguna-s-2.1-free",
    "Qwen-Qwen3.8-Max-0902": "Qwen/Qwen3.8-Max-0902",
    "xai-grok-4.5": "xai/grok-4.5",
    "xiaomi-mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
    "z-ai-glm-5.3-flash": "z-ai/glm-5.3-flash",
}


def canonical_commandcode_model_id(model_id: str) -> str:
    """Normalize model ID so dashed slugs or prefixed names map to canonical wire format."""
    clean = model_id.strip()
    if clean.startswith("command-code/"):
        clean = clean[len("command-code/"):]
    elif clean.startswith("commandcode-oauth/"):
        clean = clean[len("commandcode-oauth/"):]
    return COMMAND_CODE_MODEL_ALIASES.get(clean, clean)


def stream_commandcode_alpha(agent: Any, api_kwargs: Dict[str, Any], on_first_delta: Any = None) -> Any:
    """Stream a completion via Command Code /alpha/generate and return standard response."""
    raw_model = api_kwargs.get("model") or "meituan/LongCat-2.0:free"
    model = canonical_commandcode_model_id(raw_model)
    messages = api_kwargs.get("messages") or []
    tools = api_kwargs.get("tools")
    max_tokens = api_kwargs.get("max_tokens") or 4096

    token = getattr(agent, "api_key", "")
    if not token:
        from hermes_cli.auth_commandcode import _read_commandcode_cli_tokens
        with contextlib.suppress(Exception):
            token = _read_commandcode_cli_tokens().get("apiKey", "")

    if not token:
        raise RuntimeError("Command Code OAuth access token missing. Run 'hermes auth add commandcode-oauth'.")

    system_prompt, wire_msgs = _format_messages_for_commandcode(messages, model)
    if api_kwargs.get("tool_choice") == "none":
        # Honour "no tools": forwarding them anyway lets the model call something the
        # caller explicitly ruled out.
        tools = None
    elif isinstance(api_kwargs.get("tool_choice"), dict) or api_kwargs.get("tool_choice") == "required":
        # The wire protocol has no forced-tool form; forward the tools and let the model
        # choose rather than failing the turn.
        logger.debug("commandcode: tool_choice=%r is not expressible on /alpha/generate; forwarding tools", api_kwargs.get("tool_choice"))
    wire_tools = _format_tools_for_commandcode(tools)

    params: Dict[str, Any] = {
        "model": model,
        "messages": wire_msgs,
        "tools": wire_tools,
        "system": system_prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }
    temperature = api_kwargs.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        params["temperature"] = temperature

    body = {
        "config": _workspace_config(),
        "memory": "",
        "taste": None,
        "skills": None,
        "permissionMode": "standard",
        "mode": "agent",
        "params": params,
    }

    req = urllib.request.Request(COMMANDCODE_GENERATE_URL)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "cli")
    req.add_header("x-command-code-version", "0.52.1")
    req.add_header("x-cli-environment", "production")
    req.add_header("x-taste-learning", "false")
    req.add_header("x-co-flag", "false")
    req.add_header("x-session-id", str(uuid.uuid4()))
    req.data = json.dumps(body).encode("utf-8")

    first_delta_fired = False

    def _fire_first():
        nonlocal first_delta_fired
        if not first_delta_fired and on_first_delta:
            first_delta_fired = True
            with contextlib.suppress(Exception):
                on_first_delta()

    content_accum: List[str] = []
    reasoning_accum: List[str] = []
    tool_calls: List[Any] = []
    finish_reason = "stop"
    saw_finish = False
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cache_info = {"cached_tokens": 0, "cache_write_tokens": 0}

    try:
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                ev_type = event.get("type")
                if ev_type == "text-delta":
                    text = event.get("text", "")
                    if text:
                        _fire_first()
                        content_accum.append(text)
                        cb = getattr(agent, "stream_callback", None)
                        if cb:
                            with contextlib.suppress(Exception):
                                cb(text)
                elif ev_type == "reasoning-delta":
                    r_text = event.get("text", "")
                    if r_text:
                        _fire_first()
                        reasoning_accum.append(r_text)
                        t_cb = getattr(agent, "thinking_callback", None)
                        if t_cb:
                            with contextlib.suppress(Exception):
                                t_cb(r_text)
                elif ev_type == "tool-call":
                    _fire_first()
                    call_id = event.get("toolCallId") or str(uuid.uuid4())
                    tool_name = event.get("toolName", "tool")
                    inp = event.get("input", {})
                    args_str = inp if isinstance(inp, str) else json.dumps(inp)
                    tool_calls.append(SimpleNamespace(
                        id=call_id,
                        type="function",
                        function=SimpleNamespace(name=tool_name, arguments=args_str),
                    ))
                    finish_reason = "tool_calls"
                elif ev_type in ("finish", "finish-step"):
                    saw_finish = True
                    raw_usage = event.get("totalUsage") or event.get("usage") or {}
                    if raw_usage:
                        in_tok = raw_usage.get("inputTokens") or raw_usage.get("prompt_tokens") or 0
                        out_tok = raw_usage.get("outputTokens") or raw_usage.get("completion_tokens") or 0
                        usage_info["prompt_tokens"] = in_tok
                        usage_info["completion_tokens"] = out_tok
                        usage_info["total_tokens"] = in_tok + out_tok
                        # The endpoint serves most of the prefix from prompt cache; without
                        # this mapping the caller sees zero cache hits despite the discount.
                        details = raw_usage.get("inputTokenDetails") or {}
                        cache_read = (
                            details.get("cacheReadTokens")
                            or raw_usage.get("cacheReadTokens")
                            or raw_usage.get("cachedInputTokens")
                            or 0
                        )
                        cache_write = details.get("cacheWriteTokens") or raw_usage.get("cacheWriteTokens") or 0
                        # Keep the richest counts seen: a later empty finish must not
                        # zero a finish-step's totals.
                        if cache_read or cache_write:
                            cache_info["cached_tokens"] = max(cache_info["cached_tokens"], cache_read)
                            cache_info["cache_write_tokens"] = max(cache_info["cache_write_tokens"], cache_write)
                    if not tool_calls and event.get("finishReason"):
                        finish_reason = event.get("finishReason")
                elif ev_type == "error":
                    # The status is at the TOP level of the event, not inside error.
                    status = event.get("statusCode")
                    status = int(status) if isinstance(status, (int, str)) and str(status).isdigit() else None
                    err_obj = event.get("error")
                    if isinstance(err_obj, dict):
                        err_msg = err_obj.get("message") or err_obj.get("type") or "Command Code generation error"
                    else:
                        err_msg = event.get("message") or str(err_obj) or "Command Code generation error"
                    raise CommandCodeAPIError(
                        f"Command Code stream error (HTTP {status}): {err_msg}" if status else f"Command Code stream error: {err_msg}",
                        status_code=status,
                    )
    except urllib.error.HTTPError as exc:
        body_text = ""
        with contextlib.suppress(Exception):
            body_text = exc.read().decode("utf-8")
        raise CommandCodeAPIError(
            f"Command Code HTTP {exc.code}: {body_text or exc.reason}", status_code=exc.code,
        ) from exc

    # Fail closed: a run that ends after `start` with no text, no reasoning that produced
    # output, and no tool call is an error — not an empty success the caller would replay.
    if not content_accum and not tool_calls:
        raise CommandCodeAPIError(
            f"Command Code stream ended with no output (saw_finish={saw_finish}); the request was not answered.",
            status_code=502,
        )

    full_content = "".join(content_accum) if content_accum else None
    full_reasoning = "".join(reasoning_accum) if reasoning_accum else None

    msg_obj = SimpleNamespace(
        role="assistant",
        content=full_content,
        tool_calls=tool_calls if tool_calls else None,
        reasoning_content=full_reasoning,
    )
    choice_obj = SimpleNamespace(
        index=0,
        message=msg_obj,
        finish_reason=finish_reason,
    )
    usage_obj = SimpleNamespace(
        prompt_tokens=usage_info["prompt_tokens"],
        completion_tokens=usage_info["completion_tokens"],
        total_tokens=usage_info["total_tokens"],
    )
    if cache_info["cached_tokens"] or cache_info["cache_write_tokens"]:
        usage_obj.prompt_tokens_details = SimpleNamespace(
            cached_tokens=cache_info["cached_tokens"],
            cache_write_tokens=cache_info["cache_write_tokens"],
        )

    return SimpleNamespace(
        id=f"cmdcode-{uuid.uuid4().hex[:12]}",
        choices=[choice_obj],
        usage=usage_obj,
        model=model,
    )
