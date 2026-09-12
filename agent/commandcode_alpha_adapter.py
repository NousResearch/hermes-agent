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
import time
import urllib.error
import urllib.request
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.commandcode_alpha")

COMMANDCODE_GENERATE_URL = "https://api.commandcode.ai/alpha/generate"


def _format_messages_for_commandcode(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Extract system prompt and convert messages to Command Code /alpha/generate wire format."""
    system_parts: List[str] = []
    wire_msgs: List[Dict[str, Any]] = []

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
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            wire_msgs.append({
                "role": "user",
                "content": [{"type": "text", "text": text}],
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
        "environment": "linux",
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

    system_prompt, wire_msgs = _format_messages_for_commandcode(messages)
    wire_tools = _format_tools_for_commandcode(tools)

    body = {
        "config": _workspace_config(),
        "memory": "",
        "taste": None,
        "skills": None,
        "permissionMode": "standard",
        "mode": "agent",
        "params": {
            "model": model,
            "messages": wire_msgs,
            "tools": wire_tools,
            "system": system_prompt,
            "max_tokens": max_tokens,
            "stream": True,
        },
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
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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
                    raw_usage = event.get("totalUsage") or event.get("usage") or {}
                    in_tok = raw_usage.get("inputTokens") or raw_usage.get("prompt_tokens") or 0
                    out_tok = raw_usage.get("outputTokens") or raw_usage.get("completion_tokens") or 0
                    usage_info["prompt_tokens"] = in_tok
                    usage_info["completion_tokens"] = out_tok
                    usage_info["total_tokens"] = in_tok + out_tok
                    if not tool_calls and event.get("finishReason"):
                        finish_reason = event.get("finishReason")
                elif ev_type == "error":
                    err_obj = event.get("error")
                    if isinstance(err_obj, dict):
                        err_msg = err_obj.get("message") or err_obj.get("type") or "Command Code generation error"
                    else:
                        err_msg = event.get("message") or str(err_obj) or "Command Code generation error"
                    raise RuntimeError(f"Command Code stream error: {err_msg}")
    except urllib.error.HTTPError as exc:
        body_text = ""
        with contextlib.suppress(Exception):
            body_text = exc.read().decode("utf-8")
        raise RuntimeError(f"Command Code HTTP {exc.code}: {body_text or exc.reason}") from exc

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

    return SimpleNamespace(
        id=f"cmdcode-{uuid.uuid4().hex[:12]}",
        choices=[choice_obj],
        usage=usage_obj,
        model=model,
    )
