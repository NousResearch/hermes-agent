#!/usr/bin/env python3
"""Local Anthropic-compatible Messages API backed by Hermes OpenAI-Codex pool.

Binds only to 127.0.0.1. Does not log or print credentials.
"""
from __future__ import annotations

import json
import hashlib
import os
import queue
import sys
import threading
import time
import uuid
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Tuple

HERMES_REPO = Path(os.environ.get("HERMES_AGENT_REPO", "/Users/wangrenzhu/work/hermes-agent"))
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

CONFIG_PATH = Path(os.environ.get("CODEX_ANTHROPIC_BRIDGE_CONFIG", str(Path.home() / ".hermes/cc-switch-codex-bridge/config.json")))
DEFAULT_MODEL = os.environ.get("CODEX_ANTHROPIC_BRIDGE_MODEL", "gpt-5.5")
# Claude Code can otherwise wait for a single stuck Codex/Codex-pool response for
# many minutes. Keep each model attempt bounded, then retry with a freshly
# selected pool entry (round_robin in ~/.hermes/config.yaml).
DEFAULT_TIMEOUT = float(os.environ.get("CODEX_ANTHROPIC_BRIDGE_TIMEOUT", "120"))
# Claude Code may send a much larger per-request timeout than we want for the
# local Codex pool. Cap each backend attempt so slow/stuck pool entries fail
# fast and the retry loop can select a fresh credential/route.
MAX_REQUEST_TIMEOUT = float(os.environ.get("CODEX_ANTHROPIC_BRIDGE_MAX_REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT)))
MAX_RETRIES = int(os.environ.get("CODEX_ANTHROPIC_BRIDGE_MAX_RETRIES", "2"))
RETRY_BACKOFF = float(os.environ.get("CODEX_ANTHROPIC_BRIDGE_RETRY_BACKOFF", "0.75"))
STREAM_PING_INTERVAL = float(os.environ.get("CODEX_ANTHROPIC_BRIDGE_STREAM_PING_INTERVAL", "5"))
DEFAULT_CONTEXT_WINDOW_TOKENS = int(os.environ.get("CODEX_ANTHROPIC_BRIDGE_CONTEXT_WINDOW_TOKENS", "272000"))
PROTOCOL_LOG = os.environ.get("CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG", "0").lower() in {"1", "true", "yes", "on"}
PROTOCOL_LOG_FILE = os.environ.get("CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG_FILE", "").strip()
THINKING_SIGNATURE_PLACEHOLDER = os.environ.get(
    "CODEX_ANTHROPIC_BRIDGE_THINKING_SIGNATURE_PLACEHOLDER", "1"
).lower() in {"1", "true", "yes", "on"}
# The Codex/OpenAI backend used by this bridge currently rejects chat messages
# with role="tool". Claude Code sends Anthropic tool_result blocks on user
# messages after Read/Write/Bash/Workflow tool use. Default to flattening prior
# tool_use/tool_result history into ordinary text transcript markers so the
# next model turn can continue without sending backend-rejected tool roles.
STRUCTURED_TOOL_HISTORY = os.environ.get(
    "CODEX_ANTHROPIC_BRIDGE_STRUCTURED_TOOL_HISTORY", "0"
).lower() in {"1", "true", "yes", "on"}
DEFAULT_MODEL_ALIASES = {
    # Claude Code / subagent-friendly aliases.  These keep the backend model
    # explicit while letting Claude choose an appropriate depth/cost profile.
    "gpt-5.5": {"model": "gpt-5.5", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "gpt-5.5-xhigh": {"model": "gpt-5.5", "effort": "xhigh", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "gpt-5.5-high": {"model": "gpt-5.5", "effort": "high", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "gpt-5.5-medium": {"model": "gpt-5.5", "effort": "medium", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "gpt-5.5-low": {"model": "gpt-5.5", "effort": "low", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "gpt-5.5-fast": {"model": "gpt-5.5", "effort": "low", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "gpt-5.5-mini": {"model": "gpt-5.5", "effort": "low", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    # Backward/forward compatible names for workflows that ask for smaller
    # Codex models directly.  If the account pool rejects one, the request
    # fails honestly instead of silently upgrading to 5.5.
    "gpt-5.4": {"model": "gpt-5.4"},
    "gpt-5.4-xhigh": {"model": "gpt-5.4", "effort": "xhigh"},
    "gpt-5.4-high": {"model": "gpt-5.4", "effort": "high"},
    "gpt-5.4-medium": {"model": "gpt-5.4", "effort": "medium"},
    "gpt-5.4-low": {"model": "gpt-5.4", "effort": "low"},
    # Claude Code native / Dynamic Workflow aliases.  Keep the requested
    # Anthropic-looking model id in responses so Claude Code can preserve its
    # native model-routing and thinking UX, while resolving locally to the
    # Hermes OpenAI-Codex credential pool.
    "claude-haiku-4-8": {"model": "gpt-5.5", "effort": "high", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "claude-haiku-4-8-latest": {"model": "gpt-5.5", "effort": "high", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "claude-sonnet-4-8": {"model": "gpt-5.5", "effort": "xhigh", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "claude-sonnet-4-8-latest": {"model": "gpt-5.5", "effort": "xhigh", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "claude-opus-4-8": {"model": "gpt-5.5", "effort": "xhigh", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "claude-opus-4-8-latest": {"model": "gpt-5.5", "effort": "xhigh", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    # Compatibility aliases kept for older Claude Code settings/workflows.
    "claude-sonnet-4-5": {"model": "gpt-5.5", "effort": "high", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "claude-sonnet-4-5-latest": {"model": "gpt-5.5", "effort": "high", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "claude-opus-4-1": {"model": "gpt-5.5", "effort": "xhigh", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
    "claude-opus-4-1-latest": {"model": "gpt-5.5", "effort": "xhigh", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS},
}

BRIDGE_TOOL_CONTINUATION_INSTRUCTION = (
    "Local bridge instruction: when user content includes completed tool "
    "results, continue the requested task until it is complete. Do not stop "
    "merely because a tool returned. If another tool is needed, call it; "
    "otherwise summarize the result."
)
THINKING_SIGNATURE = "codex-bridge-placeholder:not-anthropic-origin"


def _load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}


def _model_aliases() -> Dict[str, Dict[str, str]]:
    cfg = _load_config()
    aliases: Dict[str, Dict[str, str]] = dict(DEFAULT_MODEL_ALIASES)
    configured = cfg.get("model_aliases") if isinstance(cfg, dict) else None
    if isinstance(configured, dict):
        for name, value in configured.items():
            if isinstance(value, str):
                aliases[str(name)] = {"model": value}
            elif isinstance(value, dict):
                model = str(value.get("model") or name)
                effort = value.get("effort")
                entry = {"model": model}
                if effort:
                    entry["effort"] = str(effort)
                context_window = value.get("context_window") or value.get("context_window_tokens")
                if context_window is not None:
                    try:
                        entry["context_window"] = int(context_window)
                    except Exception:
                        pass
                aliases[str(name)] = entry
    return aliases


def _supported_model_ids() -> List[str]:
    cfg = _load_config()
    configured = cfg.get("models") if isinstance(cfg, dict) else None
    ids = list(_model_aliases().keys())
    if isinstance(configured, list) and configured:
        ids = [str(x) for x in configured if str(x).strip()] + ids
    if DEFAULT_MODEL not in ids:
        ids.insert(0, DEFAULT_MODEL)
    seen: set[str] = set()
    return [mid for mid in ids if mid and not (mid in seen or seen.add(mid))]


def _requested_thinking_effort(payload: Dict[str, Any]) -> str | None:
    thinking = payload.get("thinking")
    if isinstance(thinking, dict):
        budget = thinking.get("budget_tokens")
        try:
            budget_i = int(budget) if budget is not None else 0
        except Exception:
            budget_i = 0
        if budget_i >= 16000:
            return "high"
        if budget_i > 0:
            return "medium"
    return None


def _resolve_model_and_effort(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    requested = str(payload.get("model") or DEFAULT_MODEL)
    aliases = _model_aliases()
    alias = aliases.get(requested)
    if alias is None and requested.startswith("claude"):
        alias = {"model": DEFAULT_MODEL, "effort": "high"}
    if alias is None:
        alias = {"model": requested}
    model = str(alias.get("model") or DEFAULT_MODEL)
    effort = (
        _requested_thinking_effort(payload)
        or str(alias.get("effort") or "")
        or str(os.environ.get("CODEX_ANTHROPIC_BRIDGE_EFFORT", "medium"))
    )
    if effort == "minimal":
        effort = "low"
    if effort not in {"low", "medium", "high", "xhigh"}:
        effort = "medium"
    return requested, model, effort



def _context_window_for_payload(payload: Dict[str, Any]) -> int:
    requested = str(payload.get("model") or DEFAULT_MODEL)
    alias = _model_aliases().get(requested)
    if alias is None and requested.startswith("claude"):
        alias = {"model": DEFAULT_MODEL, "effort": "high", "context_window": DEFAULT_CONTEXT_WINDOW_TOKENS}
    if isinstance(alias, dict):
        try:
            return int(alias.get("context_window") or DEFAULT_CONTEXT_WINDOW_TOKENS)
        except Exception:
            return DEFAULT_CONTEXT_WINDOW_TOKENS
    return DEFAULT_CONTEXT_WINDOW_TOKENS


def _context_windows() -> Dict[str, int]:
    return {model_id: _context_window_for_payload({"model": model_id}) for model_id in _supported_model_ids()}


def _context_window_error(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    input_tokens = _estimate_count_tokens(payload)
    context_window = _context_window_for_payload(payload)
    if input_tokens <= context_window:
        return None
    requested_model, resolved_model, _effort = _resolve_model_and_effort(payload)
    return {
        "type": "error",
        "error": {
            "type": "context_length_exceeded",
            "message": (
                f"input estimate {input_tokens} tokens exceeds backend context window "
                f"{context_window} for requested model {requested_model} "
                f"(resolved backend {resolved_model}). The Claude-looking alias is UX-only; "
                "it cannot safely claim a larger context window than GPT-5.5."
            ),
            "input_tokens": input_tokens,
            "context_window": context_window,
            "requested_model": requested_model,
            "resolved_model": resolved_model,
        },
    }

def _anthropic_model_info(model_id: str) -> Dict[str, Any]:
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "openai-codex-pool"}


def _json_response(handler: BaseHTTPRequestHandler, status: int, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _extract_text_from_anthropic_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text") or ""))
            elif btype == "image":
                # Keep a text marker rather than dropping silently. This MVP is text/tool first.
                parts.append("[image omitted by local Codex bridge]")
            else:
                parts.append(_unsupported_block_transcript("nested", block))
    return "".join(parts)


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value)


def _unsupported_block_transcript(role: str, block: Any) -> str:
    btype = block.get("type") if isinstance(block, dict) else type(block).__name__
    return (
        f"\n\n<unsupported_{role}_content_block type={json.dumps(str(btype), ensure_ascii=False)}>\n"
        f"json: {_compact_json(block)}\n"
        f"</unsupported_{role}_content_block>\n"
    )


def _safe_error_message(exc: Exception, limit: int = 1000) -> str:
    # Avoid echoing headers, environment, or raw provider payloads. The class name
    # and truncated message are enough for bridge diagnostics without secrets.
    return f"{exc.__class__.__name__}: {str(exc)[:limit]}"


def _protocol_log_enabled() -> bool:
    return PROTOCOL_LOG or os.environ.get("CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG", "0").lower() in {"1", "true", "yes", "on"}


def _protocol_log(event: Dict[str, Any]) -> None:
    if not _protocol_log_enabled():
        return
    safe_event = {k: v for k, v in event.items() if k.lower() not in {"authorization", "api_key", "token"}}
    safe_event.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    line = json.dumps(safe_event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    target = PROTOCOL_LOG_FILE or os.environ.get("CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG_FILE", "").strip()
    if target:
        try:
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(line)
            return
        except Exception as exc:
            sys.stderr.write(json.dumps({"bridge_protocol_log_error": exc.__class__.__name__}) + "\n")
    sys.stderr.write(line)
    sys.stderr.flush()


def _content_block_types(blocks: Any) -> List[str]:
    if not isinstance(blocks, list):
        return [type(blocks).__name__]
    out: List[str] = []
    for block in blocks:
        if isinstance(block, dict):
            out.append(str(block.get("type") or "unknown"))
        else:
            out.append(type(block).__name__)
    return out


def _incoming_last_message_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    msg = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    content = msg.get("content")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    text_chars = 0
    tool_results = 0
    tool_uses = 0
    block_types: List[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            text_chars += len(str(block))
            block_types.append(type(block).__name__)
            continue
        btype = str(block.get("type") or "unknown")
        block_types.append(btype)
        if btype == "text":
            text_chars += len(str(block.get("text") or ""))
        elif btype == "tool_result":
            tool_results += 1
        elif btype == "tool_use":
            tool_uses += 1
    return {
        "role": str(msg.get("role") or ""),
        "content_block_types": block_types,
        "text_chars": text_chars,
        "tool_result_count": tool_results,
        "tool_use_count": tool_uses,
    }


def _count_tool_results(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        blocks = content if isinstance(content, list) else []
        count += sum(1 for block in blocks if isinstance(block, dict) and block.get("type") == "tool_result")
    return count


def _event_order_checksum(events: List[str]) -> str:
    return hashlib.sha256(",".join(events).encode("utf-8")).hexdigest()[:16]


def _tool_use_transcript(tool_id: str, name: str, args: Any) -> str:
    # Never put Claude Code's raw internal wrapper spelling into model history.
    # GPT/Codex backends can copy those markers back as ordinary assistant text,
    # which makes Claude Code display `[claude_tool_use ...]` instead of receiving
    # a structured Anthropic tool_use content block. Keep the flattened history
    # human-readable but deliberately non-protocol-shaped.
    return (
        "\n\n<completed_assistant_tool_call "
        f"id={json.dumps(tool_id, ensure_ascii=False)} "
        f"name={json.dumps(name, ensure_ascii=False)}>\n"
        f"input_json: {_compact_json(args)}\n"
        "</completed_assistant_tool_call>\n"
    )


def _tool_result_transcript(tool_use_id: str, content: Any) -> str:
    return (
        "\n\n<completed_tool_result "
        f"tool_use_id={json.dumps(tool_use_id, ensure_ascii=False)}>\n"
        f"{_extract_text_from_anthropic_content(content)}\n"
        "</completed_tool_result>\n"
    )


def _sanitize_tool_input(name: str, args: Any) -> Any:
    """Remove invalid empty optional fields emitted by the Codex backend.

    Claude Code's Read tool rejects pages="". The OpenAI/Codex side may emit
    that empty optional field even when asked to read a plain text file. Drop the
    invalid empty value before sending the tool_use block back to Claude Code.
    """
    if not isinstance(args, dict):
        return args
    cleaned = dict(args)
    if name == "Read" and cleaned.get("pages") == "":
        cleaned.pop("pages", None)
    return cleaned


def _anthropic_messages_to_openai(messages: List[Dict[str, Any]], system: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    has_tool_results = _count_tool_results(messages) > 0
    if system:
        if isinstance(system, list):
            sys_text = _extract_text_from_anthropic_content(system)
        else:
            sys_text = str(system)
        if sys_text.strip():
            if has_tool_results:
                sys_text = sys_text.rstrip() + "\n\n" + BRIDGE_TOOL_CONTINUATION_INSTRUCTION
            out.append({"role": "system", "content": sys_text})
    elif has_tool_results:
        out.append({"role": "system", "content": BRIDGE_TOOL_CONTINUATION_INSTRUCTION})

    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            for block in blocks:
                if not isinstance(block, dict):
                    text_parts.append(_unsupported_block_transcript("assistant", block))
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif btype == "tool_use":
                    tool_id = str(block.get("id") or f"toolu_{uuid.uuid4().hex[:16]}")
                    name = str(block.get("name") or "")
                    args = block.get("input") or {}
                    if STRUCTURED_TOOL_HISTORY:
                        if not isinstance(args, str):
                            args = json.dumps(args, ensure_ascii=False)
                        tool_calls.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        })
                    else:
                        text_parts.append(_tool_use_transcript(tool_id, name, args))
                else:
                    text_parts.append(_unsupported_block_transcript("assistant", block))
            entry: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            text_parts: List[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                btype = block.get("type")
                if btype == "tool_result":
                    tool_use_id = str(block.get("tool_use_id") or "")
                    tool_content = block.get("content")
                    if STRUCTURED_TOOL_HISTORY:
                        # Flush any preceding ordinary user text before tool output.
                        if "".join(text_parts).strip():
                            out.append({"role": "user", "content": "".join(text_parts)})
                            text_parts = []
                        out.append({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": _extract_text_from_anthropic_content(tool_content),
                        })
                    else:
                        text_parts.append(_tool_result_transcript(tool_use_id, tool_content))
                elif btype in {"text", "image"}:
                    text_parts.append(_extract_text_from_anthropic_content([block]))
                else:
                    text_parts.append(_unsupported_block_transcript("user", block))
            if "".join(text_parts).strip() or not blocks:
                out.append({"role": "user", "content": "".join(text_parts)})
    return out or [{"role": "user", "content": ""}]


def _anthropic_tools_to_openai(tools: Any) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return converted


def _selected_pool_label(client: Any) -> str | None:
    try:
        real = getattr(client, "_real_client", None)
        label = getattr(real, "_hermes_pool_label", None)
        return str(label) if label else None
    except Exception:
        return None


def _is_retryable_model_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
        return True
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    retry_markers = (
        "timeout", "timed out", "temporarily", "rate limit", "429",
        "server error", "bad gateway", "service unavailable", "gateway timeout",
        "connection", "network", "readerror", "remoteprotocolerror",
    )
    return any(marker in name or marker in text for marker in retry_markers)


def _parse_tool_arguments(tool_id: str, name: str, args_raw: Any) -> Tuple[Dict[str, Any] | None, str | None]:
    if args_raw is None or args_raw == "":
        return {}, None
    if isinstance(args_raw, dict):
        return args_raw, None
    try:
        parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except Exception as exc:
        err = f"invalid JSON arguments for tool_call id={tool_id!r} name={name!r}: {exc.__class__.__name__}"
        sys.stderr.write(json.dumps({"bridge_tool_arguments_error": err}, ensure_ascii=False) + "\n")
        sys.stderr.flush()
        return None, err
    if isinstance(parsed, dict):
        return parsed, None
    err = f"non-object JSON arguments for tool_call id={tool_id!r} name={name!r}: {type(parsed).__name__}"
    sys.stderr.write(json.dumps({"bridge_tool_arguments_error": err}, ensure_ascii=False) + "\n")
    sys.stderr.flush()
    return {"value": parsed}, err


def _map_stop_reason(finish_reason: Any, has_tool_use: bool) -> str:
    if has_tool_use:
        return "tool_use"
    fr = str(finish_reason or "").lower()
    if fr in {"length", "max_tokens"}:
        return "max_tokens"
    if fr in {"content_filter", "safety", "refusal"}:
        return "refusal"
    if fr in {"stop_sequence"}:
        return "stop_sequence"
    # If the backend claimed a tool call but every call was malformed and skipped,
    # return a diagnostic text turn instead of making Claude Code wait for a tool.
    if fr in {"tool_calls", "function_call"}:
        return "end_turn"
    return "end_turn"


def _usage_from_openai(usage: Any) -> Dict[str, int]:
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _estimate_count_tokens(payload: Dict[str, Any]) -> int:
    counted = {
        "system": payload.get("system"),
        "messages": payload.get("messages") or [],
        "tools": payload.get("tools") or [],
        "tool_choice": payload.get("tool_choice"),
    }
    text = json.dumps(counted, ensure_ascii=False, sort_keys=True)
    return max(1, len(text) // 4)


def _structured_reasoning_text(msg: Any) -> str:
    parts: List[str] = []
    for field in ("reasoning", "reasoning_content"):
        val = getattr(msg, field, None)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    details = getattr(msg, "reasoning_details", None)
    if isinstance(details, list):
        for detail in details:
            summary = detail.get("summary") if isinstance(detail, dict) else getattr(detail, "summary", None)
            if isinstance(summary, str) and summary.strip():
                parts.append(summary.strip())
            elif isinstance(summary, list):
                for item in summary:
                    text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
    seen: set[str] = set()
    return "\n\n".join(part for part in parts if not (part in seen or seen.add(part))).strip()


def _codex_call(
    payload: Dict[str, Any],
    message_id: str | None = None,
    on_text_delta=None,
    on_reasoning_delta=None,
) -> Tuple[Dict[str, Any], str | None, Dict[str, Any]]:
    from agent.auxiliary_client import _build_codex_client

    requested_model, model, effort = _resolve_model_and_effort(payload)

    messages = _anthropic_messages_to_openai(payload.get("messages") or [], payload.get("system"))
    tools = _anthropic_tools_to_openai(payload.get("tools"))
    requested_timeout = float(payload.get("timeout") or DEFAULT_TIMEOUT)
    timeout_s = min(requested_timeout, MAX_REQUEST_TIMEOUT) if MAX_REQUEST_TIMEOUT > 0 else requested_timeout
    if requested_timeout != timeout_s:
        sys.stderr.write(
            json.dumps(
                {
                    "bridge_timeout_clamped": True,
                    "requested_timeout": requested_timeout,
                    "effective_timeout": timeout_s,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stderr.flush()
    attempts = max(1, MAX_RETRIES + 1)
    selected_label: str | None = None
    last_exc: Exception | None = None

    reasoning_parts: List[str] = []

    def _capture_reasoning_delta(text: str) -> None:
        if text:
            reasoning_parts.append(text)
            if on_reasoning_delta is not None:
                on_reasoning_delta(text)

    for attempt in range(attempts):
        # Build a fresh client on every attempt so Hermes pool.select() advances
        # under the configured round_robin strategy instead of retrying a stuck
        # credential/route forever.
        client, resolved_model = _build_codex_client(model)
        if client is None:
            raise RuntimeError("No OpenAI-Codex credential available in Hermes pool")
        selected_label = _selected_pool_label(client)
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": timeout_s,
            "extra_body": {"reasoning": {"effort": effort}},
        }
        if on_text_delta is not None:
            kwargs["_on_text_delta"] = on_text_delta
        kwargs["_on_reasoning_delta"] = _capture_reasoning_delta
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_model_error(exc)
            if attempt >= attempts - 1 or not retryable:
                raise
            sys.stderr.write(
                json.dumps(
                    {
                        "bridge_retry": True,
                        "attempt": attempt + 1,
                        "max_attempts": attempts,
                        "error_type": exc.__class__.__name__,
                        "next_attempt_uses_fresh_pool_selection": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sys.stderr.flush()
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    else:
        assert last_exc is not None
        raise last_exc
    choice = resp.choices[0]
    msg = choice.message
    text = getattr(msg, "content", None) or ""
    content_blocks: List[Dict[str, Any]] = []
    reasoning_text = "".join(reasoning_parts).strip() or _structured_reasoning_text(msg)
    if reasoning_text:
        # Claude Code stream-json exposes thinking to cc-connect only when the
        # final assistant message contains a thinking content block. Raw SSE
        # thinking_delta alone is not surfaced by Claude Code's stream-json.
        thinking_block = {"type": "thinking", "thinking": reasoning_text}
        if THINKING_SIGNATURE_PLACEHOLDER:
            thinking_block["signature"] = THINKING_SIGNATURE
        content_blocks.append(thinking_block)
    if text:
        content_blocks.append({"type": "text", "text": text})
    tool_calls = getattr(msg, "tool_calls", None) or []
    valid_tool_uses = 0
    malformed_tool_count = 0
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = str(getattr(fn, "name", "") if fn else getattr(tc, "name", ""))
        args_raw = getattr(fn, "arguments", "{}") if fn else getattr(tc, "arguments", "{}")
        tool_id = str(getattr(tc, "id", "") or f"toolu_{uuid.uuid4().hex[:16]}")
        args, parse_error = _parse_tool_arguments(tool_id, name, args_raw)
        if parse_error:
            malformed_tool_count += 1
            content_blocks.append({
                "type": "text",
                "text": (
                    f"<bridge_tool_call_argument_error id={json.dumps(tool_id, ensure_ascii=False)} "
                    f"name={json.dumps(name, ensure_ascii=False)}>"
                    f"{parse_error}"
                    "</bridge_tool_call_argument_error>"
                ),
            })
        if args is None:
            continue
        content_blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": _sanitize_tool_input(name, args),
        })
        valid_tool_uses += 1
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    finish_reason = getattr(choice, "finish_reason", None)
    stop_reason = _map_stop_reason(finish_reason, valid_tool_uses > 0)
    usage = getattr(resp, "usage", None)
    result = {
        "id": message_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _usage_from_openai(usage),
    }
    meta = {
        "requested_model": requested_model,
        "resolved_model": model,
        "effort": effort,
        "finish_reason": str(finish_reason or ""),
        "stop_reason": stop_reason,
        "content_block_types": _content_block_types(content_blocks),
        "tool_use_ids": [str(b.get("id") or "") for b in content_blocks if isinstance(b, dict) and b.get("type") == "tool_use"],
        "tool_use_names": [str(b.get("name") or "") for b in content_blocks if isinstance(b, dict) and b.get("type") == "tool_use"],
        "malformed_tool_count": malformed_tool_count,
    }
    # Only expose non-secret selected label if available.
    label = None
    try:
        real = getattr(client, "_real_client", None)
        label = getattr(real, "_hermes_pool_label", None)
    except Exception:
        pass
    return result, label, meta


def _sse_write(handler: BaseHTTPRequestHandler, event: str, data: Dict[str, Any]) -> None:
    events = getattr(handler, "_bridge_sse_events", None)
    if isinstance(events, list):
        events.append(event)
    raw = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
    handler.wfile.write(raw)
    handler.wfile.flush()


def _sse_emit_result_content(handler: BaseHTTPRequestHandler, result: Dict[str, Any], start_index: int = 0) -> None:
    for rel_idx, block in enumerate(result.get("content") or []):
        idx = start_index + rel_idx
        if not isinstance(block, dict):
            block = {"type": "text", "text": _unsupported_block_transcript("assistant", block)}
        if block.get("type") == "thinking":
            start_block = {"type": "thinking", "thinking": ""}
            if THINKING_SIGNATURE_PLACEHOLDER:
                start_block["signature"] = THINKING_SIGNATURE
            _sse_write(handler, "content_block_start", {"type": "content_block_start", "index": idx, "content_block": start_block})
            thinking = block.get("thinking") or block.get("text") or ""
            if thinking:
                _sse_write(handler, "content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "thinking_delta", "thinking": thinking}})
            if THINKING_SIGNATURE_PLACEHOLDER:
                _sse_write(handler, "content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "signature_delta", "signature": THINKING_SIGNATURE}})
            _sse_write(handler, "content_block_stop", {"type": "content_block_stop", "index": idx})
        elif block.get("type") == "text":
            _sse_write(handler, "content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
            txt = block.get("text") or ""
            if txt:
                _sse_write(handler, "content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": txt}})
            _sse_write(handler, "content_block_stop", {"type": "content_block_stop", "index": idx})
        elif block.get("type") == "tool_use":
            start_block = {"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": {}}
            _sse_write(handler, "content_block_start", {"type": "content_block_start", "index": idx, "content_block": start_block})
            partial = json.dumps(block.get("input") or {}, ensure_ascii=False, separators=(",", ":"))
            _sse_write(handler, "content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": partial}})
            _sse_write(handler, "content_block_stop", {"type": "content_block_stop", "index": idx})
        else:
            _sse_write(handler, "content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
            _sse_write(handler, "content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": _unsupported_block_transcript("assistant", block)}})
            _sse_write(handler, "content_block_stop", {"type": "content_block_stop", "index": idx})


class Handler(BaseHTTPRequestHandler):
    server_version = "codex-anthropic-bridge/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid logging request headers/tokens. Keep a compact access log.
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _authorized(self) -> bool:
        cfg = _load_config()
        token = str(cfg.get("auth_token") or "")
        if not token:
            return True
        auth = self.headers.get("authorization") or self.headers.get("x-api-key") or ""
        return auth == f"Bearer {token}" or auth == token

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if request_path in {"/", "/health", "/healthz"}:
            _json_response(self, 200, {
                "ok": True,
                "model": DEFAULT_MODEL,
                "models": _supported_model_ids(),
                "backend": "hermes-openai-codex-pool",
                "context_window_tokens": DEFAULT_CONTEXT_WINDOW_TOKENS,
                "context_windows": _context_windows(),
                "features": ["model_aliases", "reasoning_effort_aliases", "streaming_thinking_delta", "context_window_guard"],
            })
            return
        if request_path.rstrip("/") == "/v1/models":
            _json_response(self, 200, {"object": "list", "data": [_anthropic_model_info(mid) for mid in _supported_model_ids()]})
            return
        _json_response(self, 404, {"type": "error", "error": {"type": "not_found_error", "message": "not found"}})

    def _handle_streaming_messages(self, payload: Dict[str, Any], request_id: str) -> None:
        message_id = f"msg_{uuid.uuid4().hex}"
        self._bridge_sse_events = []
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        requested_model, backend_model, effort = _resolve_model_and_effort(payload)
        start_message = {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        _sse_write(self, "message_start", {"type": "message_start", "message": start_message})

        done = queue.Queue(maxsize=1)
        live_events = queue.Queue()
        thinking_opened = False
        next_content_index = 0

        def on_reasoning_delta(text: str) -> None:
            if text:
                live_events.put(("reasoning", text))

        def run_backend() -> None:
            try:
                result, label, meta = _codex_call(
                    payload,
                    message_id=message_id,
                    on_reasoning_delta=on_reasoning_delta,
                )
                done.put(("ok", result, (label, meta)))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                done.put(("error", exc, None))

        threading.Thread(target=run_backend, daemon=True).start()
        last_ping = time.monotonic()
        while True:
            try:
                while True:
                    ev_kind, ev_text = live_events.get_nowait()
                    if ev_kind == "reasoning":
                        if not thinking_opened:
                            thinking_opened = True
                            _sse_write(self, "content_block_start", {
                                "type": "content_block_start",
                                "index": next_content_index,
                                "content_block": {"type": "thinking", "thinking": "", **({"signature": THINKING_SIGNATURE} if THINKING_SIGNATURE_PLACEHOLDER else {})},
                            })
                        _sse_write(self, "content_block_delta", {
                            "type": "content_block_delta",
                            "index": next_content_index,
                            "delta": {"type": "thinking_delta", "thinking": ev_text},
                        })
            except queue.Empty:
                pass

            try:
                kind, obj, label_meta = done.get_nowait()
                break
            except queue.Empty:
                now = time.monotonic()
                if now - last_ping >= STREAM_PING_INTERVAL:
                    _sse_write(self, "ping", {"type": "ping"})
                    last_ping = now
                time.sleep(0.05)

        # Drain any final reasoning events before closing the thinking block.
        try:
            while True:
                ev_kind, ev_text = live_events.get_nowait()
                if ev_kind == "reasoning":
                    if not thinking_opened:
                        thinking_opened = True
                        _sse_write(self, "content_block_start", {
                            "type": "content_block_start",
                            "index": next_content_index,
                            "content_block": {"type": "thinking", "thinking": "", **({"signature": THINKING_SIGNATURE} if THINKING_SIGNATURE_PLACEHOLDER else {})},
                        })
                    _sse_write(self, "content_block_delta", {
                        "type": "content_block_delta",
                        "index": next_content_index,
                        "delta": {"type": "thinking_delta", "thinking": ev_text},
                    })
        except queue.Empty:
            pass
        if thinking_opened:
            if THINKING_SIGNATURE_PLACEHOLDER:
                _sse_write(self, "content_block_delta", {
                    "type": "content_block_delta",
                    "index": next_content_index,
                    "delta": {"type": "signature_delta", "signature": THINKING_SIGNATURE},
                })
            _sse_write(self, "content_block_stop", {"type": "content_block_stop", "index": next_content_index})
            next_content_index += 1

        if kind == "error":
            err = _safe_error_message(obj)
            _sse_write(self, "error", {"type": "error", "error": {"type": "api_error", "message": err[:1000]}})
            _sse_write(self, "message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            })
            _sse_write(self, "message_stop", {"type": "message_stop"})
            _protocol_log({
                "event": "messages_response",
                "request_id": request_id,
                "stream": True,
                "requested_model": requested_model,
                "resolved_model": backend_model,
                "effort": effort,
                "incoming_last_message": _incoming_last_message_summary(payload),
                "tool_result_count": _count_tool_results(payload.get("messages")),
                "outgoing_content_block_types": ["error"],
                "tool_use_ids": [],
                "tool_use_names": [],
                "finish_reason": "error",
                "stop_reason": "end_turn",
                "malformed_tool_count": 0,
                "event_order_checksum": _event_order_checksum(getattr(self, "_bridge_sse_events", [])),
            })
            return

        result = obj
        _label, meta = label_meta if isinstance(label_meta, tuple) else (None, {})
        if thinking_opened and isinstance(result, dict):
            # Avoid replaying the accumulated final thinking block after the
            # live thinking_delta block already streamed to Claude Code.
            result = dict(result)
            result["content"] = [b for b in (result.get("content") or []) if not (isinstance(b, dict) and b.get("type") == "thinking")]
        _sse_emit_result_content(self, result, start_index=next_content_index)
        _sse_write(self, "message_delta", {"type": "message_delta", "delta": {"stop_reason": result.get("stop_reason"), "stop_sequence": result.get("stop_sequence")}, "usage": result.get("usage", {})})
        _sse_write(self, "message_stop", {"type": "message_stop"})
        _protocol_log({
            "event": "messages_response",
            "request_id": request_id,
            "stream": True,
            "requested_model": meta.get("requested_model", requested_model),
            "resolved_model": meta.get("resolved_model", backend_model),
            "effort": meta.get("effort", effort),
            "incoming_last_message": _incoming_last_message_summary(payload),
            "tool_result_count": _count_tool_results(payload.get("messages")),
            "outgoing_content_block_types": _content_block_types(result.get("content") or []),
            "tool_use_ids": meta.get("tool_use_ids", []),
            "tool_use_names": meta.get("tool_use_names", []),
            "finish_reason": meta.get("finish_reason", ""),
            "stop_reason": result.get("stop_reason"),
            "malformed_tool_count": meta.get("malformed_tool_count", 0),
            "event_order_checksum": _event_order_checksum(getattr(self, "_bridge_sse_events", [])),
        })

    def do_HEAD(self) -> None:
        request_path = urlparse(self.path).path
        if request_path in {"/", "/health", "/healthz", "/v1/models"}:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        request_path = urlparse(self.path).path
        if not self._authorized():
            _json_response(self, 401, {"type": "error", "error": {"type": "authentication_error", "message": "unauthorized"}})
            return
        length = int(self.headers.get("content-length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            _json_response(self, 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "invalid json"}})
            return

        if request_path.rstrip("/") == "/v1/messages/count_tokens":
            _json_response(self, 200, {"input_tokens": _estimate_count_tokens(payload)})
            return
        if request_path.rstrip("/") != "/v1/messages":
            _json_response(self, 404, {"type": "error", "error": {"type": "not_found_error", "message": "not found"}})
            return

        context_error = _context_window_error(payload)
        if context_error is not None:
            _json_response(self, 400, context_error)
            return

        try:
            request_id = self.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
            if payload.get("stream"):
                self._handle_streaming_messages(payload, request_id)
                return
            result, label, meta = _codex_call(payload)
            if label:
                result["container"] = {"selected_label": label}
            _json_response(self, 200, result)
            _protocol_log({
                "event": "messages_response",
                "request_id": request_id,
                "stream": False,
                "requested_model": meta.get("requested_model"),
                "resolved_model": meta.get("resolved_model"),
                "effort": meta.get("effort"),
                "incoming_last_message": _incoming_last_message_summary(payload),
                "tool_result_count": _count_tool_results(payload.get("messages")),
                "outgoing_content_block_types": _content_block_types(result.get("content") or []),
                "tool_use_ids": meta.get("tool_use_ids", []),
                "tool_use_names": meta.get("tool_use_names", []),
                "finish_reason": meta.get("finish_reason"),
                "stop_reason": result.get("stop_reason"),
                "malformed_tool_count": meta.get("malformed_tool_count", 0),
                "event_order_checksum": _event_order_checksum(_content_block_types(result.get("content") or []) + [str(result.get("stop_reason") or "")]),
            })
        except Exception as exc:
            # No secrets in error body; truncate.
            _json_response(self, 500, {"type": "error", "error": {"type": exc.__class__.__name__, "message": _safe_error_message(exc)}})
            _protocol_log({
                "event": "messages_error",
                "request_id": self.headers.get("x-request-id") or "unknown",
                "stream": bool(payload.get("stream")),
                "incoming_last_message": _incoming_last_message_summary(payload),
                "tool_result_count": _count_tool_results(payload.get("messages")),
                "error_type": exc.__class__.__name__,
            })
            traceback.print_exc(file=sys.stderr)


def main() -> None:
    cfg = _load_config()
    host = str(cfg.get("host") or "127.0.0.1")
    port = int(cfg.get("port") or 15722)
    if host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Refusing to bind non-local host")
    srv = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"started": True, "host": host, "port": port, "model": DEFAULT_MODEL}, ensure_ascii=False), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
