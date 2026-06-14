"""Shared llama.cpp OpenAI-compatible endpoint contract checks."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MIN_CONTEXT_SIZE = 64000
TARGET_CONTEXT_SIZE = 65536


def _get_json(url: str, timeout: float = 15.0) -> tuple[int, Any]:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, json.loads(body) if body.strip() else {}


def _post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, json.loads(body) if body.strip() else {}


def extract_context_size(props_payload: dict[str, Any]) -> int | None:
    candidates: list[Any] = [
        props_payload.get("default_generation_settings", {}).get("n_ctx")
        if isinstance(props_payload.get("default_generation_settings"), dict)
        else None,
        props_payload.get("n_ctx"),
        props_payload.get("total_slots", {}).get("n_ctx")
        if isinstance(props_payload.get("total_slots"), dict)
        else None,
    ]
    slot = props_payload.get("slot")
    if isinstance(slot, list) and slot:
        first = slot[0]
        if isinstance(first, dict):
            candidates.append(first.get("n_ctx"))
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def extract_model_ids(models_payload: Any) -> list[str]:
    if not isinstance(models_payload, dict):
        return []
    data = models_payload.get("data", [])
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        value = item.get("id") if isinstance(item, dict) else item
        if isinstance(value, str) and value.strip():
            ids.append(value.strip())
    return ids


def looks_like_plaintext_tool_call(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    if text.startswith("{") and '"tool_calls"' in text:
        return False
    markers = (
        "tool_call",
        "function_call",
        "<tool_call>",
        "Action:",
        "Calling tool",
    )
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def run_llama_contract_checks(
    base_url: str = "http://127.0.0.1:8080",
    *,
    min_context: int = MIN_CONTEXT_SIZE,
    timeout: float = 120.0,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    result: dict[str, Any] = {
        "base_url": base,
        "checks": {},
        "ok": True,
        "min_context": min_context,
        "target_context": TARGET_CONTEXT_SIZE,
    }
    model_id = "unknown"

    def _fail(name: str, detail: str) -> None:
        result["checks"][name] = {"ok": False, "error": detail}
        result["ok"] = False

    def _pass(name: str, detail: dict[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {"ok": True}
        if detail:
            entry.update(detail)
        result["checks"][name] = entry

    try:
        status, models = _get_json(f"{base}/v1/models")
        if status != 200:
            _fail("models", f"HTTP {status}")
        else:
            ids = extract_model_ids(models)
            _pass("models", {"model_ids": ids})
            model_id = ids[0] if ids else "unknown"
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        _fail("models", str(exc))
        result["summary"] = "llama.cpp server unreachable"
        return result

    try:
        status, props = _get_json(f"{base}/props")
        if status != 200:
            _fail("props", f"HTTP {status}")
        else:
            ctx = extract_context_size(props if isinstance(props, dict) else {})
            if ctx is None:
                _fail("context_size", "could not parse n_ctx from /props")
            elif ctx < min_context:
                _fail(
                    "context_size",
                    f"context {ctx} < required minimum {min_context}",
                )
            else:
                _pass("context_size", {"n_ctx": ctx})
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        _fail("props", str(exc))

    chat_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        status, chat = _post_json(f"{base}/v1/chat/completions", chat_payload, timeout=timeout)
        if status != 200:
            _fail("chat_completion", f"HTTP {status}")
        else:
            content = (
                chat.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            _pass("chat_completion", {"content_preview": str(content)[:120]})
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        _fail("chat_completion", str(exc))

    tool_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 128,
        "temperature": 0,
    }
    try:
        status, tool_resp = _post_json(
            f"{base}/v1/chat/completions", tool_payload, timeout=timeout
        )
        if status != 200:
            _fail("tool_calling", f"HTTP {status}")
        else:
            message = tool_resp.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            if tool_calls:
                _pass("tool_calling", {"tool_calls": len(tool_calls)})
            elif looks_like_plaintext_tool_call(str(content)):
                _fail(
                    "tool_calling",
                    "tool call returned as plain text; start llama-server with --jinja",
                )
            else:
                _fail("tool_calling", "no structured tool_calls in response")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        _fail("tool_calling", str(exc))

    result["summary"] = "ok" if result["ok"] else "failed"
    return result


def validate_config_context_length(context_length: int | None) -> tuple[bool, str | None]:
    if context_length is None:
        return False, "context_length is not configured"
    if context_length < MIN_CONTEXT_SIZE:
        return (
            False,
            f"context_length {context_length} is below minimum {MIN_CONTEXT_SIZE}",
        )
    return True, None
