"""Custom httpx transport that rewrites requests to the Codex ChatGPT endpoint.

The OpenAI SDK sends chat completions format, but the Codex endpoint speaks
the responses API exclusively (with stream=true, store=false). This transport:

1. Converts chat completions request bodies to responses API format.
2. Rewrites the URL to https://chatgpt.com/backend-api/codex/responses.
3. Injects Bearer token + ChatGPT-Account-Id headers.
4. Consumes the SSE stream and reassembles a chat completions JSON response.
5. Handles 401 → refresh → retry.

This keeps the rest of the codebase unchanged — the agent loop still calls
client.chat.completions.create() and gets back a normal ChatCompletion.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from agent.codex_auth import refresh_codex_chatgpt_auth

logger = logging.getLogger(__name__)

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


def _chat_to_responses_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a chat completions request body to responses API format."""
    messages = body.get("messages", [])

    # Extract system message as instructions
    instructions = None
    input_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            instructions = msg.get("content", "")
        else:
            input_messages.append(msg)

    result: Dict[str, Any] = {
        "model": body.get("model", ""),
        "stream": True,
        "store": False,
        "input": input_messages,
    }
    if instructions:
        result["instructions"] = instructions

    # Forward tools if present
    tools = body.get("tools")
    if tools:
        # Convert from chat completions tool format to responses API format
        responses_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                responses_tools.append({
                    "type": "function",
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
        if responses_tools:
            result["tools"] = responses_tools

    return result


def _parse_sse_stream(raw_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse SSE events from raw response bytes."""
    events = []
    text = raw_bytes.decode("utf-8", errors="replace")
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                continue
    return events


def _responses_to_chat_completion(events: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """Reassemble responses API SSE events into a chat completions JSON response."""
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_call_args: Dict[str, List[str]] = {}  # id -> arg chunks
    usage = {}
    response_id = ""
    finish_reason = "stop"

    for event in events:
        event_type = event.get("type", "")

        if event_type == "response.created":
            resp = event.get("response", {})
            response_id = resp.get("id", "")

        elif event_type == "response.output_text.delta":
            text_parts.append(event.get("delta", ""))

        elif event_type == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                call_id = item.get("call_id", str(uuid.uuid4()))
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": "",
                    },
                })
                tool_call_args[item.get("id", "")] = []

        elif event_type == "response.function_call_arguments.delta":
            item_id = event.get("item_id", "")
            if item_id in tool_call_args:
                tool_call_args[item_id].append(event.get("delta", ""))

        elif event_type == "response.function_call_arguments.done":
            item_id = event.get("item_id", "")
            # Find the matching tool call and set full arguments
            for tc in tool_calls:
                if item_id in tool_call_args:
                    args_str = "".join(tool_call_args[item_id])
                    # Match by checking if this is the right tool call
                    # (tool_calls are ordered same as output items)
                    tc["function"]["arguments"] = args_str
                    del tool_call_args[item_id]
                    break

        elif event_type == "response.completed":
            resp = event.get("response", {})
            resp_usage = resp.get("usage", {})
            if resp_usage:
                usage = {
                    "prompt_tokens": resp_usage.get("input_tokens", 0),
                    "completion_tokens": resp_usage.get("output_tokens", 0),
                    "total_tokens": resp_usage.get("total_tokens",
                        resp_usage.get("input_tokens", 0) + resp_usage.get("output_tokens", 0)),
                }

    if tool_calls:
        finish_reason = "tool_calls"

    # Build chat completions response
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": response_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class CodexTransport(httpx.BaseTransport):
    """Transparent httpx transport that routes OpenAI SDK requests through Codex."""

    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        account_id: Optional[str] = None,
    ):
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._account_id = account_id
        self._inner = httpx.HTTPTransport()

    def _apply_headers(self, request: httpx.Request) -> None:
        request.headers["authorization"] = f"Bearer {self._access_token}"
        request.headers["originator"] = "opencode"
        request.headers["User-Agent"] = "hermes-agent"
        if self._account_id:
            request.headers["ChatGPT-Account-Id"] = self._account_id

    def _should_rewrite(self, request: httpx.Request) -> bool:
        path = request.url.raw_path.decode("ascii", errors="replace")
        return "/chat/completions" in path or "/v1/responses" in path

    def _refresh(self) -> bool:
        if not self._refresh_token:
            return False
        refreshed = refresh_codex_chatgpt_auth(refresh_token=self._refresh_token)
        if not refreshed:
            return False
        self._access_token = refreshed["access_token"]
        self._refresh_token = refreshed.get("refresh_token", self._refresh_token)
        self._account_id = refreshed.get("account_id", self._account_id)
        logger.info("Refreshed Codex access token")
        return True

    def _build_codex_request(self, request: httpx.Request) -> httpx.Request:
        """Rewrite URL and convert body from chat completions to responses format."""
        try:
            original_body = json.loads(request.content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            original_body = {}

        responses_body = _chat_to_responses_body(original_body)
        content = json.dumps(responses_body).encode("utf-8")

        # Build clean headers — drop SDK-specific ones that confuse Cloudflare
        headers = {
            "content-type": "application/json",
            "content-length": str(len(content)),
            "accept": "text/event-stream",
        }

        return httpx.Request(
            method=request.method,
            url=CODEX_RESPONSES_URL,
            headers=headers,
            content=content,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not self._should_rewrite(request):
            return self._inner.handle_request(request)

        # Save original model for response reconstruction
        try:
            model = json.loads(request.content).get("model", "")
        except Exception:
            model = ""

        request = self._build_codex_request(request)
        self._apply_headers(request)

        response = self._inner.handle_request(request)

        if response.status_code == 401 and self._refresh():
            self._apply_headers(request)
            response = self._inner.handle_request(request)

        if response.status_code != 200:
            return response

        # Consume the SSE stream and convert to chat completions JSON
        raw = response.read()
        events = _parse_sse_stream(raw)
        chat_completion = _responses_to_chat_completion(events, model)
        result_bytes = json.dumps(chat_completion).encode("utf-8")

        return httpx.Response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "content-length": str(len(result_bytes)),
            },
            content=result_bytes,
        )

    def close(self) -> None:
        self._inner.close()
