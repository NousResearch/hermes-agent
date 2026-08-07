"""Loopback OpenAI-compatible bridge backed by Hermes' host-owned LLM facade.

HindsightEmbedded runs its LLM calls in a managed daemon and accepts an
OpenAI-compatible endpoint, while Hermes plugins receive the host-owned
``ctx.llm`` facade instead of provider credentials.  This bridge connects the
two without copying Hermes credentials into Hindsight configuration.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_MAX_BODY_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_CONCURRENT = 4


class HermesLlmBridgeError(Exception):
    """A safe, client-facing bridge error with an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI text content into a bounded bridge input string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise HermesLlmBridgeError("messages must be a non-empty array")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HermesLlmBridgeError("each message must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise HermesLlmBridgeError("each message must have a role")
        role = role.strip()
        normalized_message: dict[str, Any] = {
            "role": role,
            "content": _content_to_text(message.get("content")),
        }

        raw_tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(raw_tool_calls, list) and raw_tool_calls:
            tool_calls: list[dict[str, Any]] = []
            for raw_call in raw_tool_calls:
                if not isinstance(raw_call, dict):
                    raise HermesLlmBridgeError("assistant tool calls must be objects")
                function = raw_call.get("function")
                if not isinstance(function, dict):
                    raise HermesLlmBridgeError("assistant tool calls must include a function")
                call_id = raw_call.get("id")
                name = function.get("name")
                if not isinstance(call_id, str) or not call_id:
                    raise HermesLlmBridgeError("assistant tool calls must include an id")
                if not isinstance(name, str) or not name:
                    raise HermesLlmBridgeError("assistant tool calls must include a function name")
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append({
                    "id": call_id,
                    "type": raw_call.get("type") or "function",
                    "function": {"name": name, "arguments": arguments},
                })
            normalized_message["tool_calls"] = tool_calls
            if message.get("content") is None:
                normalized_message["content"] = None

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise HermesLlmBridgeError("tool messages must include tool_call_id")
            normalized_message["tool_call_id"] = tool_call_id

        normalized.append(normalized_message)
    return normalized


def _coerce_optional_number(value: Any, *, integer: bool = False) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None


class _BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], bridge: "HermesLlmBridge") -> None:
        super().__init__(address, _BridgeRequestHandler)
        self.bridge = bridge


class _BridgeRequestHandler(BaseHTTPRequestHandler):
    # Hindsight uses ordinary non-streaming completion requests.  HTTP/1.1 is
    # used so the OpenAI client can reuse the loopback connection safely.
    protocol_version = "HTTP/1.1"

    @property
    def _bridge(self) -> "HermesLlmBridge":
        return cast(_BridgeServer, self.server).bridge

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Never log request bodies, authorization headers, or prompts.
        logger.debug("Hindsight Hermes LLM bridge: " + format, *args)

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        authorization = self.headers.get("Authorization", "")
        expected = f"Bearer {self._bridge.api_key}"
        return hmac.compare_digest(authorization, expected)

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._write_json(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
            return
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "hermes-inherited",
                            "object": "model",
                            "owned_by": "hermes",
                        }
                    ],
                },
            )
            return
        self._write_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._write_json(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
            return

        path = urlsplit(self.path).path.rstrip("/")
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self._write_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})
            return

        raw_length = self.headers.get("Content-Length", "")
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError):
            self._write_json(411, {"error": {"message": "Content-Length required", "type": "invalid_request_error"}})
            return
        if content_length < 0 or content_length > _MAX_BODY_BYTES:
            self._write_json(413, {"error": {"message": "Request body too large", "type": "invalid_request_error"}})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self._write_json(
                400,
                {
                    "error": {
                        "message": "Request body must contain valid JSON",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        try:
            response = self._bridge.complete(payload)
        except HermesLlmBridgeError as exc:
            self._write_json(exc.status_code, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        except Exception as exc:
            # Do not log provider details or exception text: SDK errors can carry
            # request content, response bodies, or credential-derived context.
            logger.warning(
                "Hindsight Hermes LLM bridge completion failed (%s)",
                type(exc).__name__,
            )
            self._write_json(502, {"error": {"message": "Hermes LLM bridge completion failed", "type": "upstream_error"}})
            return
        self._write_json(200, response)


class HermesLlmBridge:
    """Expose a narrow OpenAI-compatible endpoint backed by ``ctx.llm``."""

    def __init__(
        self,
        llm: Any,
        *,
        host: str = "127.0.0.1",
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    ) -> None:
        if llm is None:
            raise ValueError("HermesLlmBridge requires the host-owned LLM facade")
        self._llm = llm
        self._host = host
        self._api_key = secrets.token_urlsafe(32)
        self._max_concurrent = max(1, int(max_concurrent))
        self._semaphore = threading.BoundedSemaphore(self._max_concurrent)
        self._server: _BridgeServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def api_key(self) -> str:
        """Ephemeral bridge credential, never a provider credential."""
        return self._api_key

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Hermes LLM bridge has not been started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        server = _BridgeServer((self._host, 0), self)
        thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="hindsight-hermes-llm-bridge",
        )
        self._server = server
        self._thread = thread
        thread.start()
        logger.info("Hindsight Hermes LLM bridge listening on loopback")

    def close(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def complete(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HermesLlmBridgeError("request body must be a JSON object")
        if payload.get("stream"):
            raise HermesLlmBridgeError("streaming is not supported by the Hindsight memory bridge")
        messages = _normalize_messages(payload.get("messages"))
        if not self._semaphore.acquire(timeout=120.0):
            raise HermesLlmBridgeError("Hermes LLM bridge is busy", status_code=429)
        try:
            response_format = payload.get("response_format")
            if isinstance(response_format, dict) and response_format.get("type") in {"json_schema", "json_object"}:
                result = self._complete_structured(messages, payload, response_format)
            else:
                result = self._complete_text(messages, payload)
            return self._openai_response(result, payload)
        finally:
            self._semaphore.release()

    def _complete_text(self, messages: list[dict[str, str]], payload: dict[str, Any]) -> Any:
        return self._llm.complete(
            messages,
            temperature=_coerce_optional_number(payload.get("temperature")),
            max_tokens=_coerce_optional_number(
                payload.get("max_completion_tokens", payload.get("max_tokens")), integer=True
            ),
            tools=payload.get("tools") if isinstance(payload.get("tools"), list) else None,
            tool_choice=payload.get("tool_choice"),
            purpose="hindsight",
        )

    def _complete_structured(
        self,
        messages: list[dict[str, str]],
        payload: dict[str, Any],
        response_format: dict[str, Any],
    ) -> Any:
        schema_wrapper = response_format.get("json_schema")
        schema = schema_wrapper.get("schema") if isinstance(schema_wrapper, dict) else None
        schema_name = schema_wrapper.get("name") if isinstance(schema_wrapper, dict) else None
        system_prompt = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system" and message["content"]
        )
        input_text = "\n\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
            if message["role"] != "system"
        )
        if not input_text:
            input_text = "Return the requested structured response."
        from agent.plugin_llm import PluginLlmTextInput

        return self._llm.complete_structured(
            instructions="Follow the system instructions and return the requested response.",
            input=[PluginLlmTextInput(text=input_text)],
            json_schema=schema,
            json_mode=response_format.get("type") in {"json_schema", "json_object"},
            schema_name=schema_name,
            system_prompt=system_prompt or None,
            temperature=_coerce_optional_number(payload.get("temperature")),
            max_tokens=_coerce_optional_number(
                payload.get("max_completion_tokens", payload.get("max_tokens")), integer=True
            ),
            purpose="hindsight",
        )

    @staticmethod
    def _openai_response(result: Any, payload: dict[str, Any]) -> dict[str, Any]:
        text = getattr(result, "text", "")
        if not isinstance(text, str):
            text = str(text)
        if not text and getattr(result, "parsed", None) is not None:
            text = json.dumps(result.parsed, ensure_ascii=False)
        usage = getattr(result, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        model = getattr(result, "model", None) or payload.get("model") or "hermes-inherited"
        tool_calls = getattr(result, "tool_calls", None) or []
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None if tool_calls else text,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-hermes-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }


__all__ = ["HermesLlmBridge", "HermesLlmBridgeError"]
