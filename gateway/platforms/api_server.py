"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless)
- POST /v1/message                 — Session-backed message endpoint
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent as an available model
- GET  /health                     — health check

Any OpenAI-compatible frontend (Open WebUI, LobeChat, LibreChat,
AnythingLLM, NextChat, ChatBox, etc.) can connect to hermes-agent
through this adapter by pointing at http://localhost:8642/v1.

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import collections
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)

from gateway.session import SessionSource, SessionStore

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100


def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


class ResponseStore:
    """
    In-memory LRU store for Responses API state.

    Each stored response includes the full internal conversation history
    (with tool calls and results) so it can be reconstructed on subsequent
    requests via previous_response_id.
    """

    def __init__(self, max_size: int = MAX_STORED_RESPONSES):
        self._store: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()
        self._max_size = max_size

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (moves to end for LRU)."""
        if response_id in self._store:
            self._store.move_to_end(response_id)
            return self._store[response_id]
        return None

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        if response_id in self._store:
            self._store.move_to_end(response_id)
        self._store[response_id] = data
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        if response_id in self._store:
            del self._store[response_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers to every response; handle OPTIONS preflight."""
        if request.method == "OPTIONS":
            return web.Response(status=200, headers=_CORS_HEADERS)
        response = await handler(request)
        response.headers.update(_CORS_HEADERS)
        return response
else:
    cors_middleware = None  # type: ignore[assignment]


class APIServerAdapter(BasePlatformAdapter):
    """
    OpenAI-compatible HTTP API server adapter.

    Runs an aiohttp web server that accepts OpenAI-format requests
    and routes them through hermes-agent's AIAgent.
    """

    def __init__(self, config: PlatformConfig, session_store: Optional[SessionStore] = None):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port", os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))))
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        self._session_store: Optional[SessionStore] = session_store
        # Conversation name → latest response_id mapping
        self._conversations: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        If no API key is configured, all requests are allowed.
        """
        if not self._api_key:
            return None  # No key configured — allow all (local-only use)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token == self._api_key:
                return None  # Auth OK

        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        platform_hint: str = "api_server",
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.
        """
        from run_agent import AIAgent
        from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        model = _resolve_gateway_model()

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            session_id=session_id,
            platform=platform_hint,
            stream_delta_callback=stream_delta_callback,
        )
        return agent

    # History normalization helper
    # ------------------------------------------------------------------

    def _normalize_history(self, history: List[Dict]) -> List[Dict]:
        """
        Normalize transcript history for agent consumption.

        Copies the same logic used in the gateway runner:
        - skip entries without ``role``
        - skip ``role == "session_meta"``
        - skip ``role == "system"``
        - preserve rich tool messages (``tool_calls``, ``tool_call_id``,
          or ``role == "tool"``) after removing only ``timestamp``
        - reduce simple messages to ``{"role": role, "content": content}``
          only when content is a non-empty string
        """
        agent_history: List[Dict] = []
        for msg in history:
            role = msg.get("role")
            if not role:
                continue

            if role == "session_meta":
                continue

            if role == "system":
                continue

            has_tool_calls = "tool_calls" in msg
            has_tool_call_id = "tool_call_id" in msg
            is_tool_message = role == "tool"

            if has_tool_calls or has_tool_call_id or is_tool_message:
                clean_msg = {k: v for k, v in msg.items() if k != "timestamp"}
                agent_history.append(clean_msg)
            else:
                content = msg.get("content")
                if content:
                    agent_history.append({"role": role, "content": content})

        return agent_history

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "hermes-agent"})

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — return hermes-agent as an available model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": "hermes-agent",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hermes",
                    "permission": [],
                    "root": "hermes-agent",
                    "parent": None,
                }
            ],
        })

    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        stream = body.get("stream", False)

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                # Accumulate system messages
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role in ("user", "assistant"):
                conversation_messages.append({"role": role, "content": content})

        # Extract the last user message as the primary input
        user_message = ""
        history = []
        if conversation_messages:
            user_message = conversation_messages[-1].get("content", "")
            history = conversation_messages[:-1]

        if not user_message:
            return web.json_response(
                {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
                status=400,
            )

        session_id = str(uuid.uuid4())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", "hermes-agent")
        created = int(time.time())

        if stream:
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                _stream_q.put(delta)

            # Start agent in background
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
            ))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q, agent_task
            )

        # Non-streaming: run the agent and return full response
        try:
            result, usage = await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Error running agent for chat completions: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": f"Internal server error: {e}", "type": "server_error"}},
                status=500,
            )

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        return web.json_response(response_data)

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task,
    ) -> "web.StreamResponse":
        """Write real streaming SSE from agent's stream_delta_callback queue."""
        import queue as _q

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)

        # Role chunk
        role_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())

        # Stream content chunks as they arrive from the agent
        loop = asyncio.get_event_loop()
        while True:
            try:
                delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
            except _q.Empty:
                if agent_task.done():
                    # Drain any remaining items
                    while True:
                        try:
                            delta = stream_q.get_nowait()
                            if delta is None:
                                break
                            content_chunk = {
                                "id": completion_id, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                            }
                            await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
                        except _q.Empty:
                            break
                    break
                continue

            if delta is None:  # End of stream sentinel
                break

            content_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())

        # Get usage from completed agent
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            result, agent_usage = await agent_task
            usage = agent_usage or usage
        except Exception:
            pass

        # Finish chunk
        finish_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")

        return response

    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses — OpenAI Responses API format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        raw_input = body.get("input")
        if raw_input is None:
            return web.json_response(
                {"error": {"message": "Missing 'input' field", "type": "invalid_request_error"}},
                status=400,
            )

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = body.get("store", True)

        # conversation and previous_response_id are mutually exclusive
        if conversation and previous_response_id:
            return web.json_response(
                {"error": {"message": "Cannot use both 'conversation' and 'previous_response_id'", "type": "invalid_request_error"}},
                status=400,
            )

        # Resolve conversation name to latest response_id
        if conversation:
            previous_response_id = self._conversations.get(conversation)
            # No error if conversation doesn't exist yet — it's a new conversation

        # Normalize input to message list
        input_messages: List[Dict[str, str]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    # Handle content that may be a list of content parts
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "input_text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and part.get("type") == "output_text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = "\n".join(text_parts)
                    input_messages.append({"role": role, "content": content})
        else:
            return web.json_response(
                {"error": {"message": "'input' must be a string or array", "type": "invalid_request_error"}},
                status=400,
            )

        # Reconstruct conversation history from previous_response_id
        conversation_history: List[Dict[str, str]] = []
        if previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                return web.json_response(
                    {"error": {"message": f"Previous response not found: {previous_response_id}", "type": "invalid_request_error"}},
                    status=404,
                )
            conversation_history = list(stored.get("conversation_history", []))
            # If no instructions provided, carry forward from previous
            if instructions is None:
                instructions = stored.get("instructions")

        # Append new input messages to history (all but the last become history)
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        # Last input message is the user_message
        user_message = input_messages[-1].get("content", "") if input_messages else ""
        if not user_message:
            return web.json_response(
                {"error": {"message": "No user message found in input", "type": "invalid_request_error"}},
                status=400,
            )

        # Truncation support
        if body.get("truncation") == "auto" and len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        # Run the agent
        session_id = str(uuid.uuid4())
        try:
            result, usage = await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Error running agent for responses: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": f"Internal server error: {e}", "type": "server_error"}},
                status=500,
            )

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        created_at = int(time.time())

        # Build the full conversation history for storage
        # (includes tool calls from the agent run)
        full_history = list(conversation_history)
        full_history.append({"role": "user", "content": user_message})
        # Add agent's internal messages if available
        agent_messages = result.get("messages", [])
        if agent_messages:
            full_history.extend(agent_messages)
        else:
            full_history.append({"role": "assistant", "content": final_response})

        # Build output items (includes tool calls + final message)
        output_items = self._extract_output_items(result)

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": body.get("model", "hermes-agent"),
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        # Store the complete response object for future chaining / GET retrieval
        if store:
            self._response_store.put(response_id, {
                "response": response_data,
                "conversation_history": full_history,
                "instructions": instructions,
            })
            # Update conversation mapping so the next request with the same
            # conversation name automatically chains to this response
            if conversation:
                self._conversations[conversation] = response_id

        return web.json_response(response_data)

    # ------------------------------------------------------------------
    # GET / DELETE response endpoints
    # ------------------------------------------------------------------

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return web.json_response(
                {"error": {"message": f"Response not found: {response_id}", "type": "invalid_request_error"}},
                status=404,
            )

        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return web.json_response(
                {"error": {"message": f"Response not found: {response_id}", "type": "invalid_request_error"}},
                status=404,
            )

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_output_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the full output item array from the agent's messages.

        Walks *result["messages"]* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    items.append({
                        "type": "function_call",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        # Final assistant message
        final = result.get("final_response", "")
        if not final:
            final = result.get("error", "(No response generated)")

        items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": final,
                }
            ],
        })
        return items

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        platform_hint: str = "api_server",
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.
        """
        loop = asyncio.get_event_loop()

        def _run():
            agent = self._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
                platform_hint=platform_hint,
            )
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
            )
            result["session_id"] = getattr(agent, "session_id", session_id)
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return result, usage

        return await loop.run_in_executor(None, _run)


    # Session-backed message handler
    # ------------------------------------------------------------------

    async def _handle_message(self, request: "web.Request") -> "web.Response":
        """POST /v1/message — session-backed message endpoint."""
        # 1. Auth check
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # 2. Session store availability
        if self._session_store is None:
            return web.json_response(
                {
                    "error": {
                        "type": "server_error",
                        "message": "Session store not available",
                    }
                },
                status=500,
            )

        # 3. Parse JSON body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {
                    "error": {
                        "message": "Invalid JSON in request body",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )

        # 4. Require text — non-empty string after strip
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return web.json_response(
                {
                    "error": {
                        "message": "Missing or empty 'text' field",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )
        text = text.strip()

        # 5. Require chat_id
        chat_id_raw = body.get("chat_id")
        if not chat_id_raw:
            return web.json_response(
                {
                    "error": {
                        "message": "Missing 'chat_id' field",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )
        chat_id = str(chat_id_raw).strip()
        if not chat_id:
            return web.json_response(
                {
                    "error": {
                        "message": "Missing 'chat_id' field",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )

        # 6. Default user_id to normalized chat_id if not provided
        user_id = str(body.get("user_id", chat_id)).strip() or chat_id

        # 7. Default platform to "webhook" if not provided
        platform_name = body.get("platform", "webhook")

        # 8. Convert platform with Platform()
        try:
            platform_enum = Platform(platform_name)
        except (ValueError, KeyError):
            return web.json_response(
                {
                    "error": {
                        "message": f"Unknown platform: {platform_name}",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )

        # 9. Construct SessionSource
        source = SessionSource(
            platform=platform_enum,
            chat_id=chat_id,
            chat_type="dm",
            user_id=user_id,
        )

        # 10. Get or create session
        session_entry = self._session_store.get_or_create_session(source)

        # 11. Load transcript
        history = self._session_store.load_transcript(session_entry.session_id)

        # 12. Normalize history
        agent_history = self._normalize_history(history)

        # 13. Record offset
        history_offset = len(agent_history)

        # 14. Run agent
        try:
            result, usage = await self._run_agent(
                user_message=text,
                conversation_history=agent_history,
                session_id=session_entry.session_id,
                platform_hint=platform_enum.value,
            )
        except Exception as e:
            logger.error(
                "Error running agent for message endpoint: %s", e, exc_info=True
            )
            return web.json_response(
                {
                    "error": {
                        "message": f"Internal server error: {e}",
                        "type": "server_error",
                    }
                },
                status=500,
            )

        # 15. Handle session-id rollover
        effective_session_id = session_entry.session_id
        result_session_id = result.get("session_id")
        if result_session_id and result_session_id != session_entry.session_id:
            session_entry.session_id = result_session_id
            effective_session_id = result_session_id
            self._session_store._save()

        # 16. Extract new messages
        all_messages = result.get("messages", [])
        if len(all_messages) > history_offset:
            new_messages = all_messages[history_offset:]
        else:
            new_messages = []

        # 17. Compute reply_text
        reply_text = None
        for msg in reversed(new_messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    reply_text = content
                    break
        if reply_text is None:
            reply_text = (
                result.get("final_response")
                or result.get("error")
                or "(No response generated)"
            )

        # 18. Timestamp
        ts = datetime.now().isoformat()

        # 19-20. Append to transcript
        if not new_messages:
            self._session_store.append_to_transcript(
                effective_session_id,
                {
                    "role": "user",
                    "content": text,
                    "timestamp": ts,
                },
            )
            self._session_store.append_to_transcript(
                effective_session_id,
                {
                    "role": "assistant",
                    "content": reply_text,
                    "timestamp": ts,
                },
            )
        else:
            for msg in new_messages:
                if msg.get("role") == "system":
                    continue
                self._session_store.append_to_transcript(
                    effective_session_id,
                    {
                        **msg,
                        "timestamp": ts,
                    },
                )

        # 21. Return response
        return web.json_response(
            {
                "response": reply_text,
                "session_id": effective_session_id,
                "ok": True,
            }
        )

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        try:
            self._app = web.Application(middlewares=[cors_middleware])
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
            self._app.router.add_post("/v1/responses", self._handle_responses)
            self._app.router.add_post("/v1/message", self._handle_message)
            self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
            self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            self._mark_connected()
            logger.info(
                "[%s] API server listening on http://%s:%d",
                self.name, self._host, self._port,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server."""
        self._mark_disconnected()
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Not used — HTTP request/response cycle handles delivery directly.
        """
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }
