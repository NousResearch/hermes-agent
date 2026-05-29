"""OpenAI-Compatible Local Endpoint Server.

Provides a local API gateway compatible with OpenAI's /v1/models and
/v1/chat/completions. Runs in a non-blocking background thread.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from provider_gateway.config import load_gateway_config
from provider_gateway.policy import build_gateway_policy
from provider_gateway.runtime import get_discovered_ollama_models, get_secure_store

logger = logging.getLogger(__name__)

# Optional Bearer token auth — set HERMES_GATEWAY_TOKEN env var to enable.
# When unset, authentication is disabled (open access from localhost only).
_GATEWAY_TOKEN: str = os.environ.get("HERMES_GATEWAY_TOKEN", "").strip()


class GatewayHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible API requests for the provider gateway."""

    def log_message(self, format: str, *args: Any) -> None:
        # Override to suppress default stdout spam of http.server
        logger.debug(format, *args)

    def do_GET(self) -> None:
        """Handle GET requests, mainly /v1/models."""
        if not self._check_auth():
            return
        if self.path == "/v1/models":
            self._handle_models()
        else:
            self._send_error(404, "Not Found")

    def do_POST(self) -> None:
        """Handle POST requests, mainly /v1/chat/completions."""
        if not self._check_auth():
            return
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_error(404, "Not Found")

    def _check_auth(self) -> bool:
        """Validate Bearer token if HERMES_GATEWAY_TOKEN is configured.

        Returns True if auth is disabled or token matches. Sends 401 and
        returns False otherwise.
        """
        if not _GATEWAY_TOKEN:
            return True  # Auth disabled — open localhost access
        auth_header = self.headers.get("Authorization", "")
        if auth_header == f"Bearer {_GATEWAY_TOKEN}":
            return True
        self._send_error(401, "Unauthorized: invalid or missing Bearer token")
        return False

    def _send_error(self, status_code: int, message: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": {"message": message}}).encode("utf-8"))

    def _handle_models(self) -> None:
        """List active gateway models and auto-discovered local Ollama models."""
        try:
            config = load_gateway_config()
            models_data = []

            # Add fallback models from config
            for model in config.fallback_models:
                models_data.append({
                    "id": model,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "provider_gateway"
                })

            # Add discovered Ollama models
            for lm in get_discovered_ollama_models():
                models_data.append({
                    "id": lm["model"],
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "ollama"
                })

            # Fallback default if list is empty
            if not models_data:
                models_data.append({
                    "id": "gpt-4o",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "openai"
                })

            response = {
                "object": "list",
                "data": models_data
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        except Exception as exc:
            logger.error("Failed to list models: %s", exc)
            self._send_error(500, f"Internal server error: {exc}")

    def _handle_chat_completions(self) -> None:
        """Process chat completions with optional streaming and PII guardrails."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            req_body = json.loads(post_data.decode("utf-8"))
        except Exception:
            self._send_error(400, "Invalid JSON body")
            return

        model = req_body.get("model", "")
        messages = req_body.get("messages", [])
        stream_requested = bool(req_body.get("stream", False))

        # 1. Resolve Route Candidates
        config = load_gateway_config()
        policy = build_gateway_policy(None, config)
        
        selected_route = None
        # Try to find a direct candidate matching the requested model
        for candidate in policy.candidates:
            if candidate.model == model or (candidate.provider + "/" + candidate.model) == model:
                selected_route = candidate
                break

        # Fallback to the first healthy/available dynamic candidate if no specific match
        if selected_route is None:
            non_primary_candidates = [c for c in policy.candidates if c.source != "primary"]
            if non_primary_candidates:
                selected_route = non_primary_candidates[0]
            elif policy.candidates:
                selected_route = policy.candidates[0]

        if selected_route is None:
            from provider_gateway.policy import ProviderRouteCandidate
            selected_route = ProviderRouteCandidate(
                provider="openai" if "gpt" in model else "unknown",
                model=model,
                source="server_default_fallback",
                base_url=None,
            )

        # 2. Get API credentials
        store = get_secure_store()
        api_key = store.get_credential(selected_route.provider)
        if not api_key:
            api_key = selected_route.api_key or "dummy-api-key"

        base_url = selected_route.base_url

        # 3. Apply PII Guardrails Preflight — fresh instance per request for thread safety
        sanitizer = None
        deanonimizer = None
        if config.enabled and config.guardrails_enabled:
            try:
                from provider_gateway.guardrails import PIISanitizer
                sanitizer = PIISanitizer()  # Per-request instance — no cross-request data leaks
                deanonimizer = sanitizer.get_deanonimizer()
                
                # Sanitize prompt messages
                for msg in messages:
                    if "content" in msg and isinstance(msg["content"], str):
                        msg["content"] = sanitizer.sanitize_prompt(msg["content"])
                req_body["messages"] = messages
            except Exception as pii_exc:
                logger.debug("Failed to apply guardrails preflight: %s", pii_exc)

        # 4. Prepare Client — lazy import to avoid mandatory openai dependency
        import openai as _openai

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        client = _openai.OpenAI(**client_kwargs)

        # Prepare request payload for OpenAI API
        api_payload: dict[str, Any] = {
            "model": selected_route.model,
            "messages": messages,
            "stream": stream_requested,
        }
        for k in ["temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"]:
            if k in req_body:
                api_payload[k] = req_body[k]

        # 5. Execute and respond
        try:
            if stream_requested:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                response_stream = client.chat.completions.create(**api_payload)
                for chunk in response_stream:
                    # Parse and apply PII Deanonimization if active
                    chunk_json = chunk.model_dump()
                    if deanonimizer is not None and chunk_json.get("choices"):
                        choices = chunk_json["choices"]
                        if choices and "delta" in choices[0] and "content" in choices[0]["delta"]:
                            content = choices[0]["delta"]["content"]
                            if content:
                                choices[0]["delta"]["content"] = deanonimizer.process_chunk(content)
                    
                    # Send standard OpenAI SSE event
                    self.wfile.write(f"data: {json.dumps(chunk_json)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                # Flush guardrail buffer if any
                if deanonimizer is not None:
                    flush_content = deanonimizer.flush()
                    if flush_content:
                        flush_chunk = {
                            "choices": [{
                                "index": 0,
                                "delta": {"content": flush_content},
                                "finish_reason": None
                            }]
                        }
                        self.wfile.write(f"data: {json.dumps(flush_chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                resp = client.chat.completions.create(**api_payload)
                resp_json = resp.model_dump()

                # Apply PII Guardrails Postflight
                if sanitizer is not None:
                    for choice in resp_json.get("choices", []):
                        msg = choice.get("message", {})
                        if "content" in msg and msg["content"]:
                            msg["content"] = sanitizer.restore_response(msg["content"])

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp_json).encode("utf-8"))

        except Exception as api_exc:
            logger.error("API proxy error: %s", api_exc)
            self._send_error(502, f"Bad gateway (provider error): {api_exc}")


def start_gateway_server(port: int = 8000, background: bool = True) -> ThreadingHTTPServer:
    """Start the local OpenAI-compatible endpoint server."""
    server = ThreadingHTTPServer(("127.0.0.1", port), GatewayHTTPRequestHandler)
    
    if background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info("Provider gateway server started on 127.0.0.1:%d (background thread)", port)
    else:
        logger.info("Provider gateway server started on 127.0.0.1:%d (blocking)", port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
            
    return server


def stop_gateway_server(server: ThreadingHTTPServer) -> None:
    """Stop the running local server instance."""
    if server:
        server.shutdown()
        server.server_close()
        logger.info("Provider gateway server stopped successfully")
