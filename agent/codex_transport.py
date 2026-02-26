"""Custom httpx transport that rewrites requests to the Codex ChatGPT endpoint.

When the OpenAI SDK makes a request to /chat/completions or /v1/responses,
this transport rewrites the URL to https://chatgpt.com/backend-api/codex/responses,
injects the Bearer token + ChatGPT-Account-Id header, and handles 401 → refresh → retry.

This approach (inspired by opencode) means the rest of the codebase doesn't need
to know about Codex at all — no format translation, no branching in the agent loop.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from agent.codex_auth import refresh_codex_chatgpt_auth

logger = logging.getLogger(__name__)

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


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

    def _rewrite(self, request: httpx.Request) -> httpx.Request:
        return httpx.Request(
            method=request.method,
            url=CODEX_RESPONSES_URL,
            headers=request.headers,
            content=request.content,
            extensions=request.extensions,
        )

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

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self._should_rewrite(request):
            request = self._rewrite(request)
        self._apply_headers(request)

        response = self._inner.handle_request(request)
        if response.status_code == 401 and self._refresh():
            self._apply_headers(request)
            response = self._inner.handle_request(request)
        return response

    def close(self) -> None:
        self._inner.close()
