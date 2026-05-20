"""OpenAI-compatible facade for Antigravity native OAuth inference."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import httpx

from agent import antigravity_oauth
from agent.antigravity_code_assist import (
    ANTIGRAVITY_CODE_ASSIST_ENDPOINT,
    CodeAssistError,
    ProjectContext,
    build_headers,
    resolve_project_context,
)
from agent.gemini_cloudcode_adapter import (
    GeminiCloudCodeClient,
    _GeminiStreamChunk,
    _gemini_http_error,
    _iter_sse_events,
    _translate_gemini_response,
    _translate_stream_event,
    build_gemini_request,
    wrap_code_assist_request,
)

MARKER_BASE_URL = "antigravity-pa://google"


class AntigravityCloudCodeClient(GeminiCloudCodeClient):
    """Minimal OpenAI-SDK-compatible facade over Antigravity Code Assist."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        project_id: str = "",
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key or "antigravity-oauth",
            base_url=base_url or MARKER_BASE_URL,
            default_headers=default_headers,
            project_id=project_id,
            **kwargs,
        )

    def _ensure_project_context(self, access_token: str, model: str) -> ProjectContext:
        if self._project_context is not None:
            return self._project_context  # type: ignore[return-value]

        env_project = antigravity_oauth.resolve_project_id_from_env()
        creds = antigravity_oauth.load_credentials()
        stored_project = creds.project_id if creds else ""
        if stored_project:
            self._project_context = ProjectContext(
                project_id=stored_project,
                managed_project_id=creds.managed_project_id if creds else "",
                source="stored",
            )
            return self._project_context

        ctx = resolve_project_context(
            access_token,
            configured_project_id=self._configured_project_id,
            env_project_id=env_project,
        )
        if ctx.project_id or ctx.managed_project_id:
            antigravity_oauth.update_project_ids(
                project_id=ctx.project_id,
                managed_project_id=ctx.managed_project_id,
            )
        self._project_context = ctx
        return ctx

    def _create_chat_completion(
        self,
        *,
        model: str = "gemini-3-flash-agent",
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Any = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        access_token = antigravity_oauth.get_valid_access_token()
        ctx = self._ensure_project_context(access_token, model)

        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = extra_body.get("thinking_config") or extra_body.get("thinkingConfig")

        inner = build_gemini_request(
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking_config=thinking_config,
        )
        wrapped = wrap_code_assist_request(
            project_id=ctx.project_id,
            model=model,
            inner_request=inner,
        )

        headers = build_headers(access_token)
        headers.update(self._default_headers)

        if stream:
            return self._stream_completion(model=model, wrapped=wrapped, headers=headers)

        url = f"{ANTIGRAVITY_CODE_ASSIST_ENDPOINT}/v1internal:generateContent"
        response = self._http.post(url, json=wrapped, headers=headers)
        if response.status_code != 200:
            raise _gemini_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise CodeAssistError(
                f"Invalid JSON from Antigravity Code Assist: {exc}",
                code="antigravity_code_assist_invalid_json",
            ) from exc
        return _translate_gemini_response(payload, model=model)

    def _stream_completion(
        self,
        *,
        model: str,
        wrapped: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Iterator[_GeminiStreamChunk]:
        url = f"{ANTIGRAVITY_CODE_ASSIST_ENDPOINT}/v1internal:streamGenerateContent?alt=sse"
        stream_headers = dict(headers)
        stream_headers["Accept"] = "text/event-stream"

        def _generator() -> Iterator[_GeminiStreamChunk]:
            try:
                with self._http.stream("POST", url, json=wrapped, headers=stream_headers) as response:
                    if response.status_code != 200:
                        response.read()
                        raise _gemini_http_error(response)
                    tool_call_counter: List[int] = [0]
                    for event in _iter_sse_events(response):
                        for chunk in _translate_stream_event(event, model, tool_call_counter):
                            yield chunk
            except httpx.HTTPError as exc:
                raise CodeAssistError(
                    f"Antigravity streaming request failed: {exc}",
                    code="antigravity_code_assist_stream_error",
                ) from exc

        return _generator()
