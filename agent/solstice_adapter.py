"""Solstice per-user-quota transport.

Lets a user bring their own Solstice subscription into Hermes. Inference is
authenticated with the user's OAuth access token (Bearer), not a static API
key; the ``peruserquota`` scope maps usage to the signed-in user's own quota.
The wire shape is the ``GenerateContent`` contract served on the
``:generateContentPerUserQuota`` variants.

Reuses the native adapter's message/tool translation, stream handling and HTTP
error mapping wholesale; only auth and the endpoint suffix differ:
  - auth = Bearer OAuth token (the user's OAuth access token), not API key
  - endpoint = ``...:generateContentPerUserQuota`` (non-stream) /
    ``...:streamGenerateContentPerUserQuota`` (stream)
  - base = ``https://generativelanguage.googleapis.com/v1alpha``

Exposes the OpenAI-SDK-compatible facade (``chat.completions.create`` +
``.stream()``) so it drops into the same transport seams, and is constructed by
the same ``agent_runtime_helpers`` seam.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional

import httpx

from agent.gemini_native_adapter import (
    GeminiNativeClient,
    _GeminiStreamChunk,
    bare_gemini_model_id,
    build_gemini_request,
)

logger = logging.getLogger(__name__)

# Inference host — single source of truth lives in
# hermes_cli/solstice.py so a launch-time host correction is one edit.
from hermes_cli.solstice import (  # noqa: E402
    SOLSTICE_INFERENCE_BASE_URL as SOLSTICE_BASE_URL,
)

# Per-user-quota variant endpoints.  The stream variant follows the native
# ``:streamGenerateContent`` naming convention and is kept as a constant so a
# correction is one line.
SOLSTICE_GENERATE_PATH = ":generateContentPerUserQuota"
SOLSTICE_STREAM_PATH = ":streamGenerateContentPerUserQuota"


def build_solstice_request(
    *,
    model: str,
    messages: list[Dict[str, Any]],
    tools: Any = None,
    tool_choice: Any = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Any = None,
    thinking_config: Any = None,
) -> Dict[str, Any]:
    """Build the standard body for the per-user-quota endpoint.

    Identical to ``build_gemini_request`` (the endpoint takes
    the standard ``GenerateContent`` body); this wrapper exists so the adapter
    has a single, explicit construction point and a stable name.
    """
    return build_gemini_request(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        thinking_config=thinking_config,
    )


class SolsticeClient:
    """OpenAI-SDK-compatible facade over the per-user-quota API.

    Auth is a **Bearer** OAuth access token (the user's Solstice
    subscription), not an API key. Reuses the native adapter's translation
    wholesale; only the auth and endpoint differ.
    """

    def __init__(
        self,
        *,
        api_key: str,  # the OAuth access token (Bearer)
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
        http_client: Optional[httpx.Client] = None,
        **_: Any,
    ) -> None:
        if not (api_key or "").strip():
            raise RuntimeError(
                "Solstice client requires an OAuth access token, but none was "
                "provided. Run `hermes auth add solstice` to sign in."
            )
        self.api_key = api_key
        self.base_url = (base_url or SOLSTICE_BASE_URL).rstrip("/")
        self._default_headers = dict(default_headers or {})
        self._http = http_client or httpx.Client(
            timeout=timeout
            or httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)
        )
        self.is_closed = False
        self.chat = _SolsticeChatNamespace(self)

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self) -> "SolsticeClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(self._default_headers)
        return headers

    def _create_chat_completion(
        self,
        *,
        model: str = "gemini-flash-latest",
        messages: Optional[list[Dict[str, Any]]] = None,
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
        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = extra_body.get("thinking_config") or extra_body.get(
                "thinkingConfig"
            )

        body = build_solstice_request(
            model=model,
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking_config=thinking_config,
        )

        if stream:
            return self._stream_completion(
                model=model, body=body, timeout=timeout
            )

        url = f"{self.base_url}/models/{bare_gemini_model_id(model)}{SOLSTICE_GENERATE_PATH}"
        response = self._http.post(
            url, json=body, headers=self._headers(), timeout=timeout
        )
        if response.status_code != 200:
            from agent.gemini_native_adapter import gemini_http_error

            raise gemini_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON from Solstice API: {exc}") from exc

        from agent.gemini_native_adapter import translate_gemini_response

        return translate_gemini_response(payload, model=model)

    def _stream_completion(
        self, *, model: str, body: Dict[str, Any], timeout: Any = None
    ) -> Iterator[_GeminiStreamChunk]:
        from agent.bounded_response import read_streaming_error_body
        from agent.gemini_native_adapter import (
            _iter_sse_events,
            gemini_http_error,
            translate_stream_event,
        )

        # alt=sse is REQUIRED. Without it the per-user-quota stream endpoint
        # returns a bare JSON array of complete responses (``[{...},{...}]``),
        # not SSE — ``_iter_sse_events`` finds no ``data:`` lines, yields
        # nothing, and the caller reports a bogus EmptyStreamError. With it the
        # response is real ``data: {...}`` SSE, matching the native adapter.
        url = (
            f"{self.base_url}/models/{bare_gemini_model_id(model)}"
            f"{SOLSTICE_STREAM_PATH}?alt=sse"
        )
        stream_headers = dict(self._headers())
        stream_headers["Accept"] = "text/event-stream"

        def _generator() -> Iterator[_GeminiStreamChunk]:
            try:
                with self._http.stream(
                    "POST", url, json=body, headers=stream_headers, timeout=timeout
                ) as response:
                    if response.status_code != 200:
                        body_text = read_streaming_error_body(response)
                        raise gemini_http_error(response, body_text=body_text)
                    tool_call_indices: Dict[str, Dict[str, Any]] = {}
                    for event in _iter_sse_events(response):
                        for chunk in translate_stream_event(
                            event, model, tool_call_indices
                        ):
                            yield chunk
            finally:
                pass

        return _generator()


class _SolsticeChatCompletions:
    """``client.chat.completions`` namespace (sync)."""

    def __init__(self, client: SolsticeClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)

    def stream(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(stream=True, **kwargs)


class _SolsticeChatNamespace:
    def __init__(self, client: SolsticeClient) -> None:
        self.completions = _SolsticeChatCompletions(client)


__all__ = [
    "SolsticeClient",
    "build_solstice_request",
    "SOLSTICE_BASE_URL",
]
