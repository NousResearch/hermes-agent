"""Solstice inference transport: the shared Interactions wire on the per-user-quota method
(``POST <host>/v1beta/interactions:createPerUserQuota``), with the user's OAuth access token as a Bearer.

That method takes only stateless requests (``store: false``; a stored interaction is a 400), so every turn
is one request: the output budget is capped at one request's, the request is never stored, and a stream
that ends before ``interaction.completed`` is an error rather than a finished turn. There is no
continuation, checkpoint or resume handling here; that belongs to the gemini provider's long decode.
Request building, thought signature replay, stream translation and error mapping come from the sibling
``interactions`` and ``native_adapter`` modules in this package.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Optional

import httpx

from agent.bounded_response import read_streaming_error_body
from agent.reasoning_effort import bare_gemini_model_id
from hermes_cli.version_info import hermes_user_agent

from . import interactions as wire
from .account import INFERENCE_BASE_URL, current_model_id
from .native_adapter import GeminiAPIError, GeminiNativeClient, _iter_sse_events

INTERACTIONS_METHOD = "interactions:createPerUserQuota"
# The method exists only on v1beta. Credential rows minted before the move store the old
# ``.../v1alpha`` base, so the version segment of any stored base is replaced, never trusted.
_API_VERSION = "v1beta"
_VERSION_SEGMENT = re.compile(r"/v\d+(?:(?:alpha|beta)\d*)?\Z", re.IGNORECASE)
# A 3600 s read timeout: a Gemini 4 turn can think for up to an hour.
DEFAULT_TIMEOUT = httpx.Timeout(connect=15.0, read=3600.0, write=30.0, pool=30.0)
# Never sent with a per-user-quota bearer (the first three), or set here and not overridable (the rest).
_DROPPED_DEFAULT_HEADERS = {"x-goog-api-key", "x-goog-user-project", "dpop", "authorization", "user-agent"}


def interactions_url(base_url: Any) -> str:
    root = _VERSION_SEGMENT.sub("", str(base_url or INFERENCE_BASE_URL).strip().rstrip("/"))
    return f"{root}/{_API_VERSION}/{INTERACTIONS_METHOD}"


def single_request(request: dict[str, Any], cap: int) -> dict[str, Any]:
    """*request* as one stateless call: the final output budget capped at *cap*, one request's (the builder
    raises it for long decode and then sets ``store: true``), and ``store`` forced off."""
    generation = request["generation_config"]
    budget = min(int(generation["max_output_tokens"]), cap)
    return {**request, "store": False, "generation_config": {**generation, "max_output_tokens": budget}}


class SolsticeClient(GeminiNativeClient):
    """One stateless Interactions request per call, on ``interactions:createPerUserQuota``."""

    MISSING_KEY_ERROR = "Solstice needs a signed-in account. Run `hermes auth add solstice`."

    def __init__(self, *, base_url: Any = None, timeout: Any = None, **kwargs: Any) -> None:
        # No base means this endpoint, never the generic Gemini default (a different API surface).
        super().__init__(base_url=base_url or INFERENCE_BASE_URL, timeout=timeout, **kwargs)
        # Per request, not only on the httpx client: an injected keepalive client brings its own default.
        self._request_timeout = timeout or DEFAULT_TIMEOUT

    def _create_chat_completion(
        self, *, model: str = "gemini-3.8-flash", messages: Optional[list[dict[str, Any]]] = None, stream: bool = False,
        tools: Any = None, tool_choice: Any = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        top_p: Optional[float] = None, stop: Any = None, response_format: Any = None,
        extra_body: Optional[dict[str, Any]] = None, timeout: Any = None, **_: Any,
    ) -> Any:
        # ``temperature``/``top_p`` are never sent: Gemini 4 / 3.5 reject them. A long-decode budget in
        # ``extra_body`` is ignored: this method cannot continue a turn past one request.
        extra = extra_body if isinstance(extra_body, dict) else {}
        # A retired id goes out as the default model (also mapped in build_extra_body and normalize_model_id).
        model = current_model_id(model) or bare_gemini_model_id(model)
        request = single_request(wire.build_interactions_request(
            model=model, messages=messages or [], tools=tools, tool_choice=tool_choice, max_tokens=max_tokens,
            stop=stop, thinking_config=extra.get("thinking_config") or extra.get("thinkingConfig"),
            response_format=response_format or extra.get("response_format"),
        ), wire.SINGLE_REQUEST_OUTPUT_CAP)
        # An explicit per-call timeout (the agent's configured / stale-matched budget, zero included) wins;
        # else 3600 s read.
        timeout = self._request_timeout if timeout is None else timeout
        url = interactions_url(self.base_url)
        if stream:
            return self._stream(model, url, request, timeout)
        # The unary call: the method takes both modes, so the flag is always named.
        return wire.translate_interaction([self._generate(url, {**request, "stream": False}, timeout)], model)

    def _stream(self, model: str, url: str, request: dict[str, Any], timeout: Any) -> Iterator[Any]:
        state = wire.StreamState()
        headers = {**self._headers(), "Accept": "text/event-stream"}
        try:
            with self._http.stream("POST", url, json={**request, "stream": True}, headers=headers, timeout=timeout) as response:
                if response.status_code != 200:
                    raise self._http_error(response, read_streaming_error_body(response))
                for event in wire.iter_events(_iter_sse_events(response)):
                    yield from wire.translate_stream_event(event, model, state)
        except httpx.HTTPError as exc:  # nothing is stored to resume from: the turn is retried from the start
            raise GeminiAPIError(f"Gemini streaming request failed: {exc}", code="gemini_stream_error") from exc
        if not state.hop_done:  # closed cleanly mid-turn: the partial turn must not stand
            raise GeminiAPIError(f"Gemini stream ended before interaction.completed (status {state.status or 'unknown'})",
                                 code="gemini_stream_error")
        if state.status == "requires_continuation":  # never continued here: the output was cut short
            yield wire.finish_ceiling(model, state)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _headers(self) -> dict[str, str]:
        # Bearer, content type and User-Agent only: no API key, user-project or DPoP header, nor Google's client tag.
        extra = {k: v for k, v in self._default_headers.items() if k.lower() not in _DROPPED_DEFAULT_HEADERS}
        return {**extra, "Content-Type": "application/json", "Accept": "application/json", **self._auth_headers(),
                "User-Agent": hermes_user_agent("hermes-agent")}

    def _http_error(self, response: Any, body_text: Any = None) -> GeminiAPIError:
        # The Interactions mapper (snake_case error codes), without the API-key remedies (free tier,
        # Standard key, wrong key surface): they do not apply to a bearer.
        return wire.http_error(
            response, body_text=body_text, api_key=self.api_key, base_url=self.base_url, key_guidance=False)
