"""The ``gemini`` provider's client: each call goes to the Interactions API or to ``generateContent``.

Which one is per model. ``providers.gemini.models.<model>.gemini_api`` wins over
``providers.gemini.gemini_api``; each is ``interactions``, ``generate_content`` or ``auto`` (the default).
``auto`` uses Interactions for Gemini 3 and later (and the ``gemini-*-latest`` aliases), Google's recommended
API for them. Gemini 2.x stays on ``generateContent``: it steers thinking with ``thinkingBudget``, which
Interactions has no field for. So do non-Gemini ids (Gemma). Interactions exists only on the AI Studio host,
so a Vertex express or proxy base always stays on ``generateContent``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

import httpx

from agent.bounded_response import read_streaming_error_body
from agent.gemini_long_decode import long_decode_hops
from agent.gemini_native_adapter import (
    _MAX_TOKEN_SENDS,
    GeminiAPIError,
    GeminiNativeClient,
    _iter_sse_events,
    bare_gemini_model_id,
    gemini_uses_thinking_level,
)

from . import interactions as wire

logger = logging.getLogger(__name__)

PROVIDER_ID = "gemini"
API_INTERACTIONS, API_GENERATE_CONTENT, API_AUTO = "interactions", "generate_content", "auto"
_INTERACTIONS_HOST = "generativelanguage.googleapis.com"


def _configured_api(model: str, bare: str) -> str:
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:  # an unreadable config keeps the default routing; never fail the request over it
        logger.debug("gemini_api: config unreadable, using auto", exc_info=True)
        return API_AUTO
    provider = ((config or {}).get("providers") or {}).get(PROVIDER_ID) if isinstance(config, dict) else None
    if not isinstance(provider, dict):
        return API_AUTO
    models = provider.get("models")
    # The id as configured (``google/gemini-3.8-flash``) first, then its bare form.
    per_model = (models.get(model) or models.get(bare)) if isinstance(models, dict) else None
    for source in (per_model, provider):  # the first valid setting wins, an explicit per-model ``auto`` included
        value = source.get("gemini_api") if isinstance(source, dict) else None
        if isinstance(value, str) and (value := value.strip().lower()) in (API_INTERACTIONS, API_GENERATE_CONTENT, API_AUTO):
            return value
    return API_AUTO


def uses_interactions(model: str, base_url: str) -> bool:
    if _INTERACTIONS_HOST not in str(base_url or "").lower():
        return False
    bare = bare_gemini_model_id(model)
    choice = _configured_api((model or "").strip(), bare)
    # ``auto``: the models that take ``thinking_level`` (Gemini 3+ and the ``-latest`` aliases).
    return choice == API_INTERACTIONS or (choice == API_AUTO and gemini_uses_thinking_level(bare))


class GeminiClient(GeminiNativeClient):
    """``GeminiNativeClient`` (generateContent) plus the Interactions wire, chosen per call by model."""

    def _http_error(self, response: httpx.Response, body_text: Optional[str] = None) -> GeminiAPIError:
        request = getattr(response, "_request", None)  # set for every real response; absent on bare test doubles
        if request is not None and request.url.path.endswith("/interactions"):
            return wire.http_error(response, body_text=body_text, api_key=self.api_key, base_url=self.base_url)
        return super()._http_error(response, body_text)

    def _create_chat_completion(self, *, model: str = "gemini-3.8-flash", **kwargs: Any) -> Any:
        if not uses_interactions(model, self.base_url):
            return super()._create_chat_completion(model=model, **kwargs)
        return self._interactions_completion(model=model, **kwargs)

    def _interactions_completion(
        self, *, model: str, messages: Optional[list[dict[str, Any]]] = None, stream: bool = False, tools: Any = None,
        tool_choice: Any = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        top_p: Optional[float] = None, stop: Any = None, response_format: Any = None,
        extra_body: Optional[dict[str, Any]] = None, timeout: Any = None, **_: Any,
    ) -> Any:
        extra = extra_body if isinstance(extra_body, dict) else {}
        budget = extra.get("max_output_tokens")
        if isinstance(budget, int) and budget > int(max_tokens or 0):
            max_tokens = budget  # long decode: a larger total budget than one request's cap
        model = bare_gemini_model_id(model)
        request = wire.build_interactions_request(
            model=model, messages=messages or [], tools=tools, tool_choice=tool_choice, temperature=temperature,
            max_tokens=max_tokens, top_p=top_p, stop=stop,
            thinking_config=extra.get("thinking_config") or extra.get("thinkingConfig"),
            response_format=response_format or extra.get("response_format"),
        )
        url = f"{self.base_url}/interactions"
        max_hops = long_decode_hops(request["generation_config"]["max_output_tokens"]) + 1
        if stream:
            return self._interactions_stream(model, url, request, timeout, max_hops)
        hops = [self._generate(url, request, timeout)]
        while self._continues(hops[-1]) and len(hops) < max_hops:
            hops.append(self._generate(url, wire.continuation_request(request, hops[-1]["id"], hops[-1]["continuation_token"]), timeout))
        return wire.translate_interaction(hops, model)

    @staticmethod
    def _continues(interaction: dict[str, Any]) -> bool:
        return (interaction.get("status") == "requires_continuation" and bool(interaction.get("id"))
                and isinstance(interaction.get("continuation_token"), str) and bool(interaction["continuation_token"]))

    def _interactions_stream(self, model: str, url: str, request: dict[str, Any], timeout: Any, max_hops: int) -> Iterator[Any]:
        state = wire.StreamState()
        body = {**request, "stream": True}
        # ``resume``: ``(interaction id, token)`` of the latest token (a hop's end or a mid-stream checkpoint)
        # with nothing yielded after it. The next hop, or a dropped stream, continues the request from it;
        # a later chunk voids it, since resuming would repeat that chunk. ``sends`` bounds a stuck token.
        resume: Optional[tuple[str, str]] = None
        sends = 0
        while True:
            dropped: Optional[httpx.HTTPError] = None
            try:
                headers = {**self._headers(), "Accept": "text/event-stream"}
                with self._http.stream("POST", url, json=body, headers=headers, timeout=timeout) as response:
                    if response.status_code != 200:
                        raise self._http_error(response, read_streaming_error_body(response))
                    for event in wire.iter_events(_iter_sse_events(response)):
                        chunks = wire.translate_stream_event(event, model, state)
                        if chunks:
                            resume = None
                        elif (token := wire.event_token(event)) and state.interaction_id:
                            resume, sends = (state.interaction_id, token), 0
                        yield from chunks
            except httpx.HTTPError as exc:
                dropped = exc
            if state.hop_done and state.status != "requires_continuation":
                return  # the finish chunk went out with ``interaction.completed``
            if resume is None or sends >= _MAX_TOKEN_SENDS or (state.hop_done and state.completed_hops >= max_hops):
                if state.hop_done:
                    yield wire.finish_ceiling(model, state)  # nothing to continue from, or the hop ceiling: cut short
                elif dropped is not None:  # nothing committed to resume from: the turn is retried from the start
                    raise GeminiAPIError(f"Gemini streaming request failed: {dropped}", code="gemini_stream_error") from dropped
                return
            if not state.hop_done:
                logger.warning("Gemini stream dropped (%s); resuming from its last checkpoint", dropped or "ended early")
                state.close_dropped_hop()
            body = {**wire.continuation_request(request, *resume), "stream": True}
            sends += 1
            state.next_hop()
