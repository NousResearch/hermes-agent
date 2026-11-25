"""Solstice inference transport: the gemini plugin's Interactions wire on the per-user-quota method
(``POST <host>/v1beta/interactions:createPerUserQuota``), with the user's OAuth access token as a Bearer.

That method takes only stateless requests (``store: false``; a stored interaction is a 400), so every turn
is one request: the output budget is capped at one request's, the request is never stored, and a stream
that ends before ``interaction.completed`` is an error rather than a finished turn. There is no
continuation, checkpoint or resume handling here; that belongs to the gemini provider's long decode.
Request building, thought signature replay, stream translation and error mapping are the shared wire's,
so a fix there reaches this provider too. That wire is the gemini provider plugin's ``interactions`` module,
found through provider discovery (``interactions_wire``), not a relative import: a copy of this plugin in a
profile's own ``plugins/model-providers/`` loads under a top-level module name, with no sibling package.
The stream helpers (SSE event iterator, bounded error-body read) come from that plugin's ``client`` module,
found the same way, so this plugin imports only published core names besides its base client and error type.
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from types import ModuleType
from typing import Any, Iterator, Optional

import httpx

from agent.gemini_native_adapter import GeminiAPIError, GeminiNativeClient
from agent.reasoning_effort import bare_gemini_model_id
from hermes_cli.version_info import hermes_user_agent

from . import INFERENCE_BASE_URL, current_model_id

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


_WIRE_MISSING = ("Solstice sends its requests through the gemini provider plugin's Interactions wire, but {why}. "
                 "Reinstall Hermes, or remove a gemini provider plugin that replaces the bundled one.")


# What this transport calls on the discovered wire. A gemini plugin copy from before these existed (an older
# profile-local copy) is refused here, before any request, instead of failing mid-call.
_WIRE_NAMES = ("build_interactions_request", "finish_ceiling", "iter_events", "SINGLE_REQUEST_OUTPUT_CAP",
               "StreamState", "translate_interaction", "translate_stream_event", "http_error")


def interactions_wire() -> ModuleType:
    """The gemini provider plugin's ``interactions`` module: the ``interactions`` submodule of the package the
    registered ``gemini`` profile was loaded from. Raises ``GeminiAPIError`` when there is none, or when it
    lacks what this transport calls (including ``http_error``'s ``key_guidance``)."""
    wire = gemini_plugin_module("interactions")
    missing = [name for name in _WIRE_NAMES if not hasattr(wire, name)]
    if not missing:
        try:
            inspect.signature(wire.http_error).bind_partial(key_guidance=False)
        except (TypeError, ValueError):
            missing.append("http_error(key_guidance=...)")
    if missing:
        raise GeminiAPIError(_WIRE_MISSING.format(why=f"the loaded gemini provider plugin ({wire.__name__}) is too "
                                                      f"old for it (missing {', '.join(missing)})"),
                             code="gemini_wire_unavailable")
    return wire


def gemini_plugin_module(name: str) -> ModuleType:
    """Submodule *name* of the package the registered ``gemini`` profile was loaded from. Raises
    ``GeminiAPIError`` when there is none."""
    from providers import get_provider_profile

    profile = get_provider_profile("gemini")
    if profile is None:
        raise GeminiAPIError(_WIRE_MISSING.format(why="no gemini provider plugin is loaded"),
                             code="gemini_wire_unavailable")
    module = type(profile).__module__
    package = getattr(sys.modules.get(module), "__package__", None)
    try:
        if not package:
            raise ImportError(f"{module} is not part of a package")
        return importlib.import_module(f"{package}.{name}")
    except ImportError as exc:
        raise GeminiAPIError(_WIRE_MISSING.format(why=f"the loaded gemini provider plugin ({package or module}) "
                                                      f"has no {name} module: {exc}"),
                             code="gemini_wire_unavailable") from exc


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
        # Found once, here. When it is missing, each call looks again and raises the reason: a create_client
        # error would instead send this provider down the generic OpenAI client path.
        try:
            self._wire: Optional[ModuleType] = interactions_wire()
        except GeminiAPIError:
            self._wire = None

    def _create_chat_completion(
        self, *, model: str = "gemini-3.8-flash", messages: Optional[list[dict[str, Any]]] = None, stream: bool = False,
        tools: Any = None, tool_choice: Any = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        top_p: Optional[float] = None, stop: Any = None, response_format: Any = None,
        extra_body: Optional[dict[str, Any]] = None, timeout: Any = None, **_: Any,
    ) -> Any:
        # ``temperature``/``top_p`` are never sent: Gemini 4 / 3.5 reject them. A long-decode budget in
        # ``extra_body`` is ignored: this method cannot continue a turn past one request.
        wire = self._wire = self._wire or interactions_wire()
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
        wire = self._wire
        # The gemini client's own stream helpers, from the package the wire came from (already found above).
        helpers = gemini_plugin_module("client")
        state = wire.StreamState()
        headers = {**self._headers(), "Accept": "text/event-stream"}
        try:
            with self._http.stream("POST", url, json={**request, "stream": True}, headers=headers, timeout=timeout) as response:
                if response.status_code != 200:
                    raise self._http_error(response, helpers.read_streaming_error_body(response))
                for event in wire.iter_events(helpers._iter_sse_events(response)):
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
        return (self._wire or interactions_wire()).http_error(
            response, body_text=body_text, api_key=self.api_key, base_url=self.base_url, key_guidance=False)
