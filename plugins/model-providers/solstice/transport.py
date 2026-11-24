"""Solstice inference transport: the gemini plugin's Interactions wire on the per-user-quota method
(``POST <host>/v1beta/interactions:createPerUserQuota``), with the user's OAuth access token as a Bearer.

That method takes only stateless requests (``store: false``; a stored interaction is a 400), so the
client runs the wire with long decode off: one request per turn, no continuation hops, and a stream that
ends before ``interaction.completed`` is an error rather than a finished turn. Request building, thought
signature replay, stream translation and error mapping are the gemini plugin's, so a fix there reaches
this provider too.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from agent.gemini_native_adapter import GeminiAPIError
from hermes_cli.version_info import hermes_user_agent

from ..gemini import interactions as wire
from ..gemini.client import GeminiClient
from . import INFERENCE_BASE_URL

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


class SolsticeClient(GeminiClient):
    """``GeminiClient`` pinned to the stateless Interactions wire on ``interactions:createPerUserQuota``."""

    MISSING_KEY_ERROR = "Solstice needs a signed-in account. Run `hermes auth add solstice`."
    LONG_DECODE = False  # stored interactions (continuation hops) are a 400 on this method
    SAMPLING_PARAMS = False  # Gemini 4 / 3.5 reject temperature and top_p
    REQUIRE_COMPLETED_STREAM = True  # nothing is stored, so an interrupted stream can never be resumed

    def __init__(self, *, base_url: Any = None, timeout: Any = None, **kwargs: Any) -> None:
        # No base means this endpoint, never the generic Gemini default (a different API surface).
        super().__init__(base_url=base_url or INFERENCE_BASE_URL, timeout=timeout, **kwargs)
        # Per request, not only on the httpx client: an injected keepalive client brings its own default.
        self._request_timeout = timeout or DEFAULT_TIMEOUT

    def _uses_interactions(self, model: str) -> bool:
        return True  # the only wire this method speaks, whatever the model or ``providers.gemini`` config

    def _interactions_url(self) -> str:
        return interactions_url(self.base_url)

    def _create_chat_completion(self, *, timeout: Any = None, **kwargs: Any) -> Any:
        # An explicit per-call timeout (the agent's configured / stale-matched budget) wins; else 3600 s read.
        return super()._create_chat_completion(timeout=timeout or self._request_timeout, **kwargs)

    def _generate(self, url: str, request: dict[str, Any], timeout: Any) -> dict[str, Any]:
        # The unary call: the method takes both modes, so the flag is always named.
        return super()._generate(url, {**request, "stream": False}, timeout)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _headers(self) -> dict[str, str]:
        # Bearer, content type and User-Agent only: no API key, user-project or DPoP header, nor Google's client tag.
        extra = {k: v for k, v in self._default_headers.items() if k.lower() not in _DROPPED_DEFAULT_HEADERS}
        return {**extra, "Content-Type": "application/json", "Accept": "application/json", **self._auth_headers(),
                "User-Agent": hermes_user_agent("hermes-agent")}

    def _http_error(self, response: Any, body_text: Any = None) -> GeminiAPIError:
        # Every request is an Interactions one (the base mapper matches only a ``/interactions`` path). The
        # API-key remedies (free tier, Standard key, wrong key surface) do not apply to a bearer.
        return wire.http_error(response, body_text=body_text, api_key=self.api_key, base_url=self.base_url,
                               key_guidance=False)
