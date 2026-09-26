"""Z.AI Responses API provider profile (api.z.ai/api/v1).

Z.AI exposes its GLM models on separate, independently-billed endpoints:

  - ``https://api.z.ai/api/paas/v4``        — OpenAI Chat Completions wire,
    served by the built-in ``zai`` provider profile (left untouched here).
  - ``https://api.z.ai/api/coding/paas/v4`` — Coding Plan endpoint.
  - ``https://api.z.ai/api/v1``             — **OpenAI Responses wire**
    (``POST /v1/responses``), the endpoint this profile declares.

The Responses endpoint is what Hermes' ``codex_responses`` transport speaks
natively (same shape as the ``api.openai.com`` and ``api.x.ai`` lanes).
Until this profile existed, reaching it required a hand-written
``providers:`` entry in config.yaml with ``transport: codex_responses``;
registering it as a first-class plugin profile makes
``--provider zai-responses`` work out of the box. The same ``GLM_API_KEY``
variable the ``zai`` profile uses authenticates both endpoints;
``ZAI_API_KEY`` / ``Z_AI_API_KEY`` are accepted as aliases for parity and
``ZAI_RESPONSES_BASE_URL`` overrides the endpoint (relays/proxies).

Live-verified against api.z.ai (Aug 2026):

* ``POST /api/v1/responses`` with ``model: glm-5.3-flash`` → HTTP 200
  (streaming tool-calling turn).
* ``GET /api/v1/models`` → HTTP 200 with a ``{"models": [...]}`` envelope
  keyed by ``slug`` (plus ``supported_in_api``), NOT the OpenAI
  ``{"data": [...]}``/``id`` shape — hence the ``fetch_models`` override
  and the standalone :func:`parse_zai_responses_models` parser.
  Catalog served at validation time: ``glm-5.3``, ``glm-5.3-flash``,
  ``glm-5-turbo``.

Reverse hostname mapping deliberately stays pointed at the ``zai`` profile
(``api.z.ai`` in ``agent/model_metadata.py``): this profile only adds an
alternate wire for the same host, so no collision entry is registered.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

ZAI_RESPONSES_DEFAULT_BASE_URL = "https://api.z.ai/api/v1"


def _base_url() -> str:
    """Allow a base-URL override via ``ZAI_RESPONSES_BASE_URL`` (relays)."""
    return (
        os.getenv("ZAI_RESPONSES_BASE_URL", "").strip().rstrip("/")
        or ZAI_RESPONSES_DEFAULT_BASE_URL
    )


def parse_zai_responses_models(data: Any) -> list[str] | None:
    """Parse Z.ai's ``/api/v1/models`` payload into model ID strings.

    Z.ai serves ``{"models": [{"slug": ..., "supported_in_api": ...}]}``
    where ``slug`` is the model id the Responses endpoint accepts.
    Entries with ``supported_in_api: false`` are skipped. Returns None
    when nothing usable is present so callers fall back to the profile's
    ``fallback_models``.
    """
    if not isinstance(data, dict):
        return None
    entries = data.get("models")
    if not isinstance(entries, list):
        return None
    ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug") or "").strip()
        if not slug:
            continue
        if entry.get("supported_in_api") is False:
            continue
        ids.append(slug)
    # Dedupe, preserve listing order.
    return list(dict.fromkeys(ids)) or None


class ZaiResponsesProfile(ProviderProfile):
    """Z.AI GLM models on the native OpenAI Responses wire."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the live catalog, parsing Z.ai's ``{"models": [...]}`` shape.

        The base-class implementation sends the same auth/UA headers but
        only parses OpenAI ``{"data": [...]}/"id"`` envelopes, so it would
        return None on a healthy Z.ai response. Transport/auth handling is
        mirrored here; parsing goes through
        :func:`parse_zai_responses_models`.
        """
        caller_base = (base_url or "").strip()
        if caller_base and caller_base.rstrip("/") != (self.base_url or "").rstrip("/"):
            # User-configured base_url (proxy/relay): probe it directly.
            url = caller_base.rstrip("/") + "/models"
        else:
            url = (self.models_url or "").strip() or (
                self.base_url.rstrip("/") + "/models"
            )

        import urllib.request

        from hermes_cli.urllib_security import open_credentialed_url

        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        for k, v in self.default_headers.items():
            req.add_header(k, v)
        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None
        return parse_zai_responses_models(data)


zai_responses = ZaiResponsesProfile(
    name="zai-responses",
    aliases=("zai-v1", "glm-responses"),
    api_mode="codex_responses",
    # GLM_API_KEY is Z.ai's documented variable; ZAI_API_KEY / Z_AI_API_KEY
    # are accepted aliases for parity with the ``zai`` profile.
    # ZAI_RESPONSES_BASE_URL overrides the endpoint (relay/proxy setups).
    env_vars=(
        "GLM_API_KEY",
        "ZAI_API_KEY",
        "Z_AI_API_KEY",
        "ZAI_RESPONSES_BASE_URL",
    ),
    display_name="Z.AI (Responses API)",
    description="Z.AI / GLM on the native OpenAI Responses wire (api.z.ai/api/v1)",
    signup_url="https://z.ai/",
    base_url=_base_url(),
    auth_type="api_key",
    # Catalog verified live 2026-08-30 (GET /api/v1/models, HTTP 200):
    # exactly these three slugs are served with supported_in_api enabled.
    fallback_models=(
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5-turbo",
    ),
)

register_provider(zai_responses)
