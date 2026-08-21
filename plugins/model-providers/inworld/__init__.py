"""Inworld LLM router provider profile."""

import json
import logging
import time
import urllib.request
from urllib.parse import urljoin

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.inworld.ai/v1"

CATALOG_PATH = "/llm/v1alpha/models"

# Selects a model server-side, so it works before any catalog fetch.
ROUTER_MODEL = "auto"

# Matches _PROVIDER_MODELS_CACHE_TTL in hermes_cli/models.py, so a long-lived
# gateway picks up catalog changes without a restart.
_CACHE_TTL_SECONDS = 3600

# url -> (monotonic timestamp, model ids)
_CACHE: dict[str, tuple[float, list[str]]] = {}


def clear_cache() -> None:
    """Drop cached catalogs so the next fetch hits the network."""
    _CACHE.clear()


def _cached_models(url: str) -> list[str] | None:
    """Return the cached catalog for *url* while it is still fresh."""
    entry = _CACHE.get(url)
    if entry is None:
        return None
    fetched_at, models = entry
    if time.monotonic() - fetched_at >= _CACHE_TTL_SECONDS:
        _CACHE.pop(url, None)
        return None
    return list(models)


def parse_catalog(payload: object) -> list[str]:
    """Return tool-calling model ids from a catalog payload.

    Ids join provider and model with a slash: the qualified form is what
    distinguishes the same model served by several upstreams, and the short
    form the API echoes back is rejected as a request id.

    A model that does not advertise tool calling is dropped — Hermes drives
    every turn through tool calls, so it could not run the agent.
    """
    if not isinstance(payload, dict):
        return []
    entries = payload.get("models")
    if not isinstance(entries, list):
        return []

    ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("isSupported") is False:
            continue
        provider = entry.get("provider")
        model = entry.get("model")
        if not isinstance(provider, str) or not provider:
            continue
        if not isinstance(model, str) or not model:
            continue
        spec = entry.get("spec")
        capabilities = spec.get("capabilities") if isinstance(spec, dict) else None
        if not isinstance(capabilities, dict):
            continue
        if capabilities.get("functionCalling") is not True:
            continue
        ids.append(f"{provider}/{model}")
    return ids


class _InworldProfile(ProviderProfile):
    """Inworld router profile — the catalog needs its own path, auth, and shape."""

    def resolve_aux_model(self, *, vision: bool = False) -> str:
        """Fall back to the router when the pinned aux model leaves the catalog.

        ``default_aux_model`` is a hardcoded id, so it rots the day Inworld
        retires that model and every auxiliary call 404s. Consults only an
        already-cached catalog — this runs on client-resolution paths and has
        no credential to fetch with — and returns "" when it has no basis for
        an opinion, which sends the caller to ``default_aux_model`` as before.
        """
        try:
            catalogs = [
                models
                for models in (_cached_models(url) for url in list(_CACHE))
                if models
            ]
            if not catalogs:
                return ""
            if any(self.default_aux_model in models for models in catalogs):
                return ""
            logger.debug(
                "Inworld aux model %s is absent from the catalog; using %s",
                self.default_aux_model, ROUTER_MODEL,
            )
            return ROUTER_MODEL
        except Exception:
            return ""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the router catalog, or None to fall back to ``fallback_models``."""
        url = urljoin(base_url or self.base_url or DEFAULT_BASE_URL, CATALOG_PATH)

        cached = _cached_models(url)
        if cached is not None:
            return cached

        key = (api_key or "").strip()
        if not key:
            logger.debug("fetch_models(inworld): no credential, skipping %s", url)
            return None

        if not key.isascii() or any(c.isspace() for c in key):
            logger.warning(
                "Skipping Inworld catalog discovery: INWORLD_API_KEY contains "
                "characters that cannot appear in an Authorization header. "
                "Expected the base64 key string shown in the Inworld portal."
            )
            return None

        if not url.startswith(("http://", "https://")):
            logger.warning(
                "Skipping Inworld catalog discovery: %r has no http:// or "
                "https:// scheme. Set INWORLD_BASE_URL to e.g. %s.",
                url, DEFAULT_BASE_URL,
            )
            return None

        from hermes_cli.urllib_security import open_credentialed_url

        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Basic {key}")
        req.add_header("Accept", "application/json")
        for k, v in self.default_headers.items():
            req.add_header(k, v)

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(inworld): %s", exc)
            return None

        discovered = parse_catalog(payload)
        if not discovered:
            # Reached and understood the catalog, and it offered nothing usable
            # — an account/region problem an operator can act on, unlike the
            # transient transport failures logged at debug above.
            logger.warning(
                "Inworld catalog at %s returned no tool-calling models; "
                "only %r will be offered.", url, ROUTER_MODEL,
            )
            return None

        models = [ROUTER_MODEL, *discovered]
        _CACHE[url] = (time.monotonic(), list(models))
        return models


inworld = _InworldProfile(
    name="inworld",
    aliases=("inworld-ai", "inworld-router"),
    display_name="Inworld",
    description="Inworld — LLM router fronting first-party and upstream models",
    signup_url="https://portal.inworld.ai/",
    env_vars=("INWORLD_API_KEY", "INWORLD_BASE_URL"),
    base_url=DEFAULT_BASE_URL,
    auth_type="api_key",
    # The catalog spans upstreams with different completion caps; each applies
    # its own when we send none.
    default_max_tokens=None,
    default_aux_model="inworld/models/gemma-4-26b-a4b-it",
    # Live discovery is the source of truth for everything else — a stale id
    # here would route users to a model an upstream may have retired.
    fallback_models=(ROUTER_MODEL,),
)

register_provider(inworld)
