"""OpenRouter ``@preset/<slug>`` head-model resolution for the cache policy (#157875).

A preset string hides its head model, so family matchers (Claude/Kimi/Qwen) never
fire on it. Resolved here once per slug, TTL-cached, strictly fail-open: any failure
keeps the caller's original model string, so offline installs behave exactly as
they did before preset awareness existed.
"""

import logging
import os
import time
from typing import Optional, Tuple
from urllib.parse import quote

from utils import base_url_host_matches

logger = logging.getLogger(__name__)

_PRESET_HEAD_MODEL_TTL_SECONDS = 6 * 3600
_PRESET_HEAD_MODEL_FAILURE_TTL_SECONDS = 10 * 60
_PRESET_MARKER = "@preset/"
_preset_head_model_cache: dict[str, Tuple[Optional[str], float]] = {}


def resolve_preset_head_model(slug: str, api_key: str = "") -> Optional[str]:
    """Head model of an OpenRouter preset (``data.designated_version.config.model``).

    TTL-cached per slug; keyed lookups that fail are negative-cached briefly so an offline
    install does not retry the fetch every turn. A **keyless** failure is never negative-cached:
    without a key the endpoint always answers 401, which says nothing about the preset
    (@grievertime, #158147 backport report). Returns ``None`` on any failure — callers must keep
    their original model string in that case (fail-open). Presets are chains, so a non-Claude head
    simply means the caller makes no family decision; the wire model stays the preset string
    either way.
    """
    import httpx

    from agent import model_metadata_http

    slug = (slug or "").strip()
    if not slug:
        return None
    cached = _preset_head_model_cache.get(slug)
    if cached is not None:
        model, at = cached
        ttl = _PRESET_HEAD_MODEL_TTL_SECONDS if model else _PRESET_HEAD_MODEL_FAILURE_TTL_SECONDS
        if time.monotonic() - at < ttl:
            return model
    if not api_key:
        # Always a 401: skip the fetch and the cache entirely, so a later call with the
        # key can still resolve (@grievertime, #158147 backport).
        return None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    model: Optional[str] = None
    try:
        verify = model_metadata_http.resolve_verify("https://openrouter.ai")
        with httpx.Client(timeout=2.0, headers=headers, verify=verify) as client:
            resp = client.get(f"https://openrouter.ai/api/v1/presets/{quote(slug, safe='')}")
            if resp.status_code != 200:
                raise RuntimeError(f"presets API answered HTTP {resp.status_code}")
            data = (resp.json() or {}).get("data") or {}
            config = (data.get("designated_version") or {}).get("config") or {}
            head = config.get("model")
            if isinstance(head, str) and head:
                model = head
    except Exception:
        logger.debug("preset head-model lookup failed for %s", slug, exc_info=True)
        model = None
    if model is not None or api_key:
        _preset_head_model_cache[slug] = (model, time.monotonic())
    return model


def _preset_lookup_key(agent, base_url: str = "") -> str:
    """Auth key for the OpenRouter presets GET (#157875 follow-up, credit @grievertime).

    The lookup always targets openrouter.ai, so ``agent.api_key`` may ride along only when the
    route is openrouter.ai — a key minted for another provider must never reach OpenRouter.
    ``base_url`` is the caller's destination route when it knows one (the policy's ``eff_base_url``);
    at agent-init time ``agent.api_key``/``agent.base_url`` are not bound yet
    (``_init_prompt_cache_config`` runs before ``_build_client``), so the fallback is the scoped
    ``OPENROUTER_API_KEY``, the designated OpenRouter credential.
    """
    route = (base_url or getattr(agent, "base_url", "") or "").strip()
    key = (getattr(agent, "api_key", "") or "").strip()
    if key and base_url_host_matches(route, "openrouter.ai"):
        return key
    try:
        from agent import auxiliary_client

        return getattr(auxiliary_client, "_scoped_key_env")("OPENROUTER_API_KEY")
    except Exception:
        # Deliberate fail-open: preset resolution must never break on key-scope plumbing.
        logger.debug("scoped OpenRouter key unavailable; falling back to ambient env", exc_info=True)
        return (os.getenv("OPENROUTER_API_KEY") or "").strip()


def effective_family_model(eff_model: str, agent, base_url: str = "") -> Tuple[str, str]:
    """``(resolved_model, model_lower)`` the cache policy should family-match on.

    ``@preset/<slug>`` resolves its head (fail-open to the preset string); the suffixed
    ``<model>@preset/<slug>`` form already names its base and needs no fetch. ``base_url`` is the
    caller's destination route (the policy's ``eff_base_url``) so the lookup key rule judges the
    route the request will actually take, not a stale bound attribute. The wire model is never
    changed — this feeds family matching only.
    """
    if _PRESET_MARKER not in eff_model.lower():
        return eff_model, eff_model.lower()
    if eff_model.lower().startswith(_PRESET_MARKER):
        slug = eff_model[len(_PRESET_MARKER):]
        head = resolve_preset_head_model(slug, _preset_lookup_key(agent, base_url))
        resolved = head or eff_model
    else:
        resolved = eff_model.split(_PRESET_MARKER, 1)[0]
    return resolved, resolved.lower()
