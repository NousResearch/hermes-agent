"""Account-scoped speed metadata from the existing authenticated Codex catalogue.

Unknown discovery preserves the requested tier. Only an explicit advertised tier
list permits a downgrade; response echoes never drive selection or retries.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)
# Token fingerprints include the origin; token refresh/account changes cannot reuse
# an old entitlement. No raw credentials or identities are retained in this cache.
_catalogs: dict[str, tuple[dict[str, frozenset[str]], float]] = {}
_TTL = 3600
_MAX_CATALOGS = 128


def _advertised_tiers(entry: dict[str, Any]) -> frozenset[str] | None:
    tiers = entry.get("service_tiers")
    if isinstance(tiers, list):
        if any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in tiers):
            return None
        if tiers:
            return frozenset(item["id"] for item in tiers)
    speeds = entry.get("additional_speed_tiers")
    if isinstance(speeds, list) and all(isinstance(item, str) for item in speeds):
        return frozenset({"fast": "priority"}.get(item, item) for item in speeds)
    return frozenset() if tiers == [] else None


def remember_catalog(cache_key: str, entries: list[Any]) -> None:
    """Retain only speed metadata, even when a model omits its context window."""
    models = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("slug"), str):
            continue
        tiers = _advertised_tiers(entry)
        if tiers is not None:
            models[entry["slug"].strip()] = tiers
    if cache_key not in _catalogs and len(_catalogs) >= _MAX_CATALOGS:
        del _catalogs[min(_catalogs, key=lambda key: _catalogs[key][1])]
    _catalogs[cache_key] = (models, time.time())


def _tiers_for_model(access_token: str, base_url: str, model: str) -> frozenset[str] | None:
    from agent import model_metadata as metadata
    key = metadata._codex_oauth_token_fingerprint(access_token, base_url)
    cached = _catalogs.get(key)
    if cached is None or time.time() - cached[1] >= _TTL:
        # Uses the established route/auth guards, headers, positive and negative
        # TTLs. A failed refresh must never consume expired entitlement metadata.
        metadata._fetch_codex_oauth_context_lengths_with_source(access_token, base_url)
        cached = _catalogs.get(key)
    if cached is None or time.time() - cached[1] >= _TTL:
        return None
    wire_model = metadata.strip_codex_context_variant_suffix(model)
    return cached[0].get(wire_model)


def codex_account_offers_tier(
    model: Any, tier: str, *, base_url: Any = None, access_token: Any = None,
) -> bool | None:
    """Whether the signed-in ChatGPT account's Codex catalog lists ``tier`` for ``model``.

    ``None`` when unknown (no usable token, catalog unreachable, model absent) so callers never refuse
    on a failed lookup. ``access_token`` defaults to the profile's stored Codex OAuth token."""
    from agent.codex_headers import CODEX_AUX_BASE_URL, is_official_codex_base_url
    base = str(base_url or "").strip() or CODEX_AUX_BASE_URL
    if not is_official_codex_base_url(base):
        return None
    from hermes_cli.auth_constants import AuthError, _decode_jwt_claims
    # Only a ChatGPT OAuth JWT reads this catalog; a caller's pre-init key (an OpenAI API key on a
    # profile that also has one) is not this account's credential, so use the stored Codex login.
    token = access_token if isinstance(access_token, str) and _decode_jwt_claims(access_token) else None
    if not token:
        try:
            from hermes_cli.auth_codex import resolve_codex_runtime_credentials
            token = resolve_codex_runtime_credentials(read_only=True).get("api_key")
        except AuthError:  # no Codex login on this profile: unknown, never a refusal
            return None
    if not isinstance(token, str) or not token:
        return None
    from agent.model_metadata import _bare_codex_slug
    supported = _tiers_for_model(token, base, _bare_codex_slug(str(model or "")))
    return None if supported is None else tier in supported


def ultrafast_offered(model: Any, *, provider: Any = None, base_url: Any = None, access_token: Any = None) -> bool:
    """Whether ``/fast ultrafast`` (and the Ultrafast picker row) is offered on this route: the route
    gate plus, on the ChatGPT/Codex subscription, the account's own Codex catalog. That backend accepts
    ``service_tier: "ultrafast"`` from a plan without the entitlement and silently serves the default
    tier, so the catalog the official client reads is the authority. An unreadable catalog never
    blocks (request-time negotiation still clamps per credential)."""
    from urllib.parse import urlparse

    from hermes_cli.models import normalize_provider, resolve_fast_mode_overrides
    if resolve_fast_mode_overrides(model, provider=provider, base_url=base_url, tier="ultrafast") is None:
        return False
    on_codex_sub = (provider and normalize_provider(provider) == "openai-codex") or (
        (urlparse(str(base_url or "")).hostname or "").lower() == "chatgpt.com")
    return not on_codex_sub or codex_account_offers_tier(
        model, "ultrafast", base_url=base_url, access_token=access_token) is not False


def ultrafast_offered_for(cli: Any, model: Any) -> bool:
    """:func:`ultrafast_offered` on a CLI shell's resolved route (attributes absent before setup)."""
    return ultrafast_offered(model, provider=getattr(cli, "provider", None), base_url=getattr(cli, "base_url", None),
                             access_token=getattr(cli, "api_key", None))


def negotiate_request_tier(agent: Any, overrides: dict[str, Any]) -> None:
    """Resolve the current credential on each build without mutating user preference."""
    from agent.codex_headers import is_official_codex_base_url
    requested = overrides.get("service_tier")
    base_url = str(getattr(agent, "base_url", "") or "")
    if (requested not in {"ultrafast", "priority"}
            or getattr(agent, "api_mode", None) != "codex_responses"
            or getattr(agent, "provider", None) != "openai-codex"
            or not is_official_codex_base_url(base_url)):
        return
    token = getattr(agent, "api_key", None)
    if not isinstance(token, str) or not token:
        return
    model = overrides.get("model", getattr(agent, "model", ""))
    supported = _tiers_for_model(token, base_url, model)
    if supported is None:
        return
    candidates = ("ultrafast", "priority") if requested == "ultrafast" else ("priority",)
    effective = next((tier for tier in candidates if tier in supported), "default")
    if effective == "default":
        overrides.pop("service_tier", None)  # Codex represents Standard by omission.
    else:
        overrides["service_tier"] = effective
    if effective != requested:
        logger.debug("Codex service tier %s -> %s: not advertised for selected credential", requested, effective)
