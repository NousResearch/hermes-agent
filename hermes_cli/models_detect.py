"""Live-catalog and credential guards for ``detect_provider_for_model``.

Split out of ``hermes_cli.models``. The detection ladder there consults static catalogs, then the
OpenRouter catalog, and its answer used to be applied blindly. Two guards close the class of
"put the user on a provider they never selected":

* the CURRENT provider's live catalog outranks every static guess (Codex early-access ids, Nous
  Portal slugs, Ollama Cloud models absent from ``_PROVIDER_MODELS`` — #97487, the $100 Astra
  incident);
* an auto-detected TARGET must be a provider the user has credentials for. Guessing a vendor the
  user never signed into either 401s or, for OpenRouter (whose runtime resolves with an empty key
  instead of raising), silently bills a metered aggregator.
"""

from __future__ import annotations

from typing import Any, Optional

from hermes_cli.model_catalog import Listings, manifest_additions

# Providers where the ladder's own OpenRouter step is the answer, or where there is no catalog to
# consult; ``custom`` endpoints are never auto-switched away from (handled upstream).
_SKIP = frozenset({"", "auto", "openrouter", "custom"})


def current_provider_catalog_match(model_name: str, current_provider: str) -> Optional[str]:
    """Return the current provider's own spelling of *model_name* when its live (disk-cached)
    catalog serves it — exact id, or the bare part after ``vendor/`` — else ``None``.

    Goes through :func:`hermes_cli.models.cached_provider_model_ids` (1h TTL, stale-while-
    revalidate) so a model switch does not block on a cold ``/v1/models`` round-trip in the
    common case; a fetch failure yields an empty catalog and the ladder continues unchanged."""
    from hermes_cli.models import cached_provider_model_ids, normalize_provider

    provider = (current_provider or "").strip().lower()
    if provider in _SKIP or provider.startswith("custom:") or normalize_provider(provider) in _SKIP:
        return None
    wanted = (model_name or "").strip().lower()
    if not wanted:
        return None
    def served(catalog: list[str]) -> Optional[str]:
        return next((mid for mid in catalog if mid.lower() == wanted), None) or next(
            (mid for mid in catalog if "/" in mid and mid.split("/", 1)[1].lower() == wanted), None)

    try:
        # A row the remote manifest supplied and the provider's own listing did not report is a row
        # to offer, not evidence this provider serves the model: it is left out before the
        # exact-then-suffix match, so it cannot hide an id the provider did list. A match the
        # manifest re-cased is answered in the provider's own spelling.
        catalog = cached_provider_model_ids(provider)
        record = cached_provenance(normalize_provider(provider))
        published = record["manifest_only"] if record else set()
        match = served([mid for mid in catalog if mid.lower() not in published])
        return record["provider_spelling"].get(match.lower(), match) if match and record else match
    except Exception:
        return None


def _ids(value: Any) -> set[str]:
    return {str(mid).lower() for mid in value} if isinstance(value, list) else set()


def with_manifest_provenance(provider: str, fresh: Optional[dict], listings: Listings) -> Optional[dict]:
    """``fresh``, a provider-model cache row about to be stored, plus what the manifest put into it,
    taken from the build that produced it (lowercased ids):

    * ``manifest_only``: supplied by the manifest and not reported by the provider's own listing.
      Routing leaves these out.
    * ``needs_manifest``: in the row only because of the manifest: ``manifest_only`` plus ids a
      curated-intersection builder kept because the manifest named them. Display and onboarding
      treat these as published rows.
    * ``provider_spelling``: the provider's spelling of a reported id the manifest re-cased."""
    if not fresh:
        return fresh
    in_row = {str(mid).lower(): str(mid) for mid in fresh["models"]}
    supplied = listings.supplied & in_row.keys()
    only = supplied - listings.reported.keys()
    return {
        **fresh,
        "manifest_only": sorted(only),
        "needs_manifest": sorted(only | (supplied & listings.gated)),
        "provider_spelling": {
            mid: listings.reported[mid] for mid in sorted(supplied - only) if listings.reported[mid] != in_row[mid]},
    }


def served_rows(provider: str, entry: dict) -> list[str]:
    """A cache row's ids, minus those that were in it only because of a manifest row that is gone
    since (a row it dropped, a catalog since disabled); the shipped row if nothing else is left. An
    id the install now ships, or that the provider reported, is never one of them."""
    from hermes_cli.models import static_catalog

    gone = _ids(entry.get("needs_manifest"))
    if gone:
        shipped = static_catalog(provider)
        gone -= {mid.lower() for mid in (*shipped, *manifest_additions(provider, shipped))}
    if not gone:
        return entry["models"]
    return [mid for mid in entry["models"] if mid.lower() not in gone] or list(static_catalog(provider))


def cached_provenance(provider: str) -> Optional[dict[str, Any]]:
    """``manifest_only``, ``needs_manifest`` (sets) and ``provider_spelling`` of the provider's cache
    row; ``None`` when there is no usable row for these credentials. A row written before these
    fields existed never held a manifest row, so it reads as having none. ``now_shipped`` holds the
    ids the row needed the manifest for that this install ships: the row still lists them where the
    manifest put them, not where the shipped row does."""
    from hermes_cli.models import (
        _cache_entry_valid, _credential_fingerprint, _load_provider_models_cache, _normalized_cache_slug, static_catalog)

    slug = _normalized_cache_slug(provider)
    entry = _load_provider_models_cache().get(slug)
    if not _cache_entry_valid(entry, _credential_fingerprint(slug), allow_empty=True):
        return None
    shipped = {mid.lower() for mid in static_catalog(slug)}  # an upgrade may ship a formerly published id
    spelling = entry.get("provider_spelling")
    needs_manifest = _ids(entry.get("needs_manifest"))
    return {"manifest_only": _ids(entry.get("manifest_only")) - shipped,
            "needs_manifest": needs_manifest - shipped, "now_shipped": needs_manifest & shipped,
            "provider_spelling": spelling if isinstance(spelling, dict) else {}}


def relay_lists_live_only(provider: str, base_url: str = "") -> bool:
    """Whether ``provider`` is pointed at a relay whose live ids the picker lists verbatim
    (``models.provider_model_ids``), so no curated or manifest row joins them. ``base_url`` is the
    endpoint in use when known (setup, validation); otherwise the saved ``model.base_url`` decides."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import _RELAY_AWARE_CATALOG_FETCHERS, _configured_relay_base_url
    from hermes_cli.route_identity import normalize_route_base_url
    from providers import get_provider_profile

    if provider in _RELAY_AWARE_CATALOG_FETCHERS:
        return False
    if not base_url:
        return bool(_configured_relay_base_url(provider))
    endpoints = (getattr(get_provider_profile(provider), "base_url", ""),
                 getattr(PROVIDER_REGISTRY.get(provider), "inference_base_url", ""))
    return normalize_route_base_url(base_url) not in {normalize_route_base_url(url) for url in endpoints if url}


def current_provider_owns_vendor(model_name: str, current_provider: str) -> bool:
    """True when *model_name* belongs to the vendor a single-vendor first-party provider natively
    serves (``gpt-6-astra`` on ``openai-codex``, ``grok-4.6`` on ``xai-oauth``).

    A first-party session plus that vendor's own id is a selection, not a guess: when the live
    catalog could not confirm the id (fetch failed, static fallback lags an early-access rollout)
    the answer is "stay and let the vendor accept or reject it", never "a reseller lists it, so
    switch there". Aggregators, custom endpoints and multi-vendor resellers (nvidia, alibaba, ...)
    have no single native vendor and are skipped."""
    from hermes_cli.model_normalize import detect_vendor
    from hermes_cli.models import _AGGREGATOR_PROVIDERS, _PROVIDER_MODELS, normalize_provider

    provider = (current_provider or "").strip().lower()
    if provider in _SKIP or provider.startswith("custom:"):
        return False
    normalized = normalize_provider(provider)
    if normalized in _SKIP or normalized in _AGGREGATOR_PROVIDERS:
        return False
    vendor = detect_vendor(model_name or "")
    if not vendor:
        return False
    # An id the classifier cannot place (Bedrock ``us.anthropic.claude-…``) is evidence the
    # provider is NOT single-vendor; only a fully classified, single-vendor catalog owns the name.
    native = {detect_vendor(mid) for mid in _PROVIDER_MODELS.get(normalized, ())}
    return native == {vendor}


def provider_has_credentials(provider: str) -> bool:
    """Whether *provider* can be switched to without the user typing a key: env/.env key, auth
    store login, or a usable credential-pool entry. ``custom``/``custom:*`` targets only come out
    of the ladder when the user declared them in config, so they count as authenticated."""
    from hermes_cli.auth import get_auth_status, has_usable_secret
    from hermes_cli.config import get_env_value_prefer_dotenv

    pid = (provider or "").strip().lower()
    if not pid:
        return False
    if pid == "custom" or pid.startswith("custom:"):
        return True
    try:
        if pid == "openrouter" and has_usable_secret(get_env_value_prefer_dotenv("OPENROUTER_API_KEY")):
            return True
        status = get_auth_status(pid) or {}
        if status.get("logged_in") or status.get("configured"):
            return True
        from agent.credential_pool import load_pool

        pool = load_pool(pid)
        return bool(pool.has_credentials() and pool.has_available())
    except Exception:
        return False
