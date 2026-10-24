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

from typing import Optional

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
    try:
        catalog = cached_provider_model_ids(provider)
    except Exception:
        return None
    return next((mid for mid in catalog if mid.lower() == wanted), None) or next(
        (mid for mid in catalog if "/" in mid and mid.split("/", 1)[1].lower() == wanted), None)


def is_publisher_namespace_provider(provider: str) -> bool:
    """True when *provider*'s catalog ids live in a PUBLISHER namespace (``google/…``,
    ``nvidia/…``) rather than a vendor-internal one.

    That slug form is also what every aggregator (nous / openrouter / ai-gateway / kilocode)
    publishes, so a prefix match against such a catalog would silently move the user onto the
    provider's own billing account — a GCP project for vertex. Such a provider must not be
    auto-switched TO by inference; it stays reachable by name, in its own picker, and via an
    explicit model choice for that provider.
    """
    from hermes_cli.models_catalog_static import _VENDOR_NAMESPACE_CATALOG_PROVIDERS

    return (provider or "").strip().lower() in _VENDOR_NAMESPACE_CATALOG_PROVIDERS


def static_catalog_matches(
    name: str,
    current_provider: str,
    *,
    for_inference: bool = False,
    provider_keys: frozenset[str] | set[str],
    provider_models: dict[str, list[str]],
    aggregator_providers: frozenset[str],
    borrowed_model_providers: frozenset[str],
    model_in_provider_catalog,
):
    """Yield every ``(provider_id, name)`` whose static catalog lists *name*, in ladder order.

    Several first-party providers list the same slug (``gpt-5.6-luna`` on ``openai-api`` AND
    ``openai-codex``); the first is only a guess, so callers that gate on credentials need the
    siblings too (#102775).

    ``for_inference=True`` is the AUTO-SWITCH ladder: providers whose catalog ids live in a
    publisher namespace are skipped there, because that slug form is what every aggregator
    publishes — matching it would silently re-bill the user on the publisher's own account
    (a GCP project for vertex). The conflict/coherence check reads the full catalog instead.
    """
    name_lower = name.lower()
    current_keys = provider_keys
    # Step 1: direct static-catalog match. Aggregators list other vendors' models — never
    # auto-switch TO them. A custom endpoint (custom / custom:*) is never auto-switched away
    # from: the user configured it deliberately and may serve the same model name there.
    if current_provider != "custom" and not current_provider.startswith("custom:"):
        for pid in provider_models:
            if pid in current_keys or pid in aggregator_providers or pid in borrowed_model_providers:
                continue
            if for_inference and is_publisher_namespace_provider(pid):
                continue
            if model_in_provider_catalog(name_lower, {pid}):
                yield (pid, name)

    # Borrow-list providers (re-expose other vendors' models) only after every native-vendor
    # catalog, and only when one is the current provider.
    for pid in borrowed_model_providers:
        if pid not in current_keys and model_in_provider_catalog(name_lower, {pid}):
            yield (pid, name)


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
