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
