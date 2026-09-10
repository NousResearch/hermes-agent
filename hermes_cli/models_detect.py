"""Live-catalog guard for ``detect_provider_for_model``.

Split out of ``hermes_cli.models``. The detection ladder there consults static catalogs, then the
OpenRouter catalog. Providers whose static list lags their live catalog (Codex accounts with
early-access models, Nous Portal, Ollama Cloud) have no static entry to stop the ladder, so a bare
name the CURRENT provider already serves fell through to OpenRouter and the session was silently
rebuilt on a metered aggregator (#97487, WolframRvnwlf's $100 Astra incident).
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
