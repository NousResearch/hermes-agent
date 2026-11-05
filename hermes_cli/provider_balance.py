"""Provider balance / quota snapshots for the dashboard's Balance card.

Probe registry: ``ProviderProfile.balance_probe`` — a provider plugin declares the one
documented balance endpoint it owns, so a provider with none reports ``unsupported``
instead of guessing a URL (a path that happens to answer 404/HTML would read as a
working balance of zero).

``configured`` comes from the SAME source the Keys page groups by (``GET /api/env``'s
``_get_env_vars_sync`` rows + ``provider_catalog``), plus the auth status for providers
whose credential lives in ``auth.json`` / the credential pool — so the Keys page and the
Balance card can never disagree about whether a provider is set up.

Credentials are resolved inside the profile scope; the network probes then run outside it
so a 10 s upstream stall never holds the config lock. Results are cached in memory for
60 s; a failure degrades to ``status="error"`` carrying the exception's first line
(truncated, key-redacted), never a raised route.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from datetime import datetime, timezone, UTC
from typing import Any, Callable, Optional

from hermes_cli.web_deps import late

logger = logging.getLogger("hermes_cli.web_server")

# Resolved late so a test monkeypatch on the owning module wins at call time.
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")

CACHE_TTL_SECONDS = 60.0
PROBE_TIMEOUT_SECONDS = 10.0
# Exception text is user-facing: first line only, hard-capped, never the key.
_ERROR_MAX_CHARS = 200

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}

# Shipped profiles' probes, resolved once per process (see probe_registry).
_BUNDLED_PROBES: Optional[dict[str, Callable[[str], Any]]] = None


def _bundled_probes() -> dict[str, Callable[[str], Any]]:
    global _BUNDLED_PROBES
    if _BUNDLED_PROBES is None:
        from providers import bundled_profiles

        probes: dict[str, Callable[[str], Any]] = {}
        for profile in bundled_profiles():
            probe = getattr(profile, "balance_probe", None)
            if callable(probe):
                probes[profile.name] = probe
        _BUNDLED_PROBES = probes
    return _BUNDLED_PROBES


def probe_registry() -> dict[str, Callable[[str], Any]]:
    """``provider id -> balance_probe`` for every registered profile that declares one.

    A ``$HERMES_HOME`` copy of a bundled provider shadows it by name and may predate
    ``balance_probe``; falling back to the shipped profile's probe keeps that provider
    reporting its balance instead of silently downgrading it to ``unsupported``.
    """
    from providers import list_providers

    registry: dict[str, Callable[[str], Any]] = {}
    for profile in list_providers():
        probe = getattr(profile, "balance_probe", None)
        if callable(probe):
            registry[profile.name] = probe
    for slug, probe in _bundled_probes().items():
        registry.setdefault(slug, probe)
    return registry


def _error_text(exc: BaseException, api_key: str = "") -> str:
    """First line of ``exc`` — truncated, single-line, and never carrying the key."""
    first = (str(exc).strip().splitlines() or [""])[0]
    text = f"{type(exc).__name__}: {first}".strip() if first else type(exc).__name__
    if api_key and api_key in text:
        text = text.replace(api_key, "***")
    return text[:_ERROR_MAX_CHARS]


def _resolve_key(slug: str) -> str:
    """The provider's API key through Hermes' existing resolvers — never echoed back."""
    from hermes_cli.auth import resolve_api_key_provider_credentials

    try:
        return str(resolve_api_key_provider_credentials(slug).get("api_key") or "")
    except Exception:
        pass
    # A bundled/plugin profile whose auth-registry mirror has not synced yet: its own
    # declared env vars are the credential source of record for that provider.
    from hermes_cli.config import get_env_value_prefer_dotenv
    from providers import get_provider_profile

    profile = get_provider_profile(slug)
    for var in ((getattr(profile, "env_vars", ()) or ()) if profile else ()):
        value = get_env_value_prefer_dotenv(var) or ""
        if value:
            return value
    return ""


def _env_is_set(env_rows: dict, env_vars: tuple) -> bool:
    """The Keys page's ``hasAnySet`` for one provider's key env vars."""
    return any(bool((env_rows.get(var) or {}).get("is_set")) for var in env_vars)


def _auth_configured(slug: str) -> bool:
    """OAuth/pool providers keep their credential in ``auth.json``, not ``.env``."""
    from hermes_cli.auth import get_auth_status

    try:
        status = get_auth_status(slug) or {}
    except Exception:
        return False
    return bool(status.get("logged_in") or status.get("configured"))


def _probe_row(
    slug: str, label: str, configured: bool, api_key: str, probe: Optional[Callable[[str], Any]]
) -> dict:
    """One provider's balance row — numbers and quota, or a short reason there are none."""
    row: dict[str, Any] = {
        "id": slug,
        "label": label,
        "configured": bool(configured),
        # Probe exists for this provider — the UI shows its row even when no key is set.
        "supported": probe is not None,
        "status": "no_key",
        "error": None,
        "primary": None,
        "breakdown": {},
        "quota": {},
        "extra": {},
    }
    if not configured:
        return row
    if probe is None:
        row["status"] = "unsupported"
        return row
    if not api_key:
        return row

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="balance-probe")
    try:
        result = pool.submit(probe, api_key).result(timeout=PROBE_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        row["status"] = "error"
        row["error"] = f"balance probe timed out after {PROBE_TIMEOUT_SECONDS:g}s"
        return row
    except Exception as exc:
        row["status"] = "error"
        row["error"] = _error_text(exc, api_key)
        return row
    finally:
        # wait=False: an abandoned probe is bounded by its own timeout and must not hold
        # this response open (same policy as /api/model/info's context-length probe).
        pool.shutdown(wait=False)

    if not isinstance(result, dict):
        row["status"] = "error"
        row["error"] = "balance probe returned an unexpected payload"
        return row
    row.update(
        status="ok",
        primary=result.get("primary") or None,
        breakdown=result.get("breakdown") or {},
        quota=result.get("quota") or {},
        extra=result.get("extra") or {},
    )
    return row


def _collect(profile: Optional[str]) -> list[dict]:
    """Rows for every provider, in the catalog's own order.

    Catalog, credentials and keys resolve inside the profile scope (disk reads, the same
    ``.env`` the Keys page reads); only the HTTP probes below run outside it.
    """
    from hermes_cli.provider_catalog import provider_catalog
    from hermes_cli.web_routers.config_env import _get_env_vars_sync

    with _profile_scope(profile):
        probes = probe_registry()
        env_rows = _get_env_vars_sync()
        catalog = provider_catalog()
        specs = []
        for descriptor in catalog:
            slug = descriptor.slug
            configured = _env_is_set(env_rows, descriptor.api_key_env_vars) or _auth_configured(slug)
            probe = probes.get(slug)
            specs.append(
                {
                    "slug": slug,
                    "label": descriptor.label or slug,
                    "configured": configured,
                    "api_key": _resolve_key(slug) if (probe and configured) else "",
                    "probe": probe,
                }
            )
    return [_probe_row(**spec) for spec in specs]


def fetch_balances(profile: Optional[str] = None, refresh: bool = False) -> dict:
    """``{"fetched_at": ..., "providers": [...]}`` — cached 60 s, ``refresh=True`` busts it."""
    key = profile or ""
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and not refresh and time.monotonic() - cached["at"] < CACHE_TTL_SECONDS:
            return cached["payload"]

    payload = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "providers": _collect(profile),
    }
    with _CACHE_LOCK:
        _CACHE[key] = {"at": time.monotonic(), "payload": payload}
    return payload
