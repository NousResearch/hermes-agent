"""Web Search Provider Registry.

Populated by plugins via :meth:`PluginContext.register_web_search_provider`;
consumed by the ``web_search`` / ``web_extract`` wrappers in :mod:`tools.web_tools`.

Active selection, in precedence order (the ``supports_search`` /
``supports_extract`` capability filter applies at every step, so a search-only
provider configured as ``web.extract_backend`` falls through):

1. ``web.search_backend`` / ``web.extract_backend``, then ``web.backend``.
2. The single capability-eligible provider that is registered AND available.
3. Legacy preference walk (``_LEGACY_PREFERENCE``) filtered by availability —
   the historic ``tools.web_tools._get_backend()`` order, so installs that never
   set a config key keep landing on the same provider.
4. Keyless free-tier walk (``_KEYLESS_PREFERENCE``), last resort.
5. ``None`` — the tool points the user at ``hermes tools``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from agent.provider_registry import ProviderRegistry, is_available_safe
from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_KEYLESS_POLICY_WARNING_COOLDOWN_SECONDS = 300.0
_keyless_policy_warning_at: dict[str, float] = {}
_keyless_policy_warning_lock = threading.Lock()


_registry: ProviderRegistry[WebSearchProvider] = ProviderRegistry(
    label="Web", provider_cls=WebSearchProvider, logger=logger,
)
_registry.export(globals())


def _read_config_key(*path: str) -> Optional[str]:
    """Resolve a dotted config key from ``config.yaml``. Returns None on miss."""
    try:
        from hermes_cli.config import load_config_readonly

        cur = load_config_readonly()
        for segment in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(segment)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
    except Exception as exc:
        logger.debug("Could not read config %s: %s", ".".join(path), exc)
    return None


def _configured_backend(capability: str) -> Optional[str]:
    """``web.<capability>_backend`` (preferred) or ``web.backend`` (shared fallback)."""
    return _read_config_key("web", f"{capability}_backend") or _read_config_key("web", "backend")


# Paid providers first so existing paid setups don't get downgraded to a free
# tier on upgrade; filtered by ``is_available()`` at walk time.
_LEGACY_PREFERENCE = ("firecrawl", "parallel", "tavily", "perplexity", "exa", "searxng", "brave-free", "ddgs")

# Anonymous public free tiers (see plugins/web/keyless_mcp.py); strictly last
# resort, i.e. zero web credentials and no importable ddgs. Unpinned keyless
# traffic round-robins across the ring per request; an explicit `hermes tools`
# pick bypasses this walk. Disable with ``web.keyless_fallback: false``.
_KEYLESS_PREFERENCE = ("exa", "parallel", "firecrawl", "keenable")


def _keyless_preference() -> tuple:
    """Keyless walk order, starting at the ring cursor in
    :mod:`plugins.web.keyless_mcp` so resolution and dispatch agree on which
    vendor a fresh install starts at; the rest follow in ring order."""
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING, _ring_cursor

        start = _ring_cursor % len(_KEYLESS_RING)
        return tuple(_KEYLESS_RING[start:] + _KEYLESS_RING[:start])
    except Exception as exc:  # noqa: BLE001 — ring optional in stripped envs
        logger.debug("keyless ring order unavailable: %s", exc)
    return _KEYLESS_PREFERENCE


def _resolve(configured: Optional[str], *, capability: str) -> Optional[WebSearchProvider]:
    """Resolve the active provider for a capability ("search" | "extract").

    Rules, in order (see module docstring): explicit config wins even when
    ``is_available()`` is False (the dispatcher surfaces a precise
    "X_API_KEY is not set" error instead of a silent switch); then the single
    available capable provider; then the availability-filtered legacy walk;
    then the keyless free-tier walk; else None.
    """
    snapshot = _registry.merged()

    def _capable(p: WebSearchProvider) -> bool:
        return bool(
            p.supports_search() if capability == "search"
            else p.supports_extract() if capability == "extract" else False
        )

    def _available(p: WebSearchProvider) -> bool:
        return is_available_safe(p, logger, "provider %s.is_available() raised %s")

    if configured:
        provider = snapshot.get(configured)
        if provider is not None and _capable(provider):
            return provider
        if provider is None:
            logger.debug("web backend '%s' configured but not registered; falling back", configured)
        else:
            logger.debug(
                "web backend '%s' configured but does not support '%s'; falling back", configured, capability
            )

    # Fallbacks are availability-filtered so a registered-but-keyless provider
    # never becomes "active" on a fresh install.
    eligible = [p for p in snapshot.values() if _capable(p) and _available(p)]
    if len(eligible) == 1:
        return eligible[0]

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if provider is not None and provider in eligible:
            return provider

    # Keyless free tier (anonymous public MCP tiers) is last-resort only: it is
    # reachable solely when the legacy walk found nothing, never pre-empting a
    # keyed setup. Disabled via ``web.keyless_fallback: false``.
    if _keyless_tier_enabled():
        for name in _keyless_preference():
            provider = snapshot.get(name)
            if provider is None or not _capable(provider):
                continue
            try:
                if provider.is_keyless_available():
                    return provider
            except Exception as exc:  # noqa: BLE001 — buggy provider skipped
                logger.debug("provider %s.is_keyless_available() raised %s", name, exc)

    return None


def _warn_keyless_policy(reason: str, message: str, *args: object) -> None:
    """Warn once per cooldown when malformed policy disables keyless web."""
    now = time.monotonic()
    with _keyless_policy_warning_lock:
        last_warning = _keyless_policy_warning_at.get(reason)
        if (
            last_warning is not None
            and now - last_warning < _KEYLESS_POLICY_WARNING_COOLDOWN_SECONDS
        ):
            return
        _keyless_policy_warning_at[reason] = now
    logger.warning(message, *args)


def _keyless_tier_enabled() -> bool:
    """Read the authoritative anonymous-egress policy from config.yaml.

    A valid config with no explicit setting retains the fresh-install default
    (enabled).  Invalid/unavailable policy state fails closed: inability to
    establish the user's opt-out state is not permission to send queries or
    URLs to anonymous third-party endpoints.
    """
    try:
        from hermes_cli.config import load_config

        config = load_config()
        if not isinstance(config, dict):
            _warn_keyless_policy(
                "config-root",
                "Keyless web fallback disabled: config root must be a mapping, got %s",
                type(config).__name__,
            )
            return False
        web_cfg = config.get("web")
        if web_cfg is None:
            return True
        if not isinstance(web_cfg, dict):
            _warn_keyless_policy(
                "web-section",
                "Keyless web fallback disabled: web config must be a mapping, got %s",
                type(web_cfg).__name__,
            )
            return False
        value = web_cfg.get("keyless_fallback", True)
        if isinstance(value, bool):
            return value
        _warn_keyless_policy(
            "keyless-value",
            "Keyless web fallback disabled: web.keyless_fallback must be boolean, got %s",
            type(value).__name__,
        )
        return False
    except Exception as exc:  # noqa: BLE001 — config layer optional
        _warn_keyless_policy(
            "config-read",
            "Keyless web fallback disabled: could not read configuration (%s)",
            type(exc).__name__,
        )
        return False


def _disabled_web_plugin_for(configured: Optional[str] = None, *, capability: Optional[str] = None) -> Optional[str]:
    """Plugin key of a *disabled* bundled web plugin that would have provided
    the configured backend (``web.<capability>_backend`` → ``web.backend``), or None.

    Lets the dispatcher say "re-enable web-firecrawl" instead of "No web extract
    provider configured". Resolved from config.yaml rather than the resolved
    backend because a disabled provider fails the availability gate and silently
    drops to the default. Bundled web plugins live under ``web/<vendor>`` with
    the provider name differing only by hyphen/underscore, so both are normalized.

    When a user sets ``web.extract_backend: firecrawl`` (or the search equivalent) but also lists
    ``web-firecrawl`` in ``plugins.disabled``, the provider never registers and the dispatcher would
    otherwise emit a misleading "No web extract provider configured. Set web.extract_backend to ..." error —
    even though the backend IS configured correctly. This helper detects that case so the dispatcher can
    point the user at the actual cause (issue #40190 follow-up: pi314's disabled-plugin symptom).
    """
    def _norm(s: str) -> str:
        return s.strip().lower().replace("-", "_")

    if not configured and capability in ("search", "extract"):
        configured = _configured_backend(capability)
    if not configured:
        return None

    want = _norm(configured)
    try:
        from hermes_cli.plugins import get_plugin_manager

        pm = get_plugin_manager()
        for key, loaded in pm._plugins.items():
            if (
                isinstance(key, str) and key.startswith("web/") and not loaded.enabled
                and loaded.error == "disabled via config" and _norm(key.split("/", 1)[1]) == want
            ):
                return key
    except Exception as exc:  # noqa: BLE001 — diagnostics are best-effort
        logger.debug("disabled-web-plugin lookup failed: %s", exc)
    return None


def get_active_search_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web search provider."""
    return _resolve(_configured_backend("search"), capability="search")


def get_active_extract_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web extract provider."""
    return _resolve(_configured_backend("extract"), capability="extract")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
from typing import List  # noqa: F401,E402
import threading  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'hermes_home_key': ('hermes_constants', 'hermes_home_key'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
