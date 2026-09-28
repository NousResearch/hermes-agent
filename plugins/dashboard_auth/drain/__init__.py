"""DrainSecretProvider — shared-bearer-secret auth for the drain-control endpoint.

Non-interactive token capability of the ``DashboardAuthProvider`` ABC (``verify_token`` +
the ``token_auth`` middleware seam): ``nous-account-service`` provisions a per-agent unique
secret (``HERMES_DASHBOARD_DRAIN_SECRET``, env-only — it is a credential); an inbound bearer
is compared constant-time and vouched for as the ``drain-control`` principal. Fail-CLOSED
entropy gate at registration (length, distinct chars, Shannon bits); interactive ABC methods
raise. Knobs ``scope`` / ``min_secret_chars`` live under ``dashboard.drain_auth``.
"""
from __future__ import annotations

import logging

from plugins.dashboard_auth._shared import (
    SharedSecretProvider, load_config_section, register_provider, shared_secret_settings)

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-drain"

# Kept here (not imported from web_server) to avoid a heavy import at plugin load.
DRAIN_ROUTE_PATH = "/api/gateway/drain"

LAST_SKIP_REASON: str = ""


class DrainSecretProvider(SharedSecretProvider):
    """Non-interactive shared-bearer-secret provider for drain control."""

    name = "drain-secret"
    display_name = "Drain Control (service credential)"
    _principal = "drain-control"
    _default_scope = "drain"
    _NOT_INTERACTIVE = "DrainSecretProvider is a non-interactive service credential."
    _NO_START_LOGIN = "DrainSecretProvider is a non-interactive service credential; there is no login flow."


# ---- Plugin entry point ----

def _load_config_drain_auth_section() -> dict:
    return load_config_section(logger, _TAG, "dashboard", "drain_auth")


def _settings() -> dict:
    """Resolve DrainSecretProvider kwargs from env/config; raises ``SkipRegistration``."""
    return shared_secret_settings(
        lambda: _load_config_drain_auth_section(), env="HERMES_DASHBOARD_DRAIN_SECRET",
        default_scope="drain", purpose="NAS-driven drain coordination")


def register(ctx) -> None:
    """Register ``DrainSecretProvider`` when a strong secret is set; no-op (records a skip
    reason) when ``HERMES_DASHBOARD_DRAIN_SECRET`` is unset or fails the entropy gate. On
    success also registers the drain route as token-authable via the generic seam."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(ctx, logger, _TAG, DrainSecretProvider, _settings)
    if kwargs is None:
        return
    # Opt the drain endpoint into the token-auth seam so the interactive cookie gate
    # doesn't bounce NAS's bearer call. The route demands the scope the provider stamps
    # on its principal, so another stacked service credential (e.g. the kanban API
    # secret) cannot drive drain control.
    try:
        from hermes_cli.dashboard_auth.token_auth import register_token_route

        register_token_route(DRAIN_ROUTE_PATH, scope=kwargs["scope"])
    except Exception as exc:  # noqa: BLE001 — seam import must not crash plugin load
        logger.warning("dashboard-auth-drain: could not register token route %s: %s", DRAIN_ROUTE_PATH, exc)
    logger.info(
        "dashboard-auth-drain: registered drain service-credential provider (scope=%s, route=%s)",
        kwargs["scope"], DRAIN_ROUTE_PATH)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'LoginStart': ('hermes_cli.dashboard_auth', 'LoginStart'),
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
