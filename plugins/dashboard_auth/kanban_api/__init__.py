"""KanbanApiSecretProvider — shared-bearer-secret auth for the external kanban REST API.

Second consumer of the non-interactive token capability, after the drain plugin: the operator
provisions ``HERMES_KANBAN_API_SECRET`` (env-only — it is a credential); an inbound bearer is
compared constant-time and vouched for as the ``kanban-api`` principal scoped to ``kanban``.
``register()`` opts the sanitized adapter (``hermes_cli.kanban_api``, mounted at
``/api/plugins/kanban/v1``) into the token seam with a prefix registration, since its paths are
parameterised; the operator dashboard routes beside it stay on the cookie/session gate. The
registration demands the ``kanban`` scope, so another stacked service credential (e.g. the drain
secret) cannot open this surface, and vice versa.

Once the secret is set the ``/v1`` surface is token-only on every bind, loopback included — the
seam owns a registered route (fail-closed). Leave the env var unset to keep it on dashboard-
session auth. Knobs ``scope`` / ``min_secret_chars`` live under ``dashboard.kanban_api_auth``.
"""
from __future__ import annotations

import logging

from plugins.dashboard_auth._shared import (
    SharedSecretProvider, load_config_section, register_provider, shared_secret_settings)

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-kanban-api"

# Kept here (not imported from the kanban plugin) to avoid a heavy import at plugin load.
KANBAN_API_PREFIX = "/api/plugins/kanban/v1/"

LAST_SKIP_REASON: str = ""


class KanbanApiSecretProvider(SharedSecretProvider):
    """Non-interactive shared-bearer-secret provider for the kanban REST API."""

    name = "kanban-api-secret"
    display_name = "Kanban REST API (service credential)"
    _principal = "kanban-api"
    _default_scope = "kanban"
    _NOT_INTERACTIVE = "KanbanApiSecretProvider is a non-interactive service credential."
    _NO_START_LOGIN = "KanbanApiSecretProvider is a non-interactive service credential; there is no login flow."


# ---- Plugin entry point ----

def _load_config_kanban_api_auth_section() -> dict:
    return load_config_section(logger, _TAG, "dashboard", "kanban_api_auth")


def _settings() -> dict:
    """Resolve KanbanApiSecretProvider kwargs from env/config; raises ``SkipRegistration``."""
    return shared_secret_settings(
        lambda: _load_config_kanban_api_auth_section(), env="HERMES_KANBAN_API_SECRET",
        default_scope="kanban", purpose="bearer access to the kanban REST API")


def register(ctx) -> None:
    """Register ``KanbanApiSecretProvider`` when a usable secret is set; no-op (records a skip
    reason) when ``HERMES_KANBAN_API_SECRET`` is unset or fails ``assess_secret_strength``. On success
    also registers the ``/v1`` prefix as token-authable via the generic seam."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(ctx, logger, _TAG, KanbanApiSecretProvider, _settings)
    if kwargs is None:
        return
    try:
        from hermes_cli.dashboard_auth.token_auth import register_token_route_prefix

        register_token_route_prefix(
            KANBAN_API_PREFIX, scope=kwargs["scope"], provider=KanbanApiSecretProvider.name)
    except Exception as exc:  # noqa: BLE001 — seam import must not crash plugin load
        logger.warning("%s: could not register token route prefix %s: %s", _TAG, KANBAN_API_PREFIX, exc)
    logger.info(
        "%s: registered kanban API service-credential provider (scope=%s, prefix=%s)",
        _TAG, kwargs["scope"], KANBAN_API_PREFIX)
