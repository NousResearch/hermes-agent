"""Nextcloud Notifications background-service plugin entry point.

Registers a long-running service via
:meth:`PluginContext.register_background_service` so the gateway can
start it after platforms come up and stop it before they disconnect.

The service polls Nextcloud's Notifications + Activity APIs and routes
matching events to the configured platform (typically Nextcloud Talk).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

try:
    import httpx  # noqa: F401
    _HTTPX = True
except ImportError:
    _HTTPX = False

logger = logging.getLogger(__name__)


def _check_requirements() -> bool:
    return _HTTPX


def _validate_config(cfg: dict) -> bool:
    """The plugin needs at minimum a deliver target and either an env-provided
    password or one of: an explicit ``app_password_env`` in extra, or a
    Talk-platform-side credential we can inherit at runtime (the factory
    handles inheritance — here we just sanity-check the shape).
    """
    extra = cfg.get("extra") or cfg  # tolerate flat-vs-nested config
    return bool(extra.get("deliver"))


def _service_factory(config: dict, gateway_runner: Any) -> Any:
    """Build a ``NextcloudNotificationService`` from the gateway config dict.

    The dict is the raw ``services.nextcloud_notifications`` block from
    config.yaml (including ``enabled``, ``extra``, etc.). We inherit
    Nextcloud credentials from the Talk adapter when ``extra.nextcloud_url``
    is missing so operators don't have to duplicate URL/username.
    """
    from .service import NextcloudNotificationService

    extra = dict(config.get("extra") or {})

    # Inherit NC credentials from the Talk adapter when the operator
    # hasn't duplicated them. Keeps config.yaml DRY for the common case
    # where Notifications + Talk share an account.
    if not extra.get("nextcloud_url") and gateway_runner is not None:
        try:
            from gateway.config import Platform
            talk_cfg = gateway_runner.config.platforms.get(Platform("nextcloud_talk"))
            if talk_cfg and getattr(talk_cfg, "extra", None):
                talk_extra = talk_cfg.extra
                extra.setdefault("nextcloud_url", talk_extra.get("nextcloud_url", ""))
                extra.setdefault("username", talk_extra.get("username", "hermes"))
                extra.setdefault(
                    "app_password_env",
                    talk_extra.get("app_password_env", "NEXTCLOUD_TALK_APP_PASSWORD"),
                )
        except Exception:
            logger.debug("nc-notifications: could not inherit Talk credentials", exc_info=True)

    return NextcloudNotificationService(extra, gateway_runner)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_background_service(
        name="nextcloud_notifications",
        label="Nextcloud Notifications",
        service_factory=_service_factory,
        check_fn=_check_requirements,
        validate_config=_validate_config,
        install_hint="pip install httpx   # already a Hermes dependency",
    )


__all__ = ["register"]
