"""Nextcloud Files background-service plugin entry point.

Registers a long-running bidirectional WebDAV sync service via
:meth:`PluginContext.register_background_service` so the gateway can
start it after platforms come up and stop it before they disconnect.

The service syncs a remote Nextcloud folder with a local working
directory via notify_push (incoming) + inotify (outgoing) and uses
chunked uploads for files above the configured chunk size.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import httpx  # noqa: F401
    _HTTPX = True
except ImportError:
    _HTTPX = False

logger = logging.getLogger(__name__)


def _check_requirements() -> bool:
    return _HTTPX


def _validate_config(cfg: dict) -> bool:
    extra = cfg.get("extra") or cfg
    return bool(extra.get("local_path") or extra.get("deliver"))


def _service_factory(config: dict, gateway_runner: Any) -> Any:
    """Build a ``NextcloudFilesService`` from the gateway config dict.

    Inherits Nextcloud credentials from the Talk adapter when the
    ``extra`` dict doesn't supply them explicitly, so config.yaml
    stays DRY across NC services.
    """
    from .service import NextcloudFilesService

    extra = dict(config.get("extra") or {})

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
            logger.debug("nc-files: could not inherit Talk credentials", exc_info=True)

    return NextcloudFilesService(extra, gateway_runner)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_background_service(
        name="nextcloud_files",
        label="Nextcloud Files",
        service_factory=_service_factory,
        check_fn=_check_requirements,
        validate_config=_validate_config,
        install_hint="pip install httpx   # already a Hermes dependency",
    )


__all__ = ["register"]
