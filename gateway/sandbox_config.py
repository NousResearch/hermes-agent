"""Gateway-only overlay for the existing terminal environment bridge."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_BACKENDS = frozenset({"local", "docker", "modal", "daytona", "ssh", "singularity"})
_DEFAULT_SANDBOX_LIFETIME = 3600


def _is_valid_backend(value: object) -> bool:
    return isinstance(value, str) and value in _VALID_BACKENDS


def _gateway_config(config: dict) -> dict:
    gateway = config.get("gateway", {})
    return gateway if isinstance(gateway, dict) else {}


def get_gateway_terminal_backend(config: dict) -> Optional[str]:
    """Return the configured gateway backend override, if any."""
    backend = _gateway_config(config).get("terminal_backend")
    return backend if backend is None or isinstance(backend, str) else str(backend)


def get_gateway_sandbox_lifetime(config: dict) -> int:
    """Return the configured gateway lifetime, or the historical default."""
    value = _gateway_config(config).get("sandbox_lifetime", _DEFAULT_SANDBOX_LIFETIME)
    try:
        if isinstance(value, bool):
            raise ValueError
        lifetime = int(value)
        if lifetime < 0:
            raise ValueError
        return lifetime
    except (TypeError, ValueError):
        logger.warning(
            "Invalid gateway.sandbox_lifetime %r; using %ss",
            value,
            _DEFAULT_SANDBOX_LIFETIME,
        )
        return _DEFAULT_SANDBOX_LIFETIME


def should_warn_insecure_gateway(config: dict) -> bool:
    """Return whether the effective gateway backend is local."""
    terminal = config.get("terminal", {})
    if isinstance(terminal, dict) and "backend" in terminal:
        terminal_backend = terminal.get("backend")
    else:
        terminal_backend = os.environ.get("TERMINAL_ENV", "local")
    gateway_backend = get_gateway_terminal_backend(config)
    effective_backend = (
        gateway_backend if _is_valid_backend(gateway_backend) else terminal_backend
    )
    return effective_backend == "local"



def apply_gateway_backend_to_env(config: dict) -> None:
    """Apply explicit gateway backend/image/lifetime settings to ``TERMINAL_*``.

    The gateway imports this after the normal ``terminal`` config bridge.  Only
    explicitly supplied gateway values are written, so the CLI and an omitted
    gateway section keep the existing terminal behavior.
    """
    gateway = _gateway_config(config)
    backend = gateway.get("terminal_backend")
    if backend is not None:
        if not _is_valid_backend(backend):
            logger.warning(
                "Invalid gateway.terminal_backend %r; keeping the terminal backend. "
                "Valid values: %s",
                backend,
                ", ".join(sorted(_VALID_BACKENDS)),
            )
        else:
            os.environ["TERMINAL_ENV"] = backend

    image = gateway.get("sandbox_image")
    if image is not None:
        if not isinstance(image, str) or not image.strip():
            logger.warning(
                "Invalid gateway.sandbox_image %r; keeping the terminal Docker image.",
                image,
            )
        else:
            os.environ["TERMINAL_DOCKER_IMAGE"] = image

    lifetime = gateway.get("sandbox_lifetime")
    if lifetime is not None:
        try:
            if isinstance(lifetime, bool):
                raise ValueError
            lifetime_value = int(lifetime)
            if lifetime_value < 0:
                raise ValueError
        except (TypeError, ValueError):
            logger.warning(
                "Invalid gateway.sandbox_lifetime %r; keeping TERMINAL_LIFETIME_SECONDS.",
                lifetime,
            )
        else:
            os.environ["TERMINAL_LIFETIME_SECONDS"] = str(lifetime_value)

    if isinstance(backend, str) and backend in _VALID_BACKENDS:
        logger.info(
            "Gateway terminal override: backend=%s image=%s lifetime=%s",
            os.environ.get("TERMINAL_ENV", "local"),
            os.environ.get("TERMINAL_DOCKER_IMAGE", "(terminal default)"),
            os.environ.get("TERMINAL_LIFETIME_SECONDS", "(terminal default)"),
        )
