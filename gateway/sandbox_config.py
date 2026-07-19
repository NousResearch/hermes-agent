"""Gateway-only overlay for the existing terminal environment bridge."""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_BACKENDS = frozenset({"local", "docker", "modal", "daytona", "ssh", "singularity"})


def is_valid_gateway_backend(value: object) -> bool:
    """Return whether a gateway backend is supported by the terminal factory."""
    return isinstance(value, str) and value in _VALID_BACKENDS


def parse_gateway_sandbox_lifetime(value: object, default: Optional[int] = None) -> Optional[int]:
    """Parse an integer gateway lifetime without truncating or propagating errors."""
    try:
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
            raise ValueError
        lifetime = int(value)
        if lifetime < 0:
            raise ValueError
        return lifetime
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "Invalid gateway.sandbox_lifetime %r; using %s",
            value,
            f"{default}s" if default is not None else "the terminal lifetime",
        )
        return default


def _gateway_config(config: object) -> dict:
    if not isinstance(config, dict):
        return {}
    gateway = config.get("gateway", {})
    return gateway if isinstance(gateway, dict) else {}


def should_warn_insecure_gateway(config: object) -> bool:
    """Return whether the applied gateway backend is local."""
    return os.environ.get("TERMINAL_ENV", "local") == "local"


def apply_gateway_backend_to_env(config: object) -> None:
    """Apply explicit gateway backend/image/lifetime settings to ``TERMINAL_*``.

    The gateway imports this after the normal ``terminal`` config bridge.  Only
    explicitly supplied gateway values are written, so the CLI and an omitted
    gateway section keep the existing terminal behavior.
    """
    gateway = _gateway_config(config)
    backend = gateway.get("terminal_backend")
    if backend is not None:
        if not is_valid_gateway_backend(backend):
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
            os.environ["TERMINAL_DOCKER_IMAGE"] = image.strip()

    lifetime = gateway.get("sandbox_lifetime")
    if lifetime is not None:
        lifetime_value = parse_gateway_sandbox_lifetime(lifetime)
        if lifetime_value is not None:
            os.environ["TERMINAL_LIFETIME_SECONDS"] = str(lifetime_value)
    if is_valid_gateway_backend(backend):
        logger.info(
            "Gateway terminal override: backend=%s image=%s lifetime=%s",
            os.environ.get("TERMINAL_ENV", "local"),
            os.environ.get("TERMINAL_DOCKER_IMAGE", "(terminal default)"),
            os.environ.get("TERMINAL_LIFETIME_SECONDS", "(terminal default)"),
        )
