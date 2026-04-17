"""Resolve :class:`SandboxProvider` from config."""

from __future__ import annotations

from typing import Any

from sandbox.base import SandboxProvider
from sandbox.errors import SandboxConfigError
from sandbox.providers.docker import DockerSandboxProvider
from sandbox.providers.firecracker import FirecrackerSandboxProvider
from sandbox.providers.gvisor import GVisorSandboxProvider
from sandbox.providers.local import LocalSandboxProvider


def get_provider(name: str | None, config: dict[str, Any] | None) -> SandboxProvider:
    """Factory for sandbox backends (keys match ``config.yaml`` ``sandbox.type``)."""
    cfg = dict(config or {})
    key = (name or cfg.get("type") or "local").strip().lower()
    if key in ("local", "host", ""):
        return LocalSandboxProvider(cfg)
    if key == "docker":
        return DockerSandboxProvider(cfg, docker_runtime=None)
    if key in ("gvisor", "runsc"):
        return GVisorSandboxProvider(cfg)
    if key in ("firecracker", "fc"):
        return FirecrackerSandboxProvider(cfg)
    raise SandboxConfigError(f"Unknown sandbox.type: {name!r}")
