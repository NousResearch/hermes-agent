"""Isolation backends: how/where a worker sandbox runs."""
from __future__ import annotations

from ..config import Settings
from .base import WorkerBackend


def make_backend(settings: Settings) -> WorkerBackend:
    if settings.backend == "local":
        from .local import LocalBackend
        return LocalBackend(settings)
    if settings.backend == "docker":
        from .docker import DockerBackend
        return DockerBackend(settings)
    raise ValueError(f"unknown backend {settings.backend!r}")
