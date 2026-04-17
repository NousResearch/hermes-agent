"""Detect Docker / Podman CLI (diagnostics and CI only, not per-write hot path)."""

from __future__ import annotations

import shutil
from typing import Literal

ContainerRuntime = Literal["docker", "podman", "none"]


def detect_container_runtime() -> ContainerRuntime:
    """Return which container CLI is available on ``PATH``."""
    if shutil.which("docker"):
        return "docker"
    if shutil.which("podman"):
        return "podman"
    return "none"
