"""gVisor (runsc) via Docker's alternate OCI runtime."""

from __future__ import annotations

from typing import Any

from sandbox.providers.docker import DockerSandboxProvider


class GVisorSandboxProvider(DockerSandboxProvider):
    """Same as Docker with ``--runtime=runsc`` (or configured runtime)."""

    def __init__(self, config: dict[str, Any]):
        gv = config.get("gvisor") if isinstance(config.get("gvisor"), dict) else {}
        runtime = str((gv or {}).get("docker_runtime") or "runsc").strip() or "runsc"
        super().__init__(config, docker_runtime=runtime)
