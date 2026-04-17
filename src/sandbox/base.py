"""Abstract sandbox execution provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sandbox.types import FSSnapshot, SandboxExecResult


class SandboxProvider(ABC):
    """Pluggable backend for isolated skill / code-exec runs.

    Composes with :class:`tools.environments.base.BaseEnvironment` — terminal
    transport stays in ``tools/environments/*``; this type covers extra
    isolation for short-lived child processes (e.g. ``execute_code``).
    """

    def __init__(self, config: dict[str, Any]):
        self._config = dict(config or {})

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @abstractmethod
    async def exec_cmd(
        self,
        argv: list[str],
        cwd: str,
        env: dict[str, str],
        timeout_sec: float,
    ) -> SandboxExecResult:
        """Run *argv* with working directory *cwd* and environment *env*."""

    async def run_skill(
        self,
        command: str,
        cwd: str,
        env: dict[str, str],
        timeout_sec: float,
    ) -> SandboxExecResult:
        """Default: run via ``/bin/sh -c`` (override for native skill runners)."""
        return await self.exec_cmd(["/bin/sh", "-c", command], cwd, env, timeout_sec)

    async def snapshot_fs(self, paths: list[str]) -> FSSnapshot:
        """Optional filesystem snapshot for rollback hooks (stub by default)."""
        return FSSnapshot(snapshot_id="", root=paths[0] if paths else "")

    async def rollback(self, snapshot_id: str) -> None:
        """Restore *snapshot_id* if supported."""
        del snapshot_id
        return None
