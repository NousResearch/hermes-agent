"""Firecracker microVM placeholder (external API client required)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sandbox.base import SandboxProvider
from sandbox.errors import SandboxNotSupportedError
from sandbox.types import SandboxExecResult


@runtime_checkable
class FirecrackerClient(Protocol):
    """Process boundary for a real Firecracker integration."""

    def start_vm(self) -> str: ...
    def exec(self, argv: list[str], cwd: str, env: dict[str, str]) -> tuple[str, str, int]: ...
    def stop(self, vm_id: str) -> None: ...


class FirecrackerSandboxProvider(SandboxProvider):
    """Stub provider — wire a :class:`FirecrackerClient` before production use."""

    async def exec_cmd(
        self,
        argv: list[str],
        cwd: str,
        env: dict[str, str],
        timeout_sec: float,
    ) -> SandboxExecResult:
        del argv, cwd, env, timeout_sec
        raise SandboxNotSupportedError(
            "Firecracker sandbox execution is not wired in this build; "
            "use sandbox.type=docker or gvisor, or attach a FirecrackerClient."
        )
