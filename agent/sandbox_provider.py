"""Provider-neutral sandbox runtime contract for Hermes execution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SandboxSpec:
    """Inputs needed by a sandbox provider to create a Hermes environment."""

    provider: str
    mode: str
    task_id: str
    scope: str
    cwd: str
    host_cwd: str
    timeout: int
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxRuntimeInfo:
    """Lightweight status returned by sandbox providers for CLI/UI display."""

    provider: str
    runtime_id: str
    running: bool
    mode: str = ""
    source: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class SandboxProvider(ABC):
    """Base class for sandbox provider plugins.

    Providers return objects compatible with ``tools.environments.base.BaseEnvironment``.
    The concrete return type is intentionally not enforced here to keep the
    plugin contract decoupled from one implementation module.
    """

    name: str = ""
    description: str = ""

    def is_available(self) -> bool:
        """Return whether the provider can run in the current process."""

        return True

    @abstractmethod
    def create_environment(self, spec: SandboxSpec):
        """Create an execution environment for *spec*."""

    def describe_runtime(self, scope: str, config: Optional[dict[str, Any]] = None) -> SandboxRuntimeInfo:
        """Return best-effort runtime status for a scope."""

        return SandboxRuntimeInfo(
            provider=self.name,
            runtime_id=scope,
            running=False,
            details={"message": "status not implemented"},
        )

    def delete_runtime(self, scope: str, config: Optional[dict[str, Any]] = None) -> None:
        """Delete the runtime for *scope* if the provider supports it."""

        raise NotImplementedError(f"{self.name} does not support runtime deletion")
