"""
Modal tool backend (stub).

We intentionally ship a placeholder implementation so AgentEnv can expose a
backend switch without forcing Modal as a hard dependency for Hermes-Agent.

When org access is available, this backend will be implemented by running a
long-lived Modal worker (or pool) that owns N slots and exposes `execute_batch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..slots.executor import ExecutionResult
from ..slots.slot import Slot
from .base import ToolBackend


@dataclass(frozen=True)
class ModalBackendConfig:
    # Placeholders for future implementation.
    app_name: str = "atropos-sandbox"
    function_name: str = "sandbox_server"
    volume_name: Optional[str] = None
    volume_mount_path: str = "/data"

    @classmethod
    def from_agent_env_config(cls, cfg: Any) -> "ModalBackendConfig":
        return cls(
            app_name=str(getattr(cfg, "modal_app_name", cls.app_name)),
            function_name=str(getattr(cfg, "modal_function_name", cls.function_name)),
            volume_name=(getattr(cfg, "modal_volume_name", None) or None),
            volume_mount_path=str(getattr(cfg, "modal_volume_mount_path", cls.volume_mount_path)),
        )


class ModalToolBackend(ToolBackend):
    def __init__(self, config: ModalBackendConfig):
        self.config = config

    @property
    def default_timeout_s(self) -> Optional[float]:
        return None

    def _unavailable(self) -> RuntimeError:
        return RuntimeError(
            "Modal tool backend is not implemented yet. "
            "Keep `--env.tool_pool_mode nomad` for now."
        )

    async def start(self) -> None:
        raise self._unavailable()

    async def stop(self, *, purge: bool = False) -> None:  # noqa: ARG002
        # If start() isn't implemented, stop() is also unavailable.
        raise self._unavailable()

    async def acquire(self, trajectory_id: Optional[str] = None) -> Slot:  # noqa: ARG002
        raise self._unavailable()

    async def release(self, slot: Slot, *, reset_workspace: bool = False) -> None:  # noqa: ARG002
        raise self._unavailable()

    async def execute_batch(
        self,
        requests: List[Tuple[Slot, str, Dict[str, Any]]],
        *,
        timeout_s: Optional[float] = None,  # noqa: ARG002
    ) -> List[ExecutionResult]:
        raise self._unavailable()

