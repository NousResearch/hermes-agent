"""
Backend interfaces for AgentEnv tool execution.

The goal of this module is to decouple ToolExecutor / AgentEnv from any single
execution backend (Nomad/Docker today; Modal later).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..slots.executor import ExecutionResult
from ..slots.slot import Slot


class ToolBackend(Protocol):
    """
    Minimal interface required by ToolExecutor.

    Backends provide:
    - lifecycle (start/stop)
    - slot acquisition/release (workspace affinity)
    - batched tool execution across slots
    - optional artifact helpers (for env verification / demos)
    """

    @property
    def default_timeout_s(self) -> Optional[float]:
        """Default sandbox execution timeout in seconds (if any)."""

    async def start(self) -> None:
        """Start the backend (provision workers/containers, health checks, etc)."""

    async def stop(self, *, purge: bool = False) -> None:
        """Stop the backend and optionally purge remote resources."""

    async def acquire(self, trajectory_id: Optional[str] = None) -> Slot:
        """Acquire a slot for a trajectory (workspace affinity)."""

    async def release(self, slot: Slot, *, reset_workspace: bool = False) -> None:
        """Release a slot back to the pool."""

    async def execute_batch(
        self,
        requests: List[Tuple[Slot, str, Dict[str, Any]]],
        *,
        timeout_s: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """Execute a batch of sandbox tool calls and return results in order."""

    # ---------------------------------------------------------------------
    # Optional artifact helpers (supported by the Nomad sandbox-server today)
    # ---------------------------------------------------------------------

    async def read_artifact(
        self,
        slot: Slot,
        path: str,
        *,
        encoding: str = "text",
        max_bytes: Optional[int] = None,
        include_sha256: bool = False,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def list_artifacts(
        self,
        slot: Slot,
        path: str = ".",
        *,
        recursive: bool = False,
        max_entries: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def archive_artifacts(
        self,
        slot: Slot,
        path: str = ".",
        *,
        archive_format: str = "tar.gz",
        max_bytes: Optional[int] = None,
        max_entries: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

