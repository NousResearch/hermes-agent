"""
Nomad/Docker tool backend.

This backend is the current default for AgentEnv: it provisions a Nomad job
running `sandbox_server.py` and multiplexes stateless slots inside each container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..slots import Slot, SlotPool, SlotPoolConfig
from ..slots.executor import ExecutionResult
from .base import ToolBackend


@dataclass(frozen=True)
class NomadBackendConfig:
    nomad_address: str
    sandbox_job_id: str
    sandbox_image: str
    slots_per_container: int
    min_containers: int
    max_containers: int
    privileged: bool
    acquire_timeout_s: float
    purge_job_on_start: bool

    @classmethod
    def from_agent_env_config(cls, cfg: Any) -> "NomadBackendConfig":
        return cls(
            nomad_address=str(getattr(cfg, "nomad_address")),
            sandbox_job_id=str(getattr(cfg, "sandbox_job_id")),
            sandbox_image=str(getattr(cfg, "sandbox_image")),
            slots_per_container=int(getattr(cfg, "slots_per_container")),
            min_containers=int(getattr(cfg, "min_containers")),
            max_containers=int(getattr(cfg, "max_containers")),
            privileged=bool(getattr(cfg, "privileged")),
            acquire_timeout_s=float(getattr(cfg, "acquire_timeout_s")),
            purge_job_on_start=bool(getattr(cfg, "purge_job_on_start", False)),
        )


class NomadToolBackend(ToolBackend):
    def __init__(self, config: NomadBackendConfig):
        self.config = config
        self.pool = SlotPool(
            SlotPoolConfig(
                nomad_address=config.nomad_address,
                job_id=config.sandbox_job_id,
                image=config.sandbox_image,
                slots_per_container=config.slots_per_container,
                min_containers=config.min_containers,
                max_containers=config.max_containers,
                privileged=config.privileged,
                acquire_timeout=config.acquire_timeout_s,
                purge_job_on_start=bool(config.purge_job_on_start),
            )
        )

    @property
    def default_timeout_s(self) -> Optional[float]:
        t = getattr(self.pool.executor, "timeout", None)
        total = getattr(t, "total", None)
        try:
            return float(total) if total is not None else None
        except Exception:
            return None

    async def start(self) -> None:
        await self.pool.start()

    async def stop(self, *, purge: bool = False) -> None:
        await self.pool.stop(purge_job=purge)

    async def acquire(self, trajectory_id: Optional[str] = None) -> Slot:
        return await self.pool.acquire(trajectory_id)

    async def release(self, slot: Slot, *, reset_workspace: bool = False) -> None:
        await self.pool.release(slot, reset_workspace=reset_workspace)

    async def execute_batch(
        self,
        requests: List[Tuple[Slot, str, Dict[str, Any]]],
        *,
        timeout_s: Optional[float] = None,
    ) -> List[ExecutionResult]:
        return await self.pool.execute_batch(requests, timeout=timeout_s)

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
        return await self.pool.executor.read_artifact(
            slot,
            path,
            encoding=encoding,
            max_bytes=max_bytes,
            include_sha256=include_sha256,
            timeout=timeout_s,
        )

    async def list_artifacts(
        self,
        slot: Slot,
        path: str = ".",
        *,
        recursive: bool = False,
        max_entries: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        return await self.pool.executor.list_artifacts(
            slot,
            path,
            recursive=recursive,
            max_entries=max_entries,
            timeout=timeout_s,
        )

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
        return await self.pool.executor.archive_artifacts(
            slot,
            path,
            archive_format=archive_format,
            max_bytes=max_bytes,
            max_entries=max_entries,
            timeout=timeout_s,
        )

    def get_stats(self) -> Dict[str, Any]:
        return self.pool.get_stats()

