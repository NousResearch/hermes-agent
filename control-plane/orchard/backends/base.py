"""Backend interface. A backend owns the *lifecycle* and *isolation* of workers.

Implementations decide what a "sandbox" is:
  - local  : a subprocess on the shared host (FS-perm isolation only)
  - docker : a container per employee (namespaces — real OS boundary)
  - microvm: (future) Firecracker/gVisor per employee with snapshot restore

All methods are async and safe to call repeatedly (idempotent where noted).
"""
from __future__ import annotations

import abc

from ..config import Settings
from ..models import Employee


class WorkerBackend(abc.ABC):
    def __init__(self, settings: Settings):
        self.settings = settings

    @abc.abstractmethod
    async def is_ready(self, employee: Employee) -> bool:
        """True if the worker sandbox is up and answering."""

    @abc.abstractmethod
    async def ensure_ready(self, employee: Employee) -> None:
        """Wake the worker if asleep and block until it can serve (or raise)."""

    @abc.abstractmethod
    async def send(self, employee: Employee, session: str, message: str) -> str:
        """Deliver a message to the worker's agent and return its reply."""

    @abc.abstractmethod
    async def sleep(self, employee: Employee) -> None:
        """Tear down the worker sandbox to reclaim resources. Idempotent."""

    async def shutdown_all(self) -> None:
        """Best-effort cleanup on control-plane exit."""
        return None
