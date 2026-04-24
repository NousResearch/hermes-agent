"""Abstract Sandbox protocol for per-thread execution environments."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


class SandboxError(Exception):
    """Raised for sandbox-level failures (path escape, not acquired, etc.)."""


@runtime_checkable
class Sandbox(Protocol):
    """A per-thread sandbox — minimally supports workspace + path translation.

    Instance lifecycle:
        provider.acquire(thread_id) → Sandbox
        sandbox.workspace_dir        (pathlib.Path)
        sandbox.uploads_dir          (pathlib.Path)
        sandbox.outputs_dir          (pathlib.Path)
        sandbox.translate(virtual_path) → real path
        sandbox.read_file(virtual_path) → str
        sandbox.write_file(virtual_path, content) → None
        sandbox.list_dir(virtual_path) → list[str]
        provider.release(thread_id)  (optional teardown)

    `sandbox_id` uniquely identifies this sandbox instance — e.g. thread_id
    or "local" for the shared local provider.
    """

    id: str
    thread_id: str

    @property
    def workspace_dir(self) -> Path: ...

    @property
    def uploads_dir(self) -> Path: ...

    @property
    def outputs_dir(self) -> Path: ...

    def translate(self, virtual_path: str) -> Path: ...

    def read_file(self, virtual_path: str) -> str: ...

    def write_file(self, virtual_path: str, content: str) -> None: ...

    def list_dir(self, virtual_path: str) -> list[str]: ...


class SandboxProvider(Protocol):
    """Factory + lifecycle owner for Sandbox instances."""

    def acquire(self, thread_id: str) -> Sandbox: ...

    def get(self, thread_id: str) -> Sandbox | None: ...

    def release(self, thread_id: str) -> None: ...
