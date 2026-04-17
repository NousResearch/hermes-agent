"""Post-exec diff and rollback registration (lightweight defaults)."""

from __future__ import annotations

import filecmp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sandbox.base import SandboxProvider
from sandbox.types import FSSnapshot


@dataclass
class DiffReport:
    """Summary of directory changes between two snapshots."""

    only_left: list[str] = field(default_factory=list)
    only_right: list[str] = field(default_factory=list)
    diff_files: list[str] = field(default_factory=list)
    fun_diff_files: list[str] = field(default_factory=list)


def post_exec_diff(left: Path, right: Path) -> DiffReport:
    """Directory diff using :mod:`filecmp` (best-effort, no VCS)."""
    cmp = filecmp.dircmp(left, right)
    return DiffReport(
        only_left=list(cmp.left_only),
        only_right=list(cmp.right_only),
        diff_files=list(cmp.diff_files),
        fun_diff_files=list(cmp.funny_files),
    )


RollbackFn = Callable[[str], Any]
_REGISTRY: list[tuple[str, RollbackFn]] = []


def register_rollback_hook(name: str, fn: RollbackFn) -> None:
    """Register a named rollback callback (self-evolution loops may use this)."""
    _REGISTRY.append((name, fn))


def clear_rollback_hooks() -> None:
    """Test helper."""
    _REGISTRY.clear()


async def rollback_with_provider(provider: SandboxProvider, snap: FSSnapshot) -> None:
    """Invoke provider rollback then any registered hooks."""
    if snap.snapshot_id:
        await provider.rollback(snap.snapshot_id)
    for _, fn in _REGISTRY:
        fn(snap.snapshot_id)
