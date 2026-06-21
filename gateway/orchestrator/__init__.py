"""Parallel external-agent orchestration primitives.

Phase 1~2 are intentionally additive and not wired into gateway commands yet:
agent doctor, dry-run lane execution, and structured synthesis only.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_doctor", "run_lanes", "synthesize"]


def __getattr__(name: str) -> Any:
    """Lazy exports avoid importing submodules before ``python -m`` execution."""

    if name == "run_doctor":
        from .doctor import run_doctor

        return run_doctor
    if name == "run_lanes":
        from .runner import run_lanes

        return run_lanes
    if name == "synthesize":
        from .synthesis import synthesize

        return synthesize
    raise AttributeError(name)
