"""Native update seam for Hermes CLI.

This module intentionally keeps the first slice tiny: pure request/result/event
dataclasses plus a narrow delegation object that can be driven by legacy
behavior for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class UpdateRequest:
    """Request passed into the native update seam."""

    args: Any
    gateway_mode: bool = False


@dataclass(frozen=True)
class UpdateResult:
    """Result produced by the update engine."""

    exit_code: int = 0
    message: str = ""


@dataclass(frozen=True)
class UpdateEvent:
    """Structured event emitted by the update engine."""

    kind: str
    message: str = ""
    details: tuple[tuple[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NativeUpdateEngine:
    """Tiny seam that delegates to an injected legacy runner for now."""

    legacy_runner: Callable[[UpdateRequest], UpdateResult]

    def run(self, request: UpdateRequest) -> UpdateResult:
        return self.legacy_runner(request)