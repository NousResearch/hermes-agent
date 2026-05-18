"""Shared result types for CMH subprocess wrapper foundations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PreflightResult:
    status: str
    ok: bool
    message: str
    argv: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
