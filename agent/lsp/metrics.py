"""In-process counters and timing accumulators for the LSP service.

Exposed through :meth:`LSPService.snapshot_metrics` so the gateway can log
structured metrics, and through :meth:`LSPService.eventlog`-style counters
that callers (and tests) can assert on without touching the gateway log.

Counters use only stdlib and live entirely in-process; if Hermes ever
ships an OpenTelemetry exporter, this module is the seam.
"""
from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict


class LSPMetrics:
    """Thread-safe in-memory counters."""

    __slots__ = (
        "spawns",
        "reuses",
        "reaps_idle",
        "reaps_cap",
        "reaps_blocked_active",
        "shutdowns",
        "shutdown_failures",
        "spawn_failures",
        "_lock",
    )

    def __init__(self) -> None:
        self.spawns: int = 0
        self.reuses: int = 0
        self.reaps_idle: int = 0
        self.reaps_cap: int = 0
        self.reaps_blocked_active: int = 0
        self.shutdowns: int = 0
        self.shutdown_failures: int = 0
        self.spawn_failures: int = 0
        self._lock = Lock()

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + by)

    def as_dict(self) -> Dict[str, int]:
        with self._lock:
            return {
                "spawns": self.spawns,
                "reuses": self.reuses,
                "reaps_idle": self.reaps_idle,
                "reaps_cap": self.reaps_cap,
                "reaps_blocked_active": self.reaps_blocked_active,
                "shutdowns": self.shutdowns,
                "shutdown_failures": self.shutdown_failures,
                "spawn_failures": self.spawn_failures,
            }


__all__ = ["LSPMetrics"]
