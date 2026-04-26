"""
Abstract interface for the data-viz → Manim bridge.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import DataVizBridgeRequest, DataVizBridgeResult


@runtime_checkable
class DataVizBridgeProtocol(Protocol):
    """Stateless bridge from structured viz intent to disk artifacts."""

    def build(self, request: DataVizBridgeRequest) -> DataVizBridgeResult:
        """
        Render a chart asset and emit a Manim CE scene script.

        Must use only tempfile/pathlib for IO; no writes outside the returned work_dir.
        """
        ...
