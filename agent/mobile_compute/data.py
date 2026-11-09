"""Data structures for Mobile Compute Node.

This module contains shared data structures used throughout the Mobile Compute Node
architecture, including result types and status indicators.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ComputeResult:
    """Result from a compute request to the iPhone.

    Attributes
    ----------
    success : bool
        Whether the task was submitted successfully.
    task_id : str
        Unique identifier for the task.
    status : str
        Current status: "queued" | "running" | "completed" | "failed".
    result : Any, optional
        The computed result, if available.
    error : str, optional
        Error message if the task failed.
    latency_ms : float
        Request latency in milliseconds.
    """
    success: bool
    task_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class MobileComputeBridgeStatus:
    """Current status of the Mobile Compute Bridge.

    Attributes
    ----------
    bridge_started : bool
        Whether the bridge has been started.
    iphone_connected : bool
        Whether the iPhone is connected.
    iphone_healthy : bool
        Whether the iPhone is healthy (responds to health checks).
    uptime_seconds : float
        How long the bridge has been running.
    task_count : int
        Number of active tasks.
    errors : list
        List of error messages.
    """
    bridge_started: bool = False
    iphone_connected: bool = False
    iphone_healthy: bool = False
    uptime_seconds: float = 0.0
    task_count: int = 0
    errors: list = field(default_factory=list)