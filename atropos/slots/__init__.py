"""
Slot-based multiplexing for atropos-agent.

Provides:
- Slot: Isolated workspace for a single trajectory
- SlotPool: Manages slots across Nomad allocations  
- SandboxExecutor: Executes tools in sandbox containers
"""

from .executor import SandboxExecutor
from .pool import SlotPool, SlotPoolConfig
from .slot import Slot, SlotState

__all__ = [
    "Slot",
    "SlotState",
    "SlotPool",
    "SlotPoolConfig",
    "SandboxExecutor",
]
