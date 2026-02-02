"""
Slot abstraction for atropos-agent.

A Slot represents an isolated workspace for a single agent trajectory.
Slots are hosted on Nomad allocations and provide workspace isolation
via filesystem directories.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class SlotState(Enum):
    """State of a slot in the pool."""
    AVAILABLE = "available"      # Ready to be acquired
    ACQUIRED = "acquired"        # Assigned to a trajectory
    EXECUTING = "executing"      # Currently executing a tool
    RELEASING = "releasing"      # Being released back to pool
    ERROR = "error"              # In error state


@dataclass
class Slot:
    """
    An isolated workspace for a single agent trajectory.
    
    Slots are the unit of scheduling - each trajectory runs in its own slot,
    with an isolated workspace directory. Multiple slots share a container.
    
    Attributes:
        slot_id: Unique identifier for this slot (e.g., "slot_0")
        alloc_id: Nomad allocation ID hosting this slot
        container_addr: HTTP address of the sandbox server (e.g., "http://10.0.0.1:8080")
        workspace_dir: Path to workspace in container (e.g., "/data/slot_0")
        state: Current state of the slot
        trajectory_id: ID of trajectory currently using this slot (if acquired)
        metadata: Additional metadata
    """
    slot_id: str
    alloc_id: str
    container_addr: str
    workspace_dir: str = ""
    state: SlotState = SlotState.AVAILABLE
    trajectory_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default workspace_dir if not provided."""
        if not self.workspace_dir:
            self.workspace_dir = f"/data/{self.slot_id}"
    
    @property
    def is_available(self) -> bool:
        """Check if slot is available for acquisition."""
        return self.state == SlotState.AVAILABLE
    
    @property
    def is_acquired(self) -> bool:
        """Check if slot is currently acquired."""
        return self.state in (SlotState.ACQUIRED, SlotState.EXECUTING)
    
    def acquire(self, trajectory_id: Optional[str] = None) -> None:
        """
        Mark slot as acquired by a trajectory.
        
        Args:
            trajectory_id: Optional ID of acquiring trajectory
        """
        if not self.is_available:
            raise RuntimeError(f"Cannot acquire slot {self.slot_id}: state is {self.state}")
        
        self.state = SlotState.ACQUIRED
        self.trajectory_id = trajectory_id or str(uuid.uuid4())
    
    def start_execution(self, execution_id: Optional[str] = None) -> None:
        """Mark slot as executing."""
        if self.state != SlotState.ACQUIRED:
            raise RuntimeError(f"Cannot start execution on slot {self.slot_id}: state is {self.state}")
        
        self.state = SlotState.EXECUTING
        if execution_id:
            self.metadata["current_execution_id"] = execution_id
    
    def end_execution(self) -> None:
        """Mark execution as complete, return to acquired state."""
        if self.state != SlotState.EXECUTING:
            raise RuntimeError(f"Cannot end execution on slot {self.slot_id}: state is {self.state}")
        
        self.state = SlotState.ACQUIRED
        self.metadata.pop("current_execution_id", None)
    
    def release(self) -> None:
        """Release slot back to available state."""
        self.state = SlotState.AVAILABLE
        self.trajectory_id = None
        self.metadata.pop("current_execution_id", None)
    
    def mark_error(self, error: str) -> None:
        """Mark slot as in error state."""
        self.state = SlotState.ERROR
        self.metadata["error"] = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "slot_id": self.slot_id,
            "alloc_id": self.alloc_id,
            "container_addr": self.container_addr,
            "workspace_dir": self.workspace_dir,
            "state": self.state.value,
            "trajectory_id": self.trajectory_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Slot":
        """Create from dictionary."""
        return cls(
            slot_id=data["slot_id"],
            alloc_id=data["alloc_id"],
            container_addr=data["container_addr"],
            workspace_dir=data.get("workspace_dir", ""),
            state=SlotState(data.get("state", "available")),
            trajectory_id=data.get("trajectory_id"),
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        return f"Slot({self.slot_id}, state={self.state.value}, alloc={self.alloc_id[:8]}...)"


def create_slots_for_allocation(
    alloc_id: str,
    container_addr: str,
    num_slots: int = 10,
) -> list["Slot"]:
    """
    Create slots for a Nomad allocation.
    
    Args:
        alloc_id: Nomad allocation ID
        container_addr: HTTP address of sandbox server
        num_slots: Number of slots to create
        
    Returns:
        List of Slot objects
    """
    slots = []
    for i in range(num_slots):
        slot_id = f"slot_{i}"
        slots.append(Slot(
            slot_id=slot_id,
            alloc_id=alloc_id,
            container_addr=container_addr,
            workspace_dir=f"/data/{slot_id}",
        ))
    return slots
