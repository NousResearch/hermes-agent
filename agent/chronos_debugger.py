"""Hermes Chronos Time-Travel / Branching Tree-of-Thought Trajectory Debugger.

Enables non-destructive timeline branching, state snapshotting, deterministic
rewind, and trajectory diffing across multi-step agent reasoning sessions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("hermes.chronos")


def get_hermes_dir() -> Path:
    """Resolve Hermes home directory."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except ImportError:
        return Path(os.path.expanduser("~/.hermes"))


@dataclass
class WorkspacePatch:
    """Snapshot of a file modification occurring in an execution frame."""
    filepath: str
    before_hash: Optional[str]
    after_hash: Optional[str]
    diff_content: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspacePatch":
        return cls(
            filepath=str(data.get("filepath", "")),
            before_hash=data.get("before_hash"),
            after_hash=data.get("after_hash"),
            diff_content=data.get("diff_content"),
        )


@dataclass
class ChronosFrame:
    """A discrete node in the Trajectory DAG capturing full system state at a step."""
    frame_id: str
    parent_frame_id: Optional[str]
    branch_name: str
    turn_index: int
    thought: str
    action_name: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    messages_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    workspace_patches: List[WorkspacePatch] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["workspace_patches"] = [p.to_dict() for p in self.workspace_patches]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ChronosFrame":
        patches = [WorkspacePatch.from_dict(p) for p in data.get("workspace_patches", [])]
        return cls(
            frame_id=str(data.get("frame_id", "")),
            parent_frame_id=data.get("parent_frame_id"),
            branch_name=str(data.get("branch_name", "main")),
            turn_index=int(data.get("turn_index", 0)),
            thought=str(data.get("thought", "")),
            action_name=data.get("action_name"),
            action_args=data.get("action_args"),
            observation=data.get("observation"),
            messages_snapshot=list(data.get("messages_snapshot", [])),
            memory_snapshot=dict(data.get("memory_snapshot", {})),
            workspace_patches=patches,
            timestamp=float(data.get("timestamp", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


class ChronosTrajectoryDebugger:
    """Manages the Directed Acyclic Graph of agent execution timelines and time-travel operations."""

    def __init__(self, session_id: Optional[str] = None, storage_dir: Optional[Path] = None):
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        if storage_dir is None:
            storage_dir = get_hermes_dir() / "chronos" / self.session_id
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.storage_dir / "trajectory_dag.json"

        self.frames: Dict[str, ChronosFrame] = {}
        self.branches: Dict[str, str] = {}  # branch_name -> head_frame_id
        self.active_branch: str = "main"
        self.current_frame_id: Optional[str] = None
        self.load_dag()

    def record_step(
        self,
        thought: str,
        action_name: Optional[str] = None,
        action_args: Optional[Dict[str, Any]] = None,
        observation: Optional[str] = None,
        messages_snapshot: Optional[List[Dict[str, Any]]] = None,
        memory_snapshot: Optional[Dict[str, Any]] = None,
        workspace_patches: Optional[List[WorkspacePatch]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChronosFrame:
        """Capture and commit a new execution frame onto the active branch."""
        parent_id = self.current_frame_id
        turn = 0
        if parent_id and parent_id in self.frames:
            turn = self.frames[parent_id].turn_index + 1

        frame_id = f"frame-{self.active_branch}-{turn}-{uuid.uuid4().hex[:6]}"
        frame = ChronosFrame(
            frame_id=frame_id,
            parent_frame_id=parent_id,
            branch_name=self.active_branch,
            turn_index=turn,
            thought=thought,
            action_name=action_name,
            action_args=copy.deepcopy(action_args),
            observation=observation,
            messages_snapshot=copy.deepcopy(messages_snapshot or []),
            memory_snapshot=copy.deepcopy(memory_snapshot or {}),
            workspace_patches=workspace_patches or [],
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self.frames[frame_id] = frame
        self.current_frame_id = frame_id
        self.branches[self.active_branch] = frame_id
        self.save_dag()
        logger.debug("Committed Chronos frame %s (branch: %s, turn: %d)", frame_id, self.active_branch, turn)
        return frame

    def rewind(self, target_frame_id: str) -> ChronosFrame:
        """Rewind active state to a specified historical frame."""
        if target_frame_id not in self.frames:
            raise KeyError(f"Frame '{target_frame_id}' does not exist in trajectory DAG.")

        target = self.frames[target_frame_id]
        self.current_frame_id = target_frame_id
        self.active_branch = target.branch_name
        self.branches[self.active_branch] = target_frame_id
        self.save_dag()
        logger.info("Rewound Chronos timeline to frame %s on branch %s", target_frame_id, self.active_branch)
        return target

    def fork_branch(self, from_frame_id: str, new_branch_name: str) -> ChronosFrame:
        """Branch off into an alternate timeline from a historical frame."""
        if from_frame_id not in self.frames:
            raise KeyError(f"Frame '{from_frame_id}' does not exist.")
        if new_branch_name in self.branches:
            raise ValueError(f"Branch '{new_branch_name}' already exists.")

        base_frame = self.frames[from_frame_id]
        self.active_branch = new_branch_name
        self.current_frame_id = from_frame_id
        self.branches[new_branch_name] = from_frame_id
        self.save_dag()
        logger.info("Forked new Chronos branch '%s' from frame %s", new_branch_name, from_frame_id)
        return base_frame

    def get_lineage(self, frame_id: Optional[str] = None) -> List[ChronosFrame]:
        """Return chronological path from root frame to the target frame."""
        fid = frame_id or self.current_frame_id
        path: List[ChronosFrame] = []
        visited: Set[str] = set()
        while fid and fid in self.frames and fid not in visited:
            visited.add(fid)
            f = self.frames[fid]
            path.append(f)
            fid = f.parent_frame_id
        path.reverse()
        return path

    def diff_frames(self, frame_a_id: str, frame_b_id: str) -> Dict[str, Any]:
        """Compare trajectory states between two frames (or branches)."""
        if frame_a_id not in self.frames or frame_b_id not in self.frames:
            raise KeyError("Both frame IDs must exist in DAG.")

        fa = self.frames[frame_a_id]
        fb = self.frames[frame_b_id]

        # Memory diff
        keys_a = set(fa.memory_snapshot.keys())
        keys_b = set(fb.memory_snapshot.keys())
        added_mem = {k: fb.memory_snapshot[k] for k in keys_b - keys_a}
        removed_mem = {k: fa.memory_snapshot[k] for k in keys_a - keys_b}
        changed_mem = {
            k: {"from": fa.memory_snapshot[k], "to": fb.memory_snapshot[k]}
            for k in keys_a & keys_b
            if fa.memory_snapshot[k] != fb.memory_snapshot[k]
        }

        # Action diff
        action_diff = {
            "frame_a_action": fa.action_name,
            "frame_b_action": fb.action_name,
            "frame_a_turn": fa.turn_index,
            "frame_b_turn": fb.turn_index,
        }

        return {
            "frame_a": frame_a_id,
            "frame_b": frame_b_id,
            "branch_a": fa.branch_name,
            "branch_b": fb.branch_name,
            "action_diff": action_diff,
            "memory_diff": {
                "added": added_mem,
                "removed": removed_mem,
                "modified": changed_mem,
            },
            "messages_count_diff": len(fb.messages_snapshot) - len(fa.messages_snapshot),
        }

    def render_ascii_tree(self) -> str:
        """Render a readable ASCII visualizer of the trajectory DAG branches."""
        if not self.frames:
            return "(Empty Trajectory DAG)"

        lines = ["=== Chronos Trajectory DAG ==="]
        for branch, head_id in sorted(self.branches.items()):
            active_marker = " [ACTIVE]" if branch == self.active_branch else ""
            lines.append(f"Branch: {branch}{active_marker}")
            lineage = self.get_lineage(head_id)
            for i, f in enumerate(lineage):
                is_cur = " (*) " if f.frame_id == self.current_frame_id else "     "
                indent = "  " * i + "+-"
                act = f" -> {f.action_name}" if f.action_name else ""
                obs = f" [{f.observation[:30]}...]" if f.observation else ""
                lines.append(f"{is_cur}{indent} Turn {f.turn_index}: {f.frame_id}{act}{obs}")
        return "\n".join(lines)

    def save_dag(self) -> None:
        """Serialize trajectory DAG to JSON."""
        data = {
            "session_id": self.session_id,
            "active_branch": self.active_branch,
            "current_frame_id": self.current_frame_id,
            "branches": self.branches,
            "frames": {fid: f.to_dict() for fid, f in self.frames.items()},
        }
        try:
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save Chronos DAG: %s", exc)

    def load_dag(self) -> None:
        """Restore trajectory DAG from JSON."""
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.session_id = data.get("session_id", self.session_id)
            self.active_branch = data.get("active_branch", "main")
            self.current_frame_id = data.get("current_frame_id")
            self.branches = data.get("branches", {})
            self.frames = {
                fid: ChronosFrame.from_dict(fdict)
                for fid, fdict in data.get("frames", {}).items()
            }
        except Exception as exc:
            logger.error("Failed to load Chronos DAG: %s", exc)
