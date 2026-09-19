# coding: utf-8
# flake8: noqa: E501
"""Subagent handle registry.

Thread safety:
    All public methods on SubagentRegistry are protected by an RLock, so
    concurrent register/resolve/set_state/remove/iteration calls are safe.
"""
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

_ALLOWED_STATES = {"running", "done", "failed", "cancelled"}


@dataclass
class SubagentHandle:
    subagent_id: str
    session_id: str
    goal: str
    parent_subagent_id: Optional[str] = None
    state: str = "running"
    role: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubagentHandle):
            return NotImplemented
        return self.subagent_id == other.subagent_id and self.session_id == other.session_id

    def __repr__(self) -> str:
        return (
            f"SubagentHandle(subagent_id={self.subagent_id!r}, "
            f"session_id={self.session_id!r}, state={self.state!r}, role={self.role!r})"
        )


class SubagentRegistry:
    def __init__(self) -> None:
        self._handles: Dict[str, SubagentHandle] = {}
        self._lock = threading.RLock()

    def register(self, handle: SubagentHandle) -> None:
        if not handle.subagent_id:
            raise ValueError("subagent_id must be a non-empty string")
        with self._lock:
            if handle.subagent_id in self._handles:
                raise ValueError(
                    f"Duplicate subagent_id: {handle.subagent_id!r}"
                )
            self._handles[handle.subagent_id] = handle

    def resolve(self, subagent_id: str) -> Optional[SubagentHandle]:
        with self._lock:
            return self._handles.get(subagent_id)

    def set_state(self, subagent_id: str, state: str) -> bool:
        if state not in _ALLOWED_STATES:
            raise ValueError(
                f"Invalid state {state!r}; allowed: {sorted(_ALLOWED_STATES)}"
            )
        with self._lock:
            handle = self._handles.get(subagent_id)
            if handle is None:
                return False
            handle.state = state
            return True

    def remove(self, subagent_id: str) -> bool:
        with self._lock:
            return self._handles.pop(subagent_id, None) is not None

    def __contains__(self, subagent_id: str) -> bool:
        with self._lock:
            return subagent_id in self._handles

    def __iter__(self):
        with self._lock:
            return iter(list(self._handles.values()))


# Module-level singleton shared by the hook handlers (__init__) and the
# tool handlers (sender) so both operate on the SAME registry. The plugin
# must not create separate SubagentRegistry() instances in different
# modules — that would make hooks register handles the send/cancel tools
# can never see.
registry = SubagentRegistry()
