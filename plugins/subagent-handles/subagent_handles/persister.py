import json
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from subagent_handles.registry import SubagentHandle, SubagentRegistry


def default_persist_root() -> str:
    """Return the default disk-backed store root under HERMES_HOME.

    Scoped to the active profile and shared by the start/stop hooks and the
    cancel tool so all writers land in the same store (survive-restart parity).
    """
    try:
        from hermes_constants import get_hermes_home

        _home = str(get_hermes_home())
    except Exception:
        _home = os.environ.get(
            "HERMES_HOME",
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "hermes"),
        )
    return os.path.join(_home, "state", "subagent-handles")


@dataclass
class SessionPersister:
    root: str
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _path(self, subagent_id: str) -> str:
        return os.path.join(self.root, f"{subagent_id}.json")

    def checkpoint(self, handle: SubagentHandle) -> None:
        payload = {
            "subagent_id": handle.subagent_id,
            "session_id": handle.session_id,
            "goal": handle.goal,
            "parent_subagent_id": handle.parent_subagent_id,
            "state": handle.state,
            "role": handle.role,
        }
        with self._lock:
            os.makedirs(self.root, exist_ok=True)
            path = self._path(handle.subagent_id)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, path)

    def load(self, subagent_id: str) -> Optional[SubagentHandle]:
        path = self._path(subagent_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return None
            if payload.get("subagent_id") != subagent_id:
                return None
            return SubagentHandle(
                subagent_id=str(payload.get("subagent_id", subagent_id)),
                session_id=str(payload.get("session_id", "")),
                goal=str(payload.get("goal", "")),
                parent_subagent_id=payload.get("parent_subagent_id"),
                state=str(payload.get("state", "running")),
                role=str(payload.get("role", "")),
            )
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def remove(self, subagent_id: str) -> bool:
        path = self._path(subagent_id)
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
            return True
        except Exception:
            return False

    def restore(self, registry: SubagentRegistry) -> Dict[str, SubagentHandle]:
        restored: Dict[str, SubagentHandle] = {}
        if not os.path.isdir(self.root):
            return restored
        for filename in os.listdir(self.root):
            if not filename.endswith(".json"):
                continue
            subagent_id = filename[: -len(".json")]
            handle = self.load(subagent_id)
            if handle is None:
                continue
            # A handle persisted as "running" belongs to a process that has
            # since exited — children are subprocesses of the owner. Reconcile
            # it to "failed" so subagent_send does not report queued to a dead
            # child after a restart.
            if handle.state == "running":
                handle.state = "failed"
            try:
                registry.register(handle)
                restored[subagent_id] = handle
            except ValueError:
                pass
        return restored
