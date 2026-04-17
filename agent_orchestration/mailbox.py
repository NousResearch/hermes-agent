"""
MailboxHub — Inter-Agent Message Passing
=========================================

Ported from Claude Code's ``src/utils/mailbox.ts`` file-mailbox pattern.

Hybrid storage: in-memory queue (primary) + JSON file (persistence).
Thread-safe via ``threading.Lock``.  Cross-process safety via ``fcntl.flock()``
for gateway mode where agents run in separate processes.

Message types:
  - message:  generic text between agents
  - task_assignment:  parent assigns a new task to child
  - result:  child reports task result to parent
  - permission_request:  child asks parent for permission escalation
  - shutdown:  parent tells child to terminate

Storage path: ``~/.hermes/orchestration/mailboxes/{session_id}/{agent_id}.json``
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Message types
MSG_MESSAGE = "message"
MSG_TASK_ASSIGNMENT = "task_assignment"
MSG_RESULT = "result"
MSG_PERMISSION_REQUEST = "permission_request"
MSG_SHUTDOWN = "shutdown"


@dataclass
class AgentMessage:
    """A single message in an agent's mailbox."""
    id: str = ""
    source: str = ""         # agent_id of sender
    target: str = ""         # agent_id of recipient
    msg_type: str = MSG_MESSAGE
    content: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Mailbox:
    """Per-agent mailbox with in-memory queue + optional file persistence.

    Ported from Claude Code's Mailbox class (src/utils/mailbox.ts).
    Uses threading.Condition for efficient wait/notify instead of polling.
    """

    def __init__(self, agent_id: str, persist_path: Optional[Path] = None):
        self.agent_id = agent_id
        self._queue: List[AgentMessage] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._persist_path = persist_path
        self._revision = 0
        self._closed = False

        # Load persisted messages on startup
        if persist_path:
            self._load_from_disk()

    @property
    def length(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def send(self, msg: AgentMessage) -> None:
        """Add a message to the mailbox and wake any waiters."""
        with self._condition:
            if self._closed:
                logger.warning("Mailbox %s: send() on closed mailbox", self.agent_id)
                return
            self._revision += 1
            self._queue.append(msg)
            self._condition.notify_all()
            self._persist_to_disk()

    def poll(self, fn: Callable[[AgentMessage], bool] = None) -> Optional[AgentMessage]:
        """Non-blocking: return and remove the first matching message, or None."""
        if fn is None:
            fn = lambda _: True
        with self._lock:
            for i, msg in enumerate(self._queue):
                if fn(msg):
                    self._revision += 1
                    removed = self._queue.pop(i)
                    self._persist_to_disk()
                    return removed
        return None

    def receive(
        self,
        fn: Callable[[AgentMessage], bool] = None,
        timeout: float = 30.0,
    ) -> Optional[AgentMessage]:
        """Blocking: wait for and return a matching message.

        Returns None on timeout or if mailbox is closed.
        """
        if fn is None:
            fn = lambda _: True

        with self._condition:
            deadline = time.monotonic() + timeout
            while True:
                # Check existing queue first
                for i, msg in enumerate(self._queue):
                    if fn(msg):
                        self._revision += 1
                        removed = self._queue.pop(i)
                        self._persist_to_disk()
                        return removed

                if self._closed:
                    return None

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None

                if not self._condition.wait(timeout=remaining):
                    return None

    def receive_all(self, fn: Callable[[AgentMessage], bool] = None) -> List[AgentMessage]:
        """Non-blocking: return and remove all matching messages."""
        if fn is None:
            fn = lambda _: True
        with self._lock:
            matched = [msg for msg in self._queue if fn(msg)]
            if matched:
                self._queue = [msg for msg in self._queue if not fn(msg)]
                self._revision += len(matched)
                self._persist_to_disk()
            return matched

    def peek(self, fn: Callable[[AgentMessage], bool] = None) -> Optional[AgentMessage]:
        """Non-blocking: return (without removing) the first matching message."""
        if fn is None:
            fn = lambda _: True
        with self._lock:
            for msg in self._queue:
                if fn(msg):
                    return msg
        return None

    def close(self) -> None:
        """Mark mailbox as closed, waking any waiting threads."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _persist_to_disk(self) -> None:
        """Write current queue to JSON file (called with lock held)."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [msg.to_dict() for msg in self._queue]
            tmp_path = self._persist_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, ensure_ascii=False, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            tmp_path.replace(self._persist_path)
        except Exception as exc:
            logger.debug("Mailbox persist failed for %s: %s", self.agent_id, exc)

    def _load_from_disk(self) -> None:
        """Load persisted messages from JSON file."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            for item in data:
                msg = AgentMessage.from_dict(item)
                self._queue.append(msg)
            self._revision = len(self._queue)
        except Exception as exc:
            logger.debug("Mailbox load failed for %s: %s", self.agent_id, exc)


class MailboxHub:
    """Central hub managing per-agent mailboxes.

    Provides lookup, creation, and cleanup.  Each agent gets its own Mailbox
    instance, persisted to ``mailbox_dir/{session_id}/{agent_id}.json``.
    """

    def __init__(self, mailbox_dir: Optional[str] = None, session_id: str = ""):
        self._mailboxes: Dict[str, Mailbox] = {}
        self._lock = threading.Lock()
        self._mailbox_dir = Path(mailbox_dir) if mailbox_dir else None
        self._session_id = session_id or uuid.uuid4().hex[:8]

    def get_or_create(self, agent_id: str) -> Mailbox:
        """Get existing mailbox or create a new one for the agent."""
        with self._lock:
            if agent_id not in self._mailboxes:
                persist_path = None
                if self._mailbox_dir:
                    persist_path = (
                        self._mailbox_dir
                        / self._session_id
                        / f"{agent_id}.json"
                    )
                self._mailboxes[agent_id] = Mailbox(
                    agent_id=agent_id, persist_path=persist_path
                )
            return self._mailboxes[agent_id]

    def send_message(
        self,
        source_id: str,
        target_id: str,
        content: str,
        msg_type: str = MSG_MESSAGE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Send a message from one agent to another."""
        msg = AgentMessage(
            source=source_id,
            target=target_id,
            msg_type=msg_type,
            content=content,
            metadata=metadata or {},
        )
        target_mailbox = self.get_or_create(target_id)
        target_mailbox.send(msg)

        # Also keep a copy in sender's outbox for traceability
        return msg

    def broadcast(
        self,
        source_id: str,
        content: str,
        msg_type: str = MSG_MESSAGE,
        exclude: Optional[set] = None,
    ) -> List[str]:
        """Broadcast a message to all known mailboxes except excluded ones."""
        exclude = exclude or set()
        exclude.add(source_id)
        sent_to = []
        with self._lock:
            for agent_id in list(self._mailboxes.keys()):
                if agent_id not in exclude:
                    self.send_message(source_id, agent_id, content, msg_type)
                    sent_to.append(agent_id)
        return sent_to

    def cleanup(self, agent_id: str) -> None:
        """Remove and close an agent's mailbox."""
        with self._lock:
            mb = self._mailboxes.pop(agent_id, None)
            if mb:
                mb.close()

    def cleanup_all(self) -> None:
        """Close and remove all mailboxes."""
        with self._lock:
            for mb in self._mailboxes.values():
                mb.close()
            self._mailboxes.clear()

    def list_agents(self) -> List[str]:
        """Return list of agent IDs with mailboxes."""
        with self._lock:
            return list(self._mailboxes.keys())

    def get_pending_count(self, agent_id: str) -> int:
        """Return number of pending messages for an agent."""
        mb = self._mailboxes.get(agent_id)
        return mb.length if mb else 0


# Module-level singleton
_mailbox_hub: Optional[MailboxHub] = None


def get_mailbox_hub() -> MailboxHub:
    """Get the global MailboxHub (creates empty one if not initialized)."""
    global _mailbox_hub
    if _mailbox_hub is None:
        _mailbox_hub = MailboxHub()
    return _mailbox_hub


def set_mailbox_hub(hub: MailboxHub) -> None:
    """Set the global MailboxHub (used by on_session_start hook)."""
    global _mailbox_hub
    _mailbox_hub = hub
