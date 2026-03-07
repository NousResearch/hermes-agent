from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from datetime import datetime
from queue import Empty, Queue
from typing import Callable, Optional

from swarm.types import MessageType, SwarmMessage


class MessageBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inboxes: dict[str, Queue[SwarmMessage]] = defaultdict(Queue)
        self._history: list[SwarmMessage] = []
        self._subscribers: dict[str, dict[str, list[Callable[[SwarmMessage], None]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._agents: set[str] = set()

    def _ensure_agent(self, agent_id: str) -> None:
        with self._lock:
            self._agents.add(agent_id)
            # access to create the queue if missing
            _ = self._inboxes[agent_id]

    def send(self, msg: SwarmMessage) -> None:
        with self._lock:
            self._history.append(msg)
            if msg.to_agent == "broadcast":
                for aid in list(self._agents):
                    if aid != msg.from_agent:
                        self._inboxes[aid].put(msg)
                        self._fire_subscribers(aid, msg)
            else:
                self._agents.add(msg.to_agent)
                self._inboxes[msg.to_agent].put(msg)
                self._fire_subscribers(msg.to_agent, msg)

    def broadcast(self, from_agent: str, msg_type: str, payload: dict) -> SwarmMessage:
        msg = SwarmMessage(
            from_agent=from_agent,
            to_agent="broadcast",
            msg_type=MessageType(msg_type),
            payload=payload,
            id=uuid.uuid4().hex[:12],
            timestamp=datetime.utcnow(),
        )
        self.send(msg)
        return msg

    def receive(
        self, agent_id: str, block: bool = True, timeout: float | None = None
    ) -> Optional[SwarmMessage]:
        self._ensure_agent(agent_id)
        try:
            return self._inboxes[agent_id].get(block=block, timeout=timeout)
        except Empty:
            return None

    def peek(self, agent_id: str) -> list[SwarmMessage]:
        self._ensure_agent(agent_id)
        with self._lock:
            q = self._inboxes[agent_id]
            return list(q.queue)

    def subscribe(
        self, agent_id: str, msg_type: str, callback: Callable[[SwarmMessage], None]
    ) -> None:
        self._ensure_agent(agent_id)
        with self._lock:
            self._subscribers[agent_id][msg_type].append(callback)

    def get_history(self, agent_id: str | None = None) -> list[SwarmMessage]:
        with self._lock:
            if agent_id is None:
                return list(self._history)
            return [
                m
                for m in self._history
                if m.from_agent == agent_id
                or m.to_agent == agent_id
                or m.to_agent == "broadcast"
            ]

    def clear(self, agent_id: str) -> None:
        self._ensure_agent(agent_id)
        with self._lock:
            q = self._inboxes[agent_id]
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break

    def _fire_subscribers(self, agent_id: str, msg: SwarmMessage) -> None:
        # Called while holding _lock — callbacks must be fast / non-blocking.
        type_key = msg.msg_type.value
        for cb in self._subscribers.get(agent_id, {}).get(type_key, []):
            cb(msg)
