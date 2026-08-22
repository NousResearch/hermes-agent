"""
Mailbox Async Queue for Hermes-Agent.

This module provides a thread-safe priority queue for async message delivery.
It is used for:
- Subagent result aggregation
- TTS pipeline queue (streaming deltas)

Usage:
    from agent.hermes.mailbox import Mailbox, MailboxMessage

    mailbox = Mailbox(max_size=10000)

    # Enqueue with priority (lower = higher priority)
    mailbox.enqueue("high priority message", priority=1)
    mailbox.enqueue("low priority message", priority=10)

    # Dequeue blocks until message available
    msg = mailbox.dequeue()
    print(f"Got: {msg.content}, priority={msg.priority}")
"""

import heapq
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MailboxMessage:
    """
    Represents a message in the mailbox priority queue.

    Attributes:
        priority: Priority value (lower = higher priority)
        content: The message content
        timestamp: When the message was enqueued (UTC)
        id: Unique message identifier
    """
    priority: int  # lower = higher priority
    content: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: str = ""

    def __lt__(self, other: "MailboxMessage") -> bool:
        """Compare messages by priority for heapq."""
        if self.priority != other.priority:
            return self.priority < other.priority
        # If priorities equal, use timestamp for FIFO ordering
        return self.timestamp < other.timestamp


class Mailbox:
    """
    Thread-safe priority queue for async message delivery.

    Messages are delivered in priority order (lower priority number = delivered first).
    When priorities are equal, FIFO ordering based on timestamp is used.

    Thread Safety:
        Uses threading.Condition for wait/notify semantics on enqueue/dequeue.
        Internal heapq is protected by the condition lock.

    Memory Bounds:
        When max_size is reached, enqueue() returns False and the message
        is not added. This prevents unbounded memory growth.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the Mailbox.

        Args:
            max_size: Maximum number of messages (default 10000).
                     enqueue() returns False when full.
        """
        self._heap: List[MailboxMessage] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._max_size = max_size
        self._counter = 0
        self._closed = False

    def enqueue(self, content: Any, priority: int = 0) -> bool:
        """
        Enqueue a message with the given priority.

        Args:
            content: The message content
            priority: Priority value (lower = higher priority, default 0)

        Returns:
            True if enqueued successfully, False if max_size exceeded

        Thread Safe:
            Yes - can be called from multiple threads concurrently.
        """
        with self._not_empty:
            if self._closed:
                logger.warning("Mailbox is closed, cannot enqueue")
                return False

            if len(self._heap) >= self._max_size:
                logger.warning(f"Mailbox full (max_size={self._max_size}), cannot enqueue")
                return False

            self._counter += 1
            msg = MailboxMessage(
                priority=priority,
                content=content,
                id=f"{self._counter}-{datetime.utcnow().timestamp()}"
            )
            heapq.heappush(self._heap, msg)
            self._not_empty.notify()
            logger.debug(f"Enqueued message: priority={priority}, id={msg.id}")
            return True

    def dequeue(self, timeout: Optional[float] = None) -> Optional[MailboxMessage]:
        """
        Dequeue the highest-priority message (lowest priority number).

        Blocks if the queue is empty until a message is available or timeout expires.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            MailboxMessage if available within timeout, None if timeout expired

        Thread Safe:
            Yes - uses condition variable for blocking wait.
        """
        with self._not_empty:
            while not self._heap:
                if self._closed:
                    return None
                if timeout is None:
                    self._not_empty.wait()
                else:
                    if not self._not_empty.wait(timeout):
                        return None

            msg = heapq.heappop(self._heap)
            logger.debug(f"Dequeued message: priority={msg.priority}, id={msg.id}")
            return msg

    def size(self) -> int:
        """
        Get the current number of messages in the queue.

        Returns:
            Number of messages currently queued
        """
        with self._lock:
            return len(self._heap)

    def close(self) -> None:
        """
        Close the mailbox, waking all waiters with None.

        After close(), dequeue() returns None and enqueue() returns False.
        """
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()
            logger.info("Mailbox closed")

    def clear(self) -> None:
        """
        Clear all messages from the queue.
        """
        with self._lock:
            self._heap.clear()
            logger.debug("Mailbox cleared")

    @property
    def is_full(self) -> bool:
        """Return True if the mailbox is at max capacity."""
        with self._lock:
            return len(self._heap) >= self._max_size

    @property
    def is_empty(self) -> bool:
        """Return True if the mailbox is empty."""
        with self._lock:
            return len(self._heap) == 0

    @property
    def max_size(self) -> int:
        """Return the maximum size of the mailbox."""
        return self._max_size
