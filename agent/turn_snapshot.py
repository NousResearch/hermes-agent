"""Optional per-turn message snapshot support for host adapters.

The core Hermes loop owns the live chat ``messages`` list. Hosts that need a
thread-safe view of that list during abnormal exits can install
``agent._turn_snapshot_callback``. When present, the conversation loop wraps the
turn-local list in ``TurnSnapshotMessages`` and publishes immutable snapshots
after list mutations.

This module is intentionally host-neutral: it does not know what the caller
will persist, and it is a no-op when no callback is installed.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any, Callable

logger = logging.getLogger(__name__)

SnapshotCallback = Callable[[dict[str, Any]], None]


class TurnSnapshotMessages(list):
    """List subclass that publishes a copied turn snapshot after mutations."""

    def __init__(
        self,
        messages: Iterable[Any] = (),
        *,
        callback: SnapshotCallback,
        current_turn_user_idx: int,
    ) -> None:
        super().__init__(messages)
        self._snapshot_callback = callback
        self._current_turn_user_idx = int(current_turn_user_idx)
        self._snapshot_revision = 0

    def publish_snapshot(
        self,
        *,
        source_path: str,
        complete: bool = False,
    ) -> None:
        callback = self._snapshot_callback
        if not callable(callback):
            return
        self._snapshot_revision += 1
        payload = {
            "messages": copy.deepcopy(list(self)),
            "current_turn_user_idx": self._current_turn_user_idx,
            "complete": bool(complete),
            "source_path": str(source_path or "unknown"),
            "revision": self._snapshot_revision,
        }
        try:
            callback(payload)
        except Exception:
            logger.warning("turn snapshot callback failed", exc_info=True)

    def _publish_mutation(self, source_path: str) -> None:
        self.publish_snapshot(source_path=source_path, complete=False)

    def append(self, item: Any) -> None:  # type: ignore[override]
        super().append(item)
        self._publish_mutation("messages.append")

    def extend(self, items: Iterable[Any]) -> None:  # type: ignore[override]
        super().extend(items)
        self._publish_mutation("messages.extend")

    def insert(self, index: int, item: Any) -> None:  # type: ignore[override]
        super().insert(index, item)
        self._publish_mutation("messages.insert")

    def pop(self, index: int = -1) -> Any:  # type: ignore[override]
        item = super().pop(index)
        self._publish_mutation("messages.pop")
        return item

    def remove(self, item: Any) -> None:  # type: ignore[override]
        super().remove(item)
        self._publish_mutation("messages.remove")

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self._publish_mutation("messages.clear")

    def __setitem__(self, index: Any, value: Any) -> None:
        super().__setitem__(index, value)
        self._publish_mutation("messages.setitem")

    def __delitem__(self, index: Any) -> None:
        super().__delitem__(index)
        self._publish_mutation("messages.delitem")

    def __iadd__(self, items: Iterable[Any]):
        result = super().__iadd__(items)
        self._publish_mutation("messages.iadd")
        return result


def install_turn_snapshot_messages(
    messages: list,
    *,
    callback: Any,
    current_turn_user_idx: int,
) -> list:
    """Return a snapshot-publishing messages list when a callback exists."""

    if not callable(callback):
        return messages
    wrapped = TurnSnapshotMessages(
        messages,
        callback=callback,
        current_turn_user_idx=current_turn_user_idx,
    )
    wrapped.publish_snapshot(source_path="turn_context", complete=False)
    return wrapped


def publish_turn_snapshot(
    messages: list,
    *,
    source_path: str,
    complete: bool = False,
) -> None:
    """Publish a snapshot from a wrapped messages list, otherwise no-op."""

    publisher = getattr(messages, "publish_snapshot", None)
    if callable(publisher):
        publisher(source_path=source_path, complete=complete)
