"""Shared primitives for snapshot-bound queued-turn management.

The runtime that owns a pending FIFO remains authoritative for mutation.  This
module only provides two narrow building blocks shared by the gateway, classic
CLI, and TUI:

* :class:`QueueSnapshotStore` freezes opaque queue IDs behind short-lived,
  one-based positions.  Opaque IDs never need to be rendered to users.
* :class:`ManagedPromptQueue` preserves ``queue.Queue``'s public behavior while
  attaching stable IDs to queued user turns for the classic CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import secrets
import time
from typing import Any, Callable, Iterable, Sequence


DEFAULT_QUEUE_SNAPSHOT_TTL_SECONDS = 300.0
DEFAULT_QUEUE_PREVIEW_CHARS = 160


def new_queue_id() -> str:
    """Return a process-local opaque identifier for one pending turn."""

    return f"q-{secrets.token_hex(8)}"


def _preview_payload(value: Any, *, limit: int = DEFAULT_QUEUE_PREVIEW_CHARS) -> tuple[str, bool]:
    """Build a bounded single-line preview without exposing attachment paths."""

    has_media = False
    text = value
    if isinstance(value, tuple) and len(value) == 2:
        text, attachments = value
        has_media = bool(attachments)
    rendered = " ".join(str(text or "").split()).replace("@", "＠")
    if not rendered:
        rendered = "[media]" if has_media else "[empty]"
    if len(rendered) > limit:
        rendered = rendered[: max(0, limit - 1)].rstrip() + "…"
    return rendered, has_media


@dataclass(frozen=True)
class QueueDisplayItem:
    """Safe projection of one user-owned pending turn."""

    queue_id: str
    preview: str
    has_media: bool = False
    created_at: Any = None


@dataclass(frozen=True)
class QueueSnapshot:
    """One immutable mapping from displayed positions to opaque queue IDs."""

    snapshot_id: str
    control_key: str
    owner_key: str
    target_key: str
    items: tuple[QueueDisplayItem, ...]
    created_at: float
    expires_at: float
    source_label: str = ""


@dataclass(frozen=True)
class QueueSelection:
    """Result of resolving a text selector against an active snapshot."""

    status: str
    queue_ids: tuple[str, ...] = ()
    target_key: str | None = None
    snapshot_id: str | None = None


class QueueSnapshotStore:
    """Keep one active queue snapshot per operator and control conversation."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_QUEUE_SNAPSHOT_TTL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock
        self._active: dict[tuple[str, str], QueueSnapshot] = {}

    @staticmethod
    def _key(control_key: Any, owner_key: Any) -> tuple[str, str] | None:
        control = str(control_key or "").strip()
        owner = str(owner_key or "").strip()
        if not control or not owner:
            return None
        return control, owner

    @staticmethod
    def _coerce_items(items: Iterable[Any]) -> tuple[QueueDisplayItem, ...]:
        projected: list[QueueDisplayItem] = []
        seen: set[str] = set()
        for raw in items or ():
            if isinstance(raw, QueueDisplayItem):
                item = raw
            elif isinstance(raw, dict):
                queue_id = str(raw.get("id") or raw.get("queue_id") or "").strip()
                preview = str(raw.get("preview") or "")
                item = QueueDisplayItem(
                    queue_id=queue_id,
                    preview=preview,
                    has_media=bool(raw.get("has_media")),
                    created_at=raw.get("created_at"),
                )
            else:
                continue
            if not item.queue_id or item.queue_id in seen:
                continue
            seen.add(item.queue_id)
            safe_preview, _ = _preview_payload(item.preview)
            projected.append(
                QueueDisplayItem(
                    queue_id=item.queue_id,
                    preview=safe_preview,
                    has_media=item.has_media,
                    created_at=item.created_at,
                )
            )
        return tuple(projected)

    def open(
        self,
        control_key: Any,
        owner_key: Any,
        target_key: Any,
        items: Iterable[Any],
        source_label: str = "",
    ) -> QueueSnapshot:
        """Create and activate a fresh snapshot, superseding the previous one."""

        key = self._key(control_key, owner_key)
        if key is None:
            raise ValueError("queue snapshots require control and owner keys")
        target = str(target_key or "").strip()
        if not target:
            raise ValueError("queue snapshots require a target key")
        now = float(self._clock())
        snapshot = QueueSnapshot(
            snapshot_id=f"qs-{secrets.token_hex(8)}",
            control_key=key[0],
            owner_key=key[1],
            target_key=target,
            items=self._coerce_items(items),
            created_at=now,
            expires_at=now + self.ttl_seconds,
            source_label=" ".join(str(source_label or "").split()),
        )
        self._active[key] = snapshot
        return snapshot

    def get(self, control_key: Any, owner_key: Any) -> QueueSnapshot | None:
        """Return the active unexpired snapshot, if any."""

        key = self._key(control_key, owner_key)
        if key is None:
            return None
        snapshot = self._active.get(key)
        if snapshot is None:
            return None
        if float(self._clock()) > snapshot.expires_at:
            return None
        return snapshot

    def invalidate(self, control_key: Any, owner_key: Any) -> None:
        key = self._key(control_key, owner_key)
        if key is not None:
            self._active.pop(key, None)

    def invalidate_target(self, target_key: Any) -> None:
        """Discard every active view that addresses one runtime queue."""

        target = str(target_key or "").strip()
        if not target:
            return
        for key, snapshot in tuple(self._active.items()):
            if snapshot.target_key == target:
                self._active.pop(key, None)

    def resolve(
        self,
        control_key: Any,
        owner_key: Any,
        selector: Any,
        *,
        snapshot_id: str | None = None,
    ) -> QueueSelection:
        """Resolve ``N`` or ``all`` only against the active frozen snapshot."""

        key = self._key(control_key, owner_key)
        if key is None:
            return QueueSelection(status="missing")
        snapshot = self._active.get(key)
        if snapshot is None:
            return QueueSelection(status="missing")
        if snapshot_id is not None and str(snapshot_id) != snapshot.snapshot_id:
            return QueueSelection(status="superseded")
        if float(self._clock()) > snapshot.expires_at:
            return QueueSelection(
                status="expired",
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )

        normalized = str(selector or "").strip().lower()
        if normalized == "all":
            if not snapshot.items:
                return QueueSelection(
                    status="empty",
                    target_key=snapshot.target_key,
                    snapshot_id=snapshot.snapshot_id,
                )
            return QueueSelection(
                status="ok",
                queue_ids=tuple(item.queue_id for item in snapshot.items),
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )
        if not normalized.isdigit() or normalized.startswith("0"):
            return QueueSelection(
                status="invalid_selector",
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )
        index = int(normalized)
        if index <= 0:
            return QueueSelection(
                status="invalid_selector",
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )
        if not snapshot.items:
            return QueueSelection(
                status="empty",
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )
        if index > len(snapshot.items):
            return QueueSelection(
                status="out_of_range",
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )
        return QueueSelection(
            status="ok",
            queue_ids=(snapshot.items[index - 1].queue_id,),
            target_key=snapshot.target_key,
            snapshot_id=snapshot.snapshot_id,
        )

    def resolve_ids(
        self,
        control_key: Any,
        owner_key: Any,
        queue_ids: Iterable[Any],
        *,
        snapshot_id: str | None,
    ) -> QueueSelection:
        """Validate opaque IDs against one explicitly named active snapshot.

        Rich controls carry opaque IDs internally, unlike text commands which
        carry a displayed position.  They still must prove that every ID came
        from the current short-lived view; otherwise an old card could mutate a
        newer queue after another view replaced it.
        """

        key = self._key(control_key, owner_key)
        if key is None:
            return QueueSelection(status="missing")
        snapshot = self._active.get(key)
        if snapshot is None:
            return QueueSelection(status="missing")
        if not snapshot_id or str(snapshot_id) != snapshot.snapshot_id:
            return QueueSelection(status="superseded")
        if float(self._clock()) > snapshot.expires_at:
            return QueueSelection(
                status="expired",
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )

        requested: list[str] = []
        seen: set[str] = set()
        for raw_queue_id in queue_ids or ():
            queue_id = str(raw_queue_id or "").strip()
            if not queue_id or queue_id in seen:
                continue
            seen.add(queue_id)
            requested.append(queue_id)
        allowed = {item.queue_id for item in snapshot.items}
        if not requested:
            return QueueSelection(
                status="empty" if not snapshot.items else "out_of_range",
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )
        if any(queue_id not in allowed for queue_id in requested):
            return QueueSelection(
                status="out_of_range",
                target_key=snapshot.target_key,
                snapshot_id=snapshot.snapshot_id,
            )
        return QueueSelection(
            status="ok",
            queue_ids=tuple(requested),
            target_key=snapshot.target_key,
            snapshot_id=snapshot.snapshot_id,
        )


def format_queue_snapshot(snapshot: QueueSnapshot) -> str:
    """Render a platform-neutral text manager without opaque identifiers."""

    title = "Queued turns"
    if snapshot.source_label:
        title = f"{title} — {snapshot.source_label}"
    if not snapshot.items:
        return (
            f"{title}\n\n"
            "No queued turns.\n\n"
            "Run `/queue` again to refresh this view. This view expires in 5 minutes."
        )

    lines = [title, ""]
    for index, item in enumerate(snapshot.items, start=1):
        media = " 📎" if item.has_media else ""
        lines.append(f"{index}. {item.preview}{media}")
    example = min(2, len(snapshot.items))
    lines.extend(
        (
            "",
            f"Remove one: `/dequeue {example}` (or `/dq {example}`)",
            "Remove all shown: `/dequeue all`",
            "Numbers belong to this view for 5 minutes. Run `/queue` to refresh.",
        )
    )
    return "\n".join(lines)


@dataclass
class _ManagedEntry:
    payload: Any
    queue_id: str
    manageable: bool
    created_at: float


class ManagedPromptQueue(queue.Queue):
    """``queue.Queue`` compatible FIFO with stable IDs for user turns."""

    def _entry(self, item: Any, *, manageable: bool) -> _ManagedEntry:
        return _ManagedEntry(
            payload=item,
            queue_id=new_queue_id(),
            manageable=manageable,
            created_at=time.time(),
        )

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        super().put(self._entry(item, manageable=True), block=block, timeout=timeout)

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)

    def put_system(
        self, item: Any, block: bool = True, timeout: float | None = None
    ) -> None:
        super().put(self._entry(item, manageable=False), block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        entry = super().get(block=block, timeout=timeout)
        return entry.payload if isinstance(entry, _ManagedEntry) else entry

    def get_nowait(self) -> Any:
        return self.get(block=False)

    def snapshot_items(self) -> list[QueueDisplayItem]:
        with self.mutex:
            entries: Sequence[Any] = tuple(self.queue)
        items: list[QueueDisplayItem] = []
        for entry in entries:
            if not isinstance(entry, _ManagedEntry) or not entry.manageable:
                continue
            preview, has_media = _preview_payload(entry.payload)
            items.append(
                QueueDisplayItem(
                    queue_id=entry.queue_id,
                    preview=preview,
                    has_media=has_media,
                    created_at=entry.created_at,
                )
            )
        return items

    def snapshot_payloads(self) -> list[Any]:
        """Return payloads in FIFO order without consuming the queue."""
        with self.mutex:
            entries: Sequence[Any] = tuple(self.queue)
        return [
            entry.payload if isinstance(entry, _ManagedEntry) else entry
            for entry in entries
        ]

    def remove_ids(self, queue_ids: Iterable[Any]) -> int:
        wanted = {str(value).strip() for value in queue_ids if str(value).strip()}
        if not wanted:
            return 0
        with self.not_full:
            kept = []
            removed = 0
            for entry in self.queue:
                if (
                    isinstance(entry, _ManagedEntry)
                    and entry.manageable
                    and entry.queue_id in wanted
                ):
                    removed += 1
                    continue
                kept.append(entry)
            if removed:
                self.queue.clear()
                self.queue.extend(kept)
                self.unfinished_tasks = max(0, self.unfinished_tasks - removed)
                if self.unfinished_tasks == 0:
                    self.all_tasks_done.notify_all()
                self.not_full.notify_all()
            return removed
