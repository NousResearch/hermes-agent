"""P1 event aggregation buffer for kanban notifications (M2-aggregation).

Implements Layer 2.5 of the notification policy described in
``DESIGN.md`` §3 (aggregation strategy) and §6.1 Phase 2 AGGREGATE.

Layer 1 (``classify_event_severity`` in ``gateway.kanban_watchers.py``)
assigns each terminal kanban event a base P0/P1/P2 severity.  Layer 2
(``effective_severity`` / ``should_push`` in
``gateway/notification_preferences.py``) gates the push through user
preferences.  This module is *Layer 2.5*: it sits between Layer 2 and
the delivery call and *defers* P1 events whose push still passes the
floor into an in-memory buffer that flushes on either of two windows:

* **Time window** — ``aggregation.time_window_seconds`` of silence
  since the most recent P1 event for the same pipeline (default 300s).
* **Count window** — ``aggregation.count_threshold`` P1 events
  accumulated for the same pipeline (default 3).

When either window fires, the buffer is flushed as a single summary
message via ``adapter.send`` (the same delivery path the legacy
per-event push uses), then the buffer is dropped.

Configuration (in ``~/.hermes/config.yaml`` under ``kanban:``)::

    kanban:
      aggregation:
        time_window_seconds: 300   # default
        count_threshold: 3         # default

Both keys are optional — missing values fall back to the documented
defaults without raising.  Setting ``aggregation.enabled: false``
disables buffering entirely and P1 events that pass the floor push
immediately (the M1-3 behaviour).

Buffer keying
-------------

The buffer is keyed by ``(board, root_task_id, subscription_key)`` where
``subscription_key`` is the notification destination
``(platform, chat_id, thread_id)``.  Each subscription gets its own
view of the same pipeline — two chats that both watch pipeline ``t_x``
each get their own buffer and own summary.  This matches the existing
notifier's per-subscription delivery semantics.

``root_task_id`` is the *pipeline root* (a task with no parents in
``task_links``).  When a P1 event arrives for a task whose root can't
be resolved (e.g. the task has no ``task_links`` entries), the event
is treated as its own root — i.e. a single-task pipeline.  This is the
right default: an orphan event still gets aggregated with its peers
under its own id instead of leaking into the legacy push path.

Integration point
-----------------

The notifier loop (``_kanban_notifier_watcher`` in
``kanban_watchers.py``) calls :func:`buffer_p1_event` *after*
``_filter_event_for_push`` returns ``should_push=True, severity=P1``.
When ``buffer_p1_event`` accepts the event, the caller skips the
per-event ``adapter.send`` (the buffer will produce the summary
later).  When it returns ``False``, the caller falls through to the
legacy push path — useful for tests that don't have an aggregator
instance.

A second call, :func:`maybe_flush_due_buffers`, runs on every
notifier tick (cheap: O(N buffers), pure Python) and emits any buffer
whose time window has elapsed.  The count window flushes
synchronously inside :func:`buffer_p1_event` so it always triggers
before the next event can slip through.

Pipeline-done flushes happen via :func:`flush_pipeline_on_completion`,
called from the notifier loop when the event being delivered is a
``completed`` for a pipeline root — this is the explicit "终结" trigger
from DESIGN.md §3.

Failure semantics
-----------------

* DB errors resolving the root → treat the event as its own root
  (aggregation still happens).
* DB errors walking parents → same fallback.
* Delivery errors from ``adapter.send`` → the buffer is dropped
  regardless (the per-subscription retry counter in the notifier loop
  handles backoff).  The legacy path's rewind-on-failure semantics
  are out of scope here.

The module never raises out of any public function.  Internal
exceptions are logged and degrade to "skip aggregation, push
immediately" so a bug in this module never takes down the notifier.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("gateway.run")


# ---------------------------------------------------------------------------
# Defaults — kept as module constants so tests can import the same values
# ---------------------------------------------------------------------------

DEFAULT_TIME_WINDOW_SECONDS = 300  # 5 minutes per DESIGN.md §3
DEFAULT_COUNT_THRESHOLD = 3        # 3 P1 events per pipeline per DESIGN.md §3


# ---------------------------------------------------------------------------
# Pure helpers — root resolution
# ---------------------------------------------------------------------------


def _resolve_root_task_id(
    conn: sqlite3.Connection, task_id: str,
) -> str:
    """Walk the parent chain to find the pipeline root for *task_id*.

    A pipeline root is a task with no ``task_links.parent_id`` entries
    pointing at it.  Returns *task_id* itself when the task has no
    parents (single-task pipeline), no descendants (orphan), or when
    the DB lookup fails for any reason.

    Defensive against:
      * Missing task (orphan event) → returns ``task_id``.
      * DB error → returns ``task_id`` and logs at debug.
      * Cycles in ``task_links`` (shouldn't happen per kanban_db but
        defend anyway via the seen-set) → returns the first ancestor
        whose parent chain bottoms out.
    """
    if not task_id or conn is None:
        return task_id or ""
    try:
        seen: set[str] = set()
        current = task_id
        # Cap depth so a pathological chain can't hang the notifier.
        for _ in range(64):
            if current in seen:
                # Cycle — treat the current node as its own root.
                return current
            seen.add(current)
            row = conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ? LIMIT 1",
                (current,),
            ).fetchone()
            if not row or not row["parent_id"]:
                return current
            current = str(row["parent_id"])
        return current  # depth cap hit — best-effort, return last seen.
    except Exception as exc:
        logger.debug(
            "aggregation: root resolution failed for %s: %s; using task_id",
            task_id, exc,
        )
        return task_id


def _resolve_root_title(
    conn: sqlite3.Connection, root_id: str,
) -> str:
    """Return the pipeline root's title, or ``root_id`` when missing.

    Used purely for the summary's banner line so the user can tell
    which pipeline the aggregation belongs to.
    """
    if not root_id or conn is None:
        return root_id or ""
    try:
        row = conn.execute(
            "SELECT title FROM tasks WHERE id = ?", (root_id,),
        ).fetchone()
        if row and row["title"]:
            return str(row["title"])
    except Exception as exc:
        logger.debug(
            "aggregation: root title lookup failed for %s: %s", root_id, exc,
        )
    return root_id


# ---------------------------------------------------------------------------
# AggregateBuffer
# ---------------------------------------------------------------------------


@dataclass
class AggregateBuffer:
    """In-memory buffer for P1 events of a single pipeline + subscription.

    Created on first P1 event for the key, mutated on each subsequent
    P1, dropped on flush.  Lifetime is bounded by the gateway process;
    on restart the buffer is empty by design (DESIGN.md §3 "处理
    buffer 重建/丢失场景（重启后空 buffer 正常）").

    Fields:
      * ``board`` — board slug (kept so the flush path can re-open
        the DB if it needs to enrich the summary with fresh
        per-descendant status counts).
      * ``root_task_id`` — the pipeline root the events belong to.
      * ``root_title`` — cached at buffer creation so the summary
        still has a name when the DB is locked at flush time.
      * ``subscription_key`` — the (platform, chat_id, thread_id)
        tuple this buffer is bound to.  Each subscription gets its
        own view; aggregating across subscriptions would force the
        notifier to track per-chat delivery state, which is more
        trouble than the simpler per-sub buffer is worth.
      * ``events`` — list of dicts carrying the original ``ev``
        dataclass fields plus a snapshot of the per-event payload
        (so flush works even if the caller mutates the original
        dataclass afterwards).
      * ``created_at`` — ``time.time()`` of buffer creation.  Used
        only for the summary's "(N min)" suffix, not for flush
        triggering (that's ``last_event_at``).
      * ``last_event_at`` — ``time.time()`` of the most recent
        buffer insertion; the time window measures from here.
      * ``flush_reason`` — string the formatter uses to label the
        trigger; set when :func:`take_due_buffer` returns the buffer.
    """
    board: str
    root_task_id: str
    root_title: str
    subscription_key: tuple
    events: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_event_at: float = field(default_factory=time.time)
    flush_reason: str = ""


# ---------------------------------------------------------------------------
# Public helpers — summary formatter
# ---------------------------------------------------------------------------


def _short_task_title(task_id: str, task_obj=None) -> str:
    """Best-effort short label for an event line.

    Prefers ``task_obj.title`` when available, else falls back to the
    task id.  Never raises; pure helper used by :func:`format_summary`.
    """
    try:
        if task_obj is not None:
            t = getattr(task_obj, "title", None)
            if t:
                return str(t)
    except Exception:
        pass
    return str(task_id or "")


def _event_line(ev: dict) -> str:
    """Build one "completed: <title>" line for a buffered event.

    Reads ``ev["kind"]`` and the matching human-facing verb.  Falls
    back to a generic ``<kind>: <task_id>`` line when the kind is
    unknown so a future kind addition can't take down the summary.
    """
    kind = str(ev.get("kind") or "")
    task_id = str(ev.get("task_id") or "")
    title = _short_task_title(task_id, ev.get("task_obj"))
    if kind == "completed":
        return f"completed: {title}"
    if kind == "blocked":
        # Inline the reason when present so a duplicate block
        # doesn't get bounced into the summary silently.
        reason = ""
        pl = ev.get("payload") or {}
        if isinstance(pl, dict) and pl.get("reason"):
            reason = f" ({str(pl['reason'])[:80]})"
        return f"blocked: {title}{reason}"
    if kind == "crashed":
        return f"crashed: {title}"
    if kind == "gave_up":
        return f"gave up: {title}"
    if kind == "timed_out":
        return f"timed out: {title}"
    if kind:
        return f"{kind}: {title}"
    return f"event: {task_id}"


def format_summary(buffer: AggregateBuffer) -> str:
    """Render *buffer* as the DESIGN.md §3 summary block.

    Output shape (matches the spec verbatim, box-drawing characters
    preserved):

        📋 Pipeline "<title>" 进度 (5min)
        ━━━━━━━━━━━━━━━━━━━━━━
        • completed: <t1>
        • completed: <t2>
        ━━━━━━━━━━━━━━━━━━━━━━
        详情: hermes kanban show <root_task_id>

    Falls back gracefully when the buffer is empty (defensive — the
    caller is supposed to drop empty buffers before reaching here).
    """
    title = buffer.root_title or buffer.root_task_id
    elapsed_s = max(0, int(time.time() - buffer.created_at))
    elapsed_min = max(1, elapsed_s // 60)  # floor 1 min so the suffix reads sensibly

    line = "━" * 26
    lines: list[str] = [
        f"📋 Pipeline \"{title}\" 进度 ({elapsed_min}min)",
        line,
    ]
    for ev in buffer.events:
        lines.append(f"• {_event_line(ev)}")
    if not buffer.events:
        lines.append("• (no events)")
    lines.append(line)
    lines.append(f"详情: hermes kanban show {buffer.root_task_id}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DB-query helpers for format_pipeline_summary (task body §4)
# ---------------------------------------------------------------------------

_MAX_DESCENDANTS = 200  # cap so a pathological tree can't hang the notifier

# Task statuses that map to each NR-4 status group.
_STATUS_DONE = frozenset({"done", "archived"})
_STATUS_ACTIVE = frozenset({"running", "todo", "ready", "triage", "scheduled"})
_STATUS_BLOCKED = frozenset({"blocked"})
_STATUS_FAILED = frozenset({"crashed", "gave_up", "timed_out"})


def _collect_descendant_statuses(
    conn: sqlite3.Connection, root_id: str,
) -> list[dict]:
    """BFS the ``task_links`` tree and return ``[{id, title, status}, ...]``.

    Includes the root task itself plus all descendants.  Capped at
    ``_MAX_DESCENDANTS`` entries so a pathological tree can't hang the
    notifier.  Never raises — returns whatever it collected so far on
    a DB error.
    """
    ids: list[str] = [root_id]
    seen: set[str] = {root_id}
    frontier: list[str] = [root_id]
    while frontier and len(ids) < _MAX_DESCENDANTS:
        next_frontier: list[str] = []
        for parent_id in frontier:
            try:
                rows = conn.execute(
                    "SELECT child_id FROM task_links WHERE parent_id = ?",
                    (parent_id,),
                ).fetchall()
            except Exception:
                break
            for (child_id,) in rows:
                cid = str(child_id) if child_id else ""
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                ids.append(cid)
                next_frontier.append(cid)
                if len(ids) >= _MAX_DESCENDANTS:
                    break
        frontier = next_frontier
    # Batch-query titles + statuses for everything collected.
    out: list[dict] = []
    placeholders = ",".join("?" for _ in ids)
    try:
        rows = conn.execute(
            f"SELECT id, title, status FROM tasks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        for r in rows:
            out.append({
                "id": str(r["id"]) if r["id"] else "",
                "title": str(r["title"] or "") if r["title"] else str(r["id"] or ""),
                "status": str(r["status"] or "") if r["status"] else "",
            })
    except Exception:
        pass
    return out


def format_pipeline_summary(
    buffer: AggregateBuffer,
    conn: Optional[sqlite3.Connection] = None,
) -> str:
    """Render *buffer* using DB-queried child-task status distribution.

    This is the task body §4 / NR-4 format that groups child tasks by
    their **current DB status** (not the buffered event kind), giving
    the user a complete pipeline snapshot::

        📋 Pipeline "<title>" 进度 (<span>)
        ━━━━━━━━━━━━━━━━━━━━━━
        ✅ 完成: taskA, taskB
        ⏳ 进行中: taskC
        ⏸ 阻塞: taskD(等待端口确认)
        ━━━━━━━━━━━━━━━━━━━━━━
        详情: hermes kanban show <root_task_id>

    When *conn* is ``None`` or the DB query fails, falls back to
    :func:`format_summary` (the buffered-events view) so the flush
    still produces *something* — a degraded summary is better than
    dropping the buffer silently.

    The time-span suffix follows NR-4: ``即时`` for count/pipeline
    triggers, ``<N>min`` for the time window.
    """
    if conn is None:
        return format_summary(buffer)

    title = buffer.root_title or buffer.root_task_id
    # Time span per NR-4.
    reason = buffer.flush_reason or ""
    if reason == "time_window":
        elapsed_s = max(0, int(time.time() - buffer.created_at))
        span = f"{max(1, elapsed_s // 60)}min"
    else:
        span = "即时"

    # Collect descendant statuses from the DB.
    tasks = _collect_descendant_statuses(conn, buffer.root_task_id)
    if not tasks:
        # DB empty or error — degrade to buffered-events summary.
        return format_summary(buffer)

    # Enrich: pull reason/summary snippets from buffered events for
    # blocked / crashed tasks so the inline detail survives.
    ev_by_task: dict[str, dict] = {}
    for ev in buffer.events:
        tid = str(ev.get("task_id") or "")
        if tid:
            ev_by_task.setdefault(tid, ev)

    # Group tasks into NR-4 status buckets.
    groups: dict[str, list[str]] = {}  # icon_label → list of formatted items
    for t in tasks:
        st = t["status"]
        tname = t["title"] or t["id"]
        if st in _STATUS_DONE:
            label = "✅ 完成"
        elif st in _STATUS_ACTIVE:
            label = "⏳ 进行中"
        elif st in _STATUS_BLOCKED:
            ev = ev_by_task.get(t["id"])
            reason_text = ""
            if ev:
                pl = ev.get("payload") or {}
                if isinstance(pl, dict) and pl.get("reason"):
                    reason_text = f"({str(pl['reason'])[:60]})"
            label = "⏸ 阻塞"
            tname = f"{tname}{reason_text}" if reason_text else tname
        elif st in _STATUS_FAILED:
            label = "❌ 失败"
        else:
            continue  # unknown status — skip rather than mislabel
        groups.setdefault(label, []).append(tname)

    sep = "━" * 26
    lines: list[str] = [
        f'📋 Pipeline "{title}" 进度 ({span})',
        sep,
    ]
    for label, items in groups.items():
        lines.append(f"{label}: {', '.join(items)}")

    # Cap at ≤7 lines total (NR-4 constraint).
    MAX_CONTENT = 3
    content_start = 2  # title + sep
    content_end = len(lines)  # before final sep + detail
    content = lines[content_start:content_end]
    if len(content) > MAX_CONTENT:
        kept = content[:MAX_CONTENT - 1]
        remaining_tasks = sum(
            len(items)
            for items in list(groups.values())[MAX_CONTENT - 1:]
        )
        kept.append(f"…及其他 {remaining_tasks} 个任务")
        lines = lines[:content_start] + kept
    lines.append(sep)
    lines.append(f"详情: hermes kanban show {buffer.root_task_id}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NotificationAggregator
# ---------------------------------------------------------------------------


class NotificationAggregator:
    """In-memory aggregator that buffers P1 events per pipeline+subscription.

    The notifier loop holds one of these per gateway instance (the
    mixin's ``__init__`` should set ``self.notification_aggregator``
    exactly once).  All public methods are safe to call from the
    notifier's async loop (they're pure-Python + cheap SQLite reads,
    no blocking network I/O).  The actual ``adapter.send`` is performed
    by the notifier loop after this module returns a ready-to-send
    buffer — that keeps the aggregator I/O-free and trivially
    testable.

    Concurrency: the aggregator is single-threaded by contract — the
    notifier loop is the only caller, and Python's GIL makes a
    multi-threaded caller safe as long as we don't ``await`` inside
    any method (we don't).
    """

    # Maximum age (seconds) a buffer may live without a flush trigger.
    # Prevents unbounded memory growth from pipelines that produce only
    # 1–2 P1 events and never reach the count threshold or root-done
    # flush.  Checked in ``take_due_buffers`` alongside the time window.
    # (Review N1 — buffer age cleanup.)
    DEFAULT_MAX_BUFFER_AGE = 3600  # 1 hour

    def __init__(
        self,
        *,
        time_window_seconds: float = DEFAULT_TIME_WINDOW_SECONDS,
        count_threshold: int = DEFAULT_COUNT_THRESHOLD,
        max_buffer_age: float = DEFAULT_MAX_BUFFER_AGE,
    ) -> None:
        self.time_window_seconds = float(time_window_seconds)
        self.count_threshold = int(count_threshold)
        self.max_buffer_age = float(max_buffer_age)
        # ``_buffers`` is keyed by (board, root_task_id, sub_key).  Values
        # are AggregateBuffer instances.  We don't dedupe across subs —
        # each subscription gets its own view (see module docstring).
        self._buffers: dict[tuple, AggregateBuffer] = {}
        # Lightweight counters used by tests + log scrapers to confirm
        # the buffer actually fired.  Production code never reads them.
        self.flush_counts: dict[str, int] = {
            "time_window": 0,
            "count_threshold": 0,
            "pipeline_done": 0,
            "max_age": 0,
        }

    # -- config helpers ----------------------------------------------------

    @classmethod
    def from_config(cls, kanban_cfg: dict) -> "NotificationAggregator":
        """Construct from a ``kanban_cfg`` dict (the ``config.kanban`` block).

        Honours ``kanban_cfg["aggregation"]["enabled"] = False`` by
        returning a disabled aggregator (still constructible, but all
        ``buffer_p1_event`` calls return ``False`` and ``take_due_buffer``
        returns ``None``).  Missing keys fall back to module defaults.

        Invalid numeric values (typo, wrong type) silently fall back to
        the corresponding default — a malformed ``config.yaml`` must
        never take down the notification pipeline.
        """
        agg_cfg = (kanban_cfg or {}).get("aggregation") or {}
        if not isinstance(agg_cfg, dict):
            agg_cfg = {}
        if agg_cfg.get("enabled") is False:
            # Caller explicitly disabled.  Build an instance that
            # no-ops by setting absurd thresholds.
            return cls(time_window_seconds=10**9, count_threshold=10**9)
        # ``time_window_seconds`` — default 300, must be a positive number.
        try:
            tw = float(agg_cfg.get("time_window_seconds", DEFAULT_TIME_WINDOW_SECONDS))
            if tw <= 0:
                tw = DEFAULT_TIME_WINDOW_SECONDS
        except (TypeError, ValueError):
            tw = DEFAULT_TIME_WINDOW_SECONDS
        # ``count_threshold`` — default 3, must be a positive integer.
        try:
            ct = int(agg_cfg.get("count_threshold", DEFAULT_COUNT_THRESHOLD))
            if ct <= 0:
                ct = DEFAULT_COUNT_THRESHOLD
        except (TypeError, ValueError):
            ct = DEFAULT_COUNT_THRESHOLD
        # ``max_buffer_age`` — default 3600s (review N1), must be positive.
        try:
            mba = float(agg_cfg.get("max_buffer_age_seconds", cls.DEFAULT_MAX_BUFFER_AGE))
            if mba <= 0:
                mba = cls.DEFAULT_MAX_BUFFER_AGE
        except (TypeError, ValueError):
            mba = cls.DEFAULT_MAX_BUFFER_AGE
        return cls(
            time_window_seconds=tw,
            count_threshold=ct,
            max_buffer_age=mba,
        )

    @property
    def enabled(self) -> bool:
        """Return ``True`` unless the caller disabled aggregation.

        The constructor sets thresholds to astronomical values when
        ``enabled=False``; checking those thresholds is the canonical
        way to tell.  Exposed as a property so the notifier loop can
        skip the aggregator entirely when disabled (small but real
        per-tick win on boards with thousands of events).
        """
        return (
            self.time_window_seconds < 10**8
            and self.count_threshold < 10**8
        )

    # -- key helpers -------------------------------------------------------

    @staticmethod
    def _subscription_key(sub: dict) -> tuple:
        """Return the canonical (platform, chat_id, thread_id) key.

        Falls back to the empty-string defaults the notifier uses when
        a field is missing so the buffer key is always hashable.
        """
        if not isinstance(sub, dict):
            sub = {}
        return (
            str(sub.get("platform") or ""),
            str(sub.get("chat_id") or ""),
            str(sub.get("thread_id") or ""),
        )

    # -- buffer mutation ---------------------------------------------------

    def buffer_p1_event(
        self,
        *,
        board: str,
        task_id: str,
        ev,
        task,
        sub: dict,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[AggregateBuffer]:
        """Append *ev* to the buffer for its pipeline; return a buffer to
        flush *now* (count threshold reached), or ``None`` to keep pushing
        events individually.

        The function never raises.  Any DB error during root resolution
        is logged at debug and the event is treated as its own root.

        Args:
          board: board slug (from delivery dict).
          task_id: the event's task id.
          ev: the ``kanban_db.Event`` dataclass or any duck-typed object
            with ``.kind`` and ``.task_id``.
          task: the ``kanban_db.Task`` for the event's task (may be ``None``).
          sub: the subscription dict the event is being delivered to.
          conn: optional pre-opened SQLite connection (avoids opening
            one per event in tight loops).  When ``None``, this method
            opens its own short-lived connection via ``kanban_db.connect``.

        Returns:
          An ``AggregateBuffer`` to flush immediately when the count
          threshold fired (the buffer is removed from the internal
          dict before return — flushing is the caller's job).
          ``None`` when the event was buffered but no flush is due yet
          (caller should skip the per-event push).
          ``None`` is also returned when the aggregator is disabled —
          caller falls through to the legacy push path.
        """
        if not self.enabled:
            return None
        if not board or not task_id:
            return None

        ev_dict = self._snapshot_event(ev, task)
        root_id = self._resolve_root(conn, task_id, board=board)
        if conn is not None and root_id:
            root_title = _resolve_root_title(conn, root_id)
        elif root_id:
            # No connection — open one just for the title lookup.  Best-effort;
            # fall back to ``root_id`` when the DB call fails.
            try:
                from hermes_cli import kanban_db as _kb
                title_conn = _kb.connect(board=board) if board else _kb.connect()
                try:
                    root_title = _resolve_root_title(title_conn, root_id)
                finally:
                    try:
                        title_conn.close()
                    except Exception:
                        pass
            except Exception:
                root_title = root_id
        else:
            root_title = task_id

        sub_key = self._subscription_key(sub)
        key = (str(board), str(root_id), sub_key)

        buf = self._buffers.get(key)
        if buf is None:
            buf = AggregateBuffer(
                board=str(board),
                root_task_id=str(root_id),
                root_title=str(root_title),
                subscription_key=sub_key,
            )
            self._buffers[key] = buf

        buf.events.append(ev_dict)
        buf.last_event_at = time.time()

        if len(buf.events) >= self.count_threshold:
            buf.flush_reason = "count_threshold"
            self.flush_counts["count_threshold"] += 1
            # Pop before returning so a flush failure doesn't double-deliver.
            return self._buffers.pop(key)
        return None

    def _snapshot_event(self, ev, task) -> dict:
        """Capture the fields the summary needs from *ev* into a dict.

        The summary formatter only reads ``kind``, ``task_id``,
        ``payload``, and the parent task's ``title`` — capture those
        now so the summary is decoupled from the live event object.
        We avoid keeping a reference to *ev* itself because the
        notifier loop mutates ``ev.payload`` in some code paths.
        """
        payload: Any = {}
        try:
            pl = getattr(ev, "payload", None)
            if isinstance(pl, dict):
                payload = dict(pl)
            elif isinstance(pl, str) and pl:
                # Some legacy rows store JSON as text; parse defensively.
                try:
                    payload = json.loads(pl)
                    if not isinstance(payload, dict):
                        payload = {"_raw": payload}
                except Exception:
                    payload = {"_raw": pl}
        except Exception:
            payload = {}
        return {
            "kind": str(getattr(ev, "kind", "") or ""),
            "task_id": str(getattr(ev, "task_id", "") or ""),
            "payload": payload,
            "task_obj": task,
        }

    def _resolve_root(
        self,
        conn: Optional[sqlite3.Connection],
        task_id: str,
        *,
        board: Optional[str] = None,
    ) -> str:
        """Resolve the pipeline root, opening a DB connection if needed.

        Centralises the ``conn or kanban_db.connect(board=...)`` dance so
        callers don't have to.  Never raises; returns ``task_id`` on
        failure.
        """
        if conn is not None:
            return _resolve_root_task_id(conn, task_id)
        try:
            from hermes_cli import kanban_db as _kb
        except Exception as exc:
            logger.debug(
                "aggregation: kanban_db import failed for root resolution: %s",
                exc,
            )
            return task_id
        try:
            own_conn = _kb.connect(board=board) if board else _kb.connect()
        except Exception as exc:
            logger.debug(
                "aggregation: connect failed for root resolution on %s: %s",
                board, exc,
            )
            return task_id
        try:
            return _resolve_root_task_id(own_conn, task_id)
        finally:
            try:
                own_conn.close()
            except Exception:
                pass

    # -- time-based flush --------------------------------------------------

    def take_due_buffers(self, now: Optional[float] = None) -> list[AggregateBuffer]:
        """Return + drop every buffer whose time window has elapsed.

        Called from the notifier loop on every tick (cheap: walks
        ``_buffers`` once).  Returns the buffers ready to flush in
        insertion order so older pipelines get reported first.

        Also re-resolves the root title for any buffer that's been
        around long enough that the DB has new info (best-effort;
        skips on error).
        """
        if not self.enabled:
            return []
        now_t = time.time() if now is None else float(now)
        due: list[AggregateBuffer] = []
        # Walk a snapshot of keys so the pop doesn't trip the iterator.
        for key in list(self._buffers.keys()):
            buf = self._buffers.get(key)
            if buf is None:
                continue
            if (now_t - buf.last_event_at) >= self.time_window_seconds:
                buf.flush_reason = "time_window"
                self._buffers.pop(key, None)
                due.append(buf)
            elif (now_t - buf.created_at) >= self.max_buffer_age:
                # Review N1: a buffer that never hit count_threshold
                # and whose pipeline never reached root-done would
                # live forever.  Force-flush at max_buffer_age so
                # memory stays bounded.
                buf.flush_reason = "max_age"
                self._buffers.pop(key, None)
                due.append(buf)
        if due:
            for buf in due:
                reason = getattr(buf, "flush_reason", "")
                if reason in self.flush_counts:
                    self.flush_counts[reason] += 1
                else:
                    self.flush_counts["time_window"] += 1
        return due

    # -- pipeline-done flush ----------------------------------------------

    def flush_pipeline_on_completion(
        self,
        *,
        board: str,
        root_task_id: str,
        sub: dict,
    ) -> Optional[AggregateBuffer]:
        """Force-flush the buffer for a pipeline that just hit ``done``.

        Called from the notifier loop when the event being delivered
        is a ``completed`` for a pipeline root.  Returns the buffer to
        flush, or ``None`` if no buffered events exist (caller
        proceeds with the normal M1-4 终结报告 path).
        """
        if not self.enabled:
            return None
        if not board or not root_task_id:
            return None
        sub_key = self._subscription_key(sub)
        key = (str(board), str(root_task_id), sub_key)
        buf = self._buffers.pop(key, None)
        if buf is None:
            return None
        buf.flush_reason = "pipeline_done"
        self.flush_counts["pipeline_done"] += 1
        return buf

    def flush_if_pipeline_root(
        self,
        *,
        board: str,
        task_id: str,
        sub: dict,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[AggregateBuffer]:
        """One-shot helper: if *task_id* is a pipeline root, flush its buffer.

        Called from the notifier loop right before delivering the
        per-event pipeline summary.  Combines the root check and the
        buffer pop in one call so the caller doesn't need to know the
        aggregation key layout.

        Returns the buffer to deliver *before* the per-event
        summary, or ``None`` when *task_id* is not a pipeline root
        or no events are buffered for it.  The ``conn`` is reused
        when supplied (the notifier loop already has one open per
        event in the flush path).
        """
        if not self.enabled or not board or not task_id:
            return None
        # Open (or reuse) a DB connection just for the root check.
        # Cheap when the notifier loop passes its open conn; falls
        # back to a short-lived connection otherwise.
        own_conn: Optional[sqlite3.Connection] = None
        try:
            check_conn = conn
            if check_conn is None:
                try:
                    from hermes_cli import kanban_db as _kb
                except Exception:
                    return None
                try:
                    own_conn = _kb.connect(board=board)
                    check_conn = own_conn
                except Exception:
                    return None
            try:
                from hermes_cli.kanban_db import connect as _kb_connect  # noqa: F401
            except Exception:
                pass
            # Inline root check: a pipeline root has no parents and at
            # least one child.  Mirrors ``_is_root_task`` in
            # ``kanban_watchers.py`` but lives here so the aggregator
            # owns its own key derivation.  Cheaper than the wrapper
            # because we only need the parent_count == 0 flag.
            parent_count = check_conn.execute(
                "SELECT COUNT(*) AS n FROM task_links WHERE child_id = ?",
                (task_id,),
            ).fetchone()
            if not parent_count or int(parent_count["n"]) != 0:
                return None
        except Exception as exc:
            logger.debug(
                "aggregation: pipeline-root check failed for %s on %s: %s",
                task_id, board, exc,
            )
            return None
        finally:
            if own_conn is not None:
                try:
                    own_conn.close()
                except Exception:
                    pass
        return self.flush_pipeline_on_completion(
            board=board, root_task_id=task_id, sub=sub,
        )

    # -- introspection (tests + debug) ------------------------------------

    def buffer_count(self) -> int:
        """Return the number of currently buffered pipelines.

        Tests assert on this after simulating a burst to verify that
        the buffer didn't drain prematurely (e.g. when one event
        arrived but count_threshold=3).
        """
        return len(self._buffers)

    def clear(self) -> int:
        """Drop every buffered pipeline.

        Returns the number of buffers dropped.  Used by the gateway's
        shutdown path and by tests that need a clean slate between
        cases.  Production never calls this mid-run.
        """
        n = len(self._buffers)
        self._buffers.clear()
        return n


__all__ = [
    "AggregateBuffer",
    "NotificationAggregator",
    "DEFAULT_TIME_WINDOW_SECONDS",
    "DEFAULT_COUNT_THRESHOLD",
    "format_summary",
    "format_pipeline_summary",
]
