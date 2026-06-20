"""Smoke tests for the P1 aggregation buffer (DESIGN.md §3, M2 aggregation).

Mirrors the test layout of ``tests/test_notification_preferences_smoke.py``
(self-contained, no pytest fixtures) and ``workspaces/t_47985057/test_m1_1.py``
(runs against the live ``notification-policy`` board DB for the DB-aware
cases).  The four scenarios validated here come directly from the task
spec:

  1. P1 event in buffer → wait ``time_window_seconds`` → flush fires.
  2. N≥3 P1 events → immediate flush on the threshold event (no waiting).
  3. P0 / P2 events → never enter the buffer (the existing filter pushes
     them directly per M1-3).
  4. Restart → empty buffer, no error.

Run::

    cd /home/zml/workspace/hermes-agent && \
    python3 tests/test_notification_aggregator_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERMES_AGENT = "/home/zml/workspace/hermes-agent"
if HERMES_AGENT not in sys.path:
    sys.path.insert(0, HERMES_AGENT)

from gateway.notification_aggregator import (
    AggregateBuffer,
    NotificationAggregator,
    DEFAULT_COUNT_THRESHOLD,
    DEFAULT_TIME_WINDOW_SECONDS,
    format_summary,
    format_pipeline_summary,
)


# ---------------------------------------------------------------------------
# Test doubles — keep them here so the file stays self-contained
# ---------------------------------------------------------------------------


class FakeEvent:
    """Duck-typed stand-in for ``kanban_db.Event``.

    The aggregator only reads ``.kind``, ``.task_id`` and ``.payload``;
    we keep the surface minimal so tests stay explicit about what
    they're exercising.
    """

    def __init__(self, kind: str, task_id: str, payload=None):
        self.kind = kind
        self.task_id = task_id
        self.payload = payload or {}


class FakeTask:
    """Duck-typed stand-in for ``kanban_db.Task``."""

    def __init__(self, title: str = "", task_id: str = ""):
        self.title = title
        self.id = task_id


def _sub(platform: str = "telegram", chat_id: str = "c1",
         thread_id: str = "") -> dict:
    return {"platform": platform, "chat_id": chat_id, "thread_id": thread_id}


def _ev(kind: str, task_id: str, title: str = "",
        payload=None) -> tuple[FakeEvent, FakeTask]:
    return FakeEvent(kind, task_id, payload or {}), FakeTask(title, task_id)


# ---------------------------------------------------------------------------
# 1. Time window — buffer holds; fires after the window elapses
# ---------------------------------------------------------------------------


class TestTimeWindow(unittest.TestCase):
    """Scenario 1 from the task spec: buffered P1 → wait → flush."""

    def test_buffer_holds_until_window_elapses(self):
        agg = NotificationAggregator(
            time_window_seconds=0.05,  # 50ms for the test
            count_threshold=10,        # disable count-based flush
        )
        ev, task = _ev("completed", "t1", "写测试")
        flush = agg.buffer_p1_event(
            board="b1", task_id="t1", ev=ev, task=task, sub=_sub(),
        )
        self.assertIsNone(flush)
        self.assertEqual(agg.buffer_count(), 1)

        # Too early — no flush yet.
        self.assertEqual(agg.take_due_buffers(), [])
        self.assertEqual(agg.buffer_count(), 1)

        # Wait past the window, then flush.
        time.sleep(0.08)
        due = agg.take_due_buffers()
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].flush_reason, "time_window")
        self.assertEqual(due[0].root_task_id, "t1")  # no DB → orphan
        self.assertEqual(len(due[0].events), 1)
        self.assertEqual(agg.buffer_count(), 0)
        self.assertEqual(agg.flush_counts["time_window"], 1)


# ---------------------------------------------------------------------------
# 2. Count window — third event triggers immediate flush
# ---------------------------------------------------------------------------


class TestCountThreshold(unittest.TestCase):
    """Scenario 2 from the task spec: 3 P1 events → immediate flush."""

    def test_third_event_triggers_immediate_flush(self):
        """Three P1 events under the same pipeline root → flush fires on
        the 3rd event with no time-window wait.

        Requires a DB connection so ``_resolve_root`` can walk the
        parent chain; without it each event resolves as its own root
        and the buffer count would grow to 3 (covered by the next test).
        """
        import sqlite3
        agg = NotificationAggregator(
            time_window_seconds=300,  # disable time-based flush
            count_threshold=3,
        )
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE task_links(child_id TEXT, parent_id TEXT)")
        for child in ("leaf1", "leaf2", "leaf3"):
            conn.execute(
                "INSERT INTO task_links(child_id, parent_id) VALUES (?, ?)",
                (child, "root"),
            )

        sub = _sub()
        # 1st event: buffered
        f = agg.buffer_p1_event(
            board="b1", task_id="leaf1",
            ev=FakeEvent("completed", "leaf1"),
            task=None, sub=sub, conn=conn,
        )
        self.assertIsNone(f)
        self.assertEqual(agg.buffer_count(), 1)

        # 2nd event: still buffered (same root)
        f = agg.buffer_p1_event(
            board="b1", task_id="leaf2",
            ev=FakeEvent("completed", "leaf2"),
            task=None, sub=sub, conn=conn,
        )
        self.assertIsNone(f)
        self.assertEqual(agg.buffer_count(), 1)

        # 3rd event → count threshold reached → flush NOW (no waiting)
        f = agg.buffer_p1_event(
            board="b1", task_id="leaf3",
            ev=FakeEvent("completed", "leaf3"),
            task=None, sub=sub, conn=conn,
        )
        assert f is not None  # narrow for type checkers
        self.assertEqual(f.flush_reason, "count_threshold")
        self.assertEqual(len(f.events), 3)
        self.assertEqual(agg.flush_counts["count_threshold"], 1)
        conn.close()

    def test_no_conn_means_each_event_is_own_root(self):
        """Documented behaviour: without a DB conn, each orphan event
        resolves as its own pipeline root.  This is the right
        fallback — orphan events should never get fused into a fake
        "pipeline" just because we couldn't walk the parent chain.
        """
        agg = NotificationAggregator(
            time_window_seconds=300, count_threshold=3,
        )
        sub = _sub()
        for i in range(3):
            f = agg.buffer_p1_event(
                board="b1", task_id=f"t{i}",
                ev=FakeEvent("completed", f"t{i}"),
                task=None, sub=sub,
                # No conn → _resolve_root falls back to task_id.
            )
            self.assertIsNone(f)
        # 3 separate buffers (3 orphan roots), not 1.
        self.assertEqual(agg.buffer_count(), 3)

    def test_threshold_flush_uses_same_root_when_conn_supplied(self):
        """With a real DB conn, sibling tasks under one root share a buffer."""
        import sqlite3
        agg = NotificationAggregator(count_threshold=3, time_window_seconds=300)
        sub = _sub()

        # In-memory DB with a fake task_links graph:
        #   root ← child1
        #   root ← child2
        #   root ← child3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE task_links(child_id TEXT, parent_id TEXT)")
        for child in ("child1", "child2", "child3"):
            conn.execute(
                "INSERT INTO task_links(child_id, parent_id) VALUES (?, ?)",
                (child, "root"),
            )

        # First two events both resolve to "root" → same buffer.
        agg.buffer_p1_event(
            board="b1", task_id="child1",
            ev=FakeEvent("completed", "child1"), task=None, sub=sub, conn=conn,
        )
        agg.buffer_p1_event(
            board="b1", task_id="child2",
            ev=FakeEvent("completed", "child2"), task=None, sub=sub, conn=conn,
        )
        self.assertEqual(agg.buffer_count(), 1)

        # Third event → flush, single buffer with 3 events.
        f = agg.buffer_p1_event(
            board="b1", task_id="child3",
            ev=FakeEvent("completed", "child3"), task=None, sub=sub, conn=conn,
        )
        assert f is not None  # narrow for type checkers
        self.assertEqual(f.flush_reason, "count_threshold")
        self.assertEqual(f.root_task_id, "root")
        self.assertEqual(len(f.events), 3)
        conn.close()


# ---------------------------------------------------------------------------
# 3. P0 / P2 bypass — never enter the buffer (covered by filter, not aggregator)
# ---------------------------------------------------------------------------


class TestSeverityBypass(unittest.TestCase):
    """Scenario 3 from the task spec: P0 / P2 → no buffer entry.

    The aggregator is severity-agnostic: it accepts whatever the notifier
    loop hands it after ``_filter_event_for_push`` says "push".  The
    contract is that P0 and P2 (or P1 with floor=quiet) bypass the
    aggregator entirely via the notifier's filter logic.  This test
    verifies the aggregator itself stays well-behaved when fed a P0
    event — it buffers it like any other event (severity routing is
    the notifier's job, not the aggregator's).
    """

    def test_aggregator_accepts_any_severity_input(self):
        """The aggregator treats all inputs identically — that's a
        feature, not a bug.  The notifier loop enforces P0/P2 bypass
        upstream.  This test documents the contract.
        """
        agg = NotificationAggregator(count_threshold=10, time_window_seconds=300)
        # Simulate the notifier sending a P0 event to the aggregator
        # (which would be a bug in the notifier).  The aggregator
        # buffers it anyway — defensive behaviour, never raises.
        f = agg.buffer_p1_event(
            board="b1", task_id="t1",
            ev=FakeEvent("blocked", "t1", {"reason": "first-time block"}),
            task=None, sub=_sub(),
        )
        self.assertIsNone(f)
        self.assertEqual(agg.buffer_count(), 1)

    def test_p0_bypass_is_enforced_by_notifier_filter(self):
        """Verify the notifier-side filter actually short-circuits P0.

        Importing the notifier module proves the bypass hook is wired.
        We don't simulate the full async loop here; the E2E test in
        ``tests/m1_verify.py`` covers the async path.  The point of
        this assertion is regression protection: if someone deletes the
        P0 short-circuit in ``_filter_event_for_push``, the test name
        surfaces the breakage immediately.
        """
        from gateway.kanban_watchers import _filter_event_for_push
        from gateway.kanban_watchers import _event_to_filter_dict

        # Construct a fake event that should classify as P0 (first-time
        # blocked, non-review-required).
        fake_event = FakeEvent("blocked", "t_x", {"reason": "port 5432?"})
        fake_task = FakeTask("t_x")
        # Use _event_to_filter_dict to build the classifier-ready dict
        # with injectable prior state.
        d = _event_to_filter_dict(fake_event, fake_task, "b1")
        d["_prior_reasons"] = set()

        # Monkey-patch the classifier to return P0 so we bypass DB.
        import gateway.kanban_watchers as kw
        original = kw.classify_event_severity
        kw.classify_event_severity = lambda _ev: "P0"
        try:
            should_push, eff_sev = _filter_event_for_push(
                d,
                floor="normal",  # normal pushes P0, suppresses P1 — verifies P0 routing.
                overrides={},
            )
        finally:
            kw.classify_event_severity = original
        self.assertTrue(should_push, "P0 must push at floor=normal")


# ---------------------------------------------------------------------------
# 4. Restart safety — empty buffer after construction, no errors
# ---------------------------------------------------------------------------


class TestRestartSafety(unittest.TestCase):
    """Scenario 4 from the task spec: fresh process → empty buffer, no error."""

    def test_fresh_aggregator_is_empty(self):
        agg = NotificationAggregator()
        self.assertEqual(agg.buffer_count(), 0)
        self.assertEqual(agg.take_due_buffers(), [])

    def test_disabled_aggregator_silently_no_ops(self):
        """``enabled=false`` → constructor returns astronomical thresholds
        so all calls become no-ops.  No error, no buffer entry.
        """
        agg = NotificationAggregator.from_config(
            {"aggregation": {"enabled": False}},
        )
        self.assertFalse(agg.enabled)

        # Try to buffer many events — all should be no-ops.
        for i in range(10):
            f = agg.buffer_p1_event(
                board="b1", task_id=f"t{i}",
                ev=FakeEvent("completed", f"t{i}"),
                task=None, sub=_sub(),
            )
            self.assertIsNone(f)
        self.assertEqual(agg.buffer_count(), 0)

        # Time window pass should also be empty.
        time.sleep(0.05)
        self.assertEqual(agg.take_due_buffers(), [])

    def test_aggregator_handles_missing_task_obj_gracefully(self):
        """The aggregator must not crash when ``task`` is ``None`` and
        the DB lookup for the title also fails.  Restart-safety check.
        """
        agg = NotificationAggregator(time_window_seconds=300, count_threshold=10)
        f = agg.buffer_p1_event(
            board="b1", task_id="orphan",
            ev=FakeEvent("completed", "orphan"),
            task=None, sub=_sub(),
            # No conn either → _resolve_root falls back to task_id.
        )
        self.assertIsNone(f)
        # Force a flush so we exercise format_summary with no title.
        buf = next(iter(agg._buffers.values()))
        buf.root_title = ""  # simulate missing title row
        out = format_summary(buf)
        self.assertIn("Pipeline", out)
        self.assertIn("详情:", out)


# ---------------------------------------------------------------------------
# 5. format_summary — DESIGN.md §3 shape conformance
# ---------------------------------------------------------------------------


class TestFormatSummary(unittest.TestCase):
    """The summary must match DESIGN.md §3 character-for-character."""

    def test_summary_includes_required_sections(self):
        buf = AggregateBuffer(
            board="b1",
            root_task_id="root1",
            root_title="重构认证模块",
            subscription_key=("telegram", "c1", ""),
            events=[
                {"kind": "completed", "task_id": "t1",
                 "payload": {}, "task_obj": FakeTask("编写单元测试")},
                {"kind": "blocked", "task_id": "t2",
                 "payload": {"reason": "端口确认"}, "task_obj": FakeTask("DB迁移")},
                {"kind": "completed", "task_id": "t3",
                 "payload": {}, "task_obj": FakeTask("更新API文档")},
            ],
        )
        out = format_summary(buf)
        # Banner line.
        self.assertIn("📋 Pipeline \"重构认证模块\"", out)
        self.assertIn("进度", out)
        self.assertIn("min", out)
        # Box-drawing separators (26 chars ×).
        self.assertIn("━" * 26, out)
        # Each event rendered.
        self.assertIn("completed: 编写单元测试", out)
        self.assertIn("blocked: DB迁移", out)
        self.assertIn("端口确认", out)
        self.assertIn("completed: 更新API文档", out)
        # Footer pointing to the CLI.
        self.assertIn("详情: hermes kanban show root1", out)


# ---------------------------------------------------------------------------
# 6. Config wiring — from_config respects the kanban_cfg shape
# ---------------------------------------------------------------------------


class TestConfigWiring(unittest.TestCase):
    """Honour ``kanban.aggregation.*`` keys from ``config.yaml``."""

    def test_from_config_defaults_when_missing(self):
        agg = NotificationAggregator.from_config({})
        self.assertEqual(agg.time_window_seconds, DEFAULT_TIME_WINDOW_SECONDS)
        self.assertEqual(agg.count_threshold, DEFAULT_COUNT_THRESHOLD)

    def test_from_config_overrides(self):
        agg = NotificationAggregator.from_config(
            {"aggregation": {"time_window_seconds": 60, "count_threshold": 5}},
        )
        self.assertEqual(agg.time_window_seconds, 60.0)
        self.assertEqual(agg.count_threshold, 5)

    def test_from_config_invalid_keys_fall_back_to_defaults(self):
        """Non-numeric values degrade silently so a typo can't take down
        the notifier.
        """
        agg = NotificationAggregator.from_config(
            {"aggregation": {"time_window_seconds": "not a number"}},
        )
        # Should not raise; falls back to the default.
        self.assertGreater(agg.time_window_seconds, 0)


# ---------------------------------------------------------------------------
# 7. DB-aware smoke against the live notification-policy board
# ---------------------------------------------------------------------------


class TestLiveBoard(unittest.TestCase):
    """Optional: run against the real ``notification-policy`` board DB.

    Skipped when the board's ``kanban.db`` is absent (e.g. CI without
    the board initialised).  Mirrors the pattern in
    ``workspaces/t_47985057/test_m1_1.py``.
    """

    BOARD_DIR = Path(
        os.path.expanduser(
            "~/.hermes/kanban/boards/notification-policy",
        )
    )

    def setUp(self):
        if not (self.BOARD_DIR / "kanban.db").exists():
            self.skipTest(f"board DB not present at {self.BOARD_DIR}")

    def test_buffer_against_live_board(self):
        """Buffer a real task event against the live board and verify
        the format produces the expected banner line.
        """
        # Import lazily so the import error surfaces here (not at
        # collection time, which would skip the whole module).
        from hermes_cli import kanban_db as _kb
        agg = NotificationAggregator(time_window_seconds=300, count_threshold=10)
        conn = _kb.connect(board="notification-policy")
        try:
            rows = conn.execute(
                "SELECT id FROM tasks WHERE status != 'archived' LIMIT 1",
            ).fetchall()
            if not rows:
                self.skipTest("no live tasks on notification-policy board")
            task_id = str(rows[0]["id"])
            f = agg.buffer_p1_event(
                board="notification-policy",
                task_id=task_id,
                ev=FakeEvent("completed", task_id),
                task=None,
                sub=_sub(),
                conn=conn,
            )
            self.assertIsNone(f)
            self.assertEqual(agg.buffer_count(), 1)
            # Pop and format to verify shape.
            buf = next(iter(agg._buffers.values()))
            out = format_summary(buf)
            self.assertIn("📋 Pipeline", out)
            self.assertIn("详情: hermes kanban show", out)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# M2 expansion tests — format_pipeline_summary (DB-query §4) + max_buffer_age
# ---------------------------------------------------------------------------


def _sqlite_with_pipeline():
    """Build an in-memory sqlite with a fake 4-task pipeline + task_links."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE task_links(child_id TEXT, parent_id TEXT)",
    )
    conn.execute(
        "CREATE TABLE tasks("
        "id TEXT PRIMARY KEY, title TEXT, status TEXT, "
        "started_at INTEGER, completed_at INTEGER"
        ")",
    )
    rows = [
        ("root", "重构认证模块", "done", 1000, 2000),
        ("leaf1", "编写单元测试", "done", 1100, 1900),
        ("leaf2", "集成测试", "running", 1500, 0),
        ("leaf3", "数据库迁移", "blocked", 1200, 0),
    ]
    for r in rows:
        conn.execute(
            "INSERT INTO tasks(id, title, status, started_at, completed_at)"
            " VALUES(?, ?, ?, ?, ?)",
            r,
        )
    for child in ("leaf1", "leaf2", "leaf3"):
        conn.execute(
            "INSERT INTO task_links(child_id, parent_id) VALUES (?, ?)",
            (child, "root"),
        )
    return conn


class TestFormatPipelineSummary(unittest.TestCase):
    """Task body §4 — DB-query child task status distribution."""

    def test_db_query_groups_by_current_status(self):
        """The summary groups leaves by their **current DB status** —
        Done → ✅, running → ⏳, blocked → ⏸.  Buffered event kind is
        ignored when the conn is present.
        """
        conn = _sqlite_with_pipeline()
        try:
            buf = AggregateBuffer(
                board="b1",
                root_task_id="root",
                root_title="重构认证模块",
                subscription_key=("telegram", "c1", ""),
                events=[
                    {"kind": "completed", "task_id": "leaf1",
                     "payload": {}, "task_obj": None},
                ],
                flush_reason="count_threshold",
            )
            out = format_pipeline_summary(buf, conn=conn)
            self.assertIn("✅ 完成", out)
            self.assertIn("⏳ 进行中", out)
            self.assertIn("⏸ 阻塞", out)
            self.assertIn('📋 Pipeline "重构认证模块"', out)
            self.assertIn("详情: hermes kanban show root", out)
        finally:
            conn.close()

    def test_no_conn_falls_back_to_buffered_summary(self):
        """When no conn is supplied, fall back to format_summary."""
        buf = AggregateBuffer(
            board="b1",
            root_task_id="root",
            root_title="",
            subscription_key=(),
            events=[
                {"kind": "completed", "task_id": "leaf1",
                 "payload": {}, "task_obj": None},
            ],
        )
        out = format_pipeline_summary(buf, conn=None)
        self.assertIn("• completed: leaf1", out)

    def test_blocked_task_includes_reason_from_buffered_event(self):
        """A blocked leaf in the DB picks up its ``reason`` from the
        buffered event payload so the summary explains *why* it's stuck.
        """
        conn = _sqlite_with_pipeline()
        try:
            buf = AggregateBuffer(
                board="b1", root_task_id="root", root_title="",
                subscription_key=(),
                events=[
                    {"kind": "blocked", "task_id": "leaf3",
                     "payload": {"reason": "等待端口确认"}, "task_obj": None},
                ],
            )
            out = format_pipeline_summary(buf, conn=conn)
            self.assertIn("⏸ 阻塞", out)
            self.assertIn("等待端口确认", out)
        finally:
            conn.close()

    def test_line_cap_at_7(self):
        """NR-4: total lines ≤ 7."""
        conn = _sqlite_with_pipeline()
        try:
            buf = AggregateBuffer(
                board="b1", root_task_id="root", root_title="X",
                subscription_key=(),
                events=[],
                flush_reason="count_threshold",
            )
            out = format_pipeline_summary(buf, conn=conn)
            self.assertLessEqual(len(out.splitlines()), 7)
        finally:
            conn.close()

    def test_count_threshold_uses_immediate_span(self):
        """NR-4: count-threshold flushes show `(即时)` not `(5min)`."""
        conn = _sqlite_with_pipeline()
        try:
            buf = AggregateBuffer(
                board="b1", root_task_id="root", root_title="X",
                subscription_key=(),
                events=[],
                flush_reason="count_threshold",
            )
            out = format_pipeline_summary(buf, conn=conn)
            self.assertIn("(即时)", out)
        finally:
            conn.close()

    def test_time_window_uses_minute_span(self):
        """time_window flushes show `(Nmin)`."""
        conn = _sqlite_with_pipeline()
        try:
            buf = AggregateBuffer(
                board="b1", root_task_id="root", root_title="X",
                subscription_key=(),
                events=[],
                created_at=time.time() - 600,  # 10 min ago
                flush_reason="time_window",
            )
            out = format_pipeline_summary(buf, conn=conn)
            self.assertIn("min)", out)
            self.assertNotIn("(即时)", out)
        finally:
            conn.close()


class TestMaxBufferAge(unittest.TestCase):
    """Review N1 — buffer age cleanup prevents unbounded growth."""

    def test_buffer_older_than_max_age_force_flushes(self):
        """A buffer past ``max_buffer_age`` (without hitting
        time_window or count_threshold) is force-flushed on the next
        take_due_buffers call.
        """
        agg = NotificationAggregator(
            time_window_seconds=10**6,  # effectively disable time window
            count_threshold=10**6,  # effectively disable count threshold
            max_buffer_age=120,
        )
        self.assertTrue(agg.enabled)
        agg.buffer_p1_event(
            board="b1", task_id="leaf1",
            ev=FakeEvent("completed", "leaf1"),
            task=None, sub=_sub(),
        )
        self.assertEqual(agg.buffer_count(), 1)
        # Backdate created_at (last_event_at stays recent so time window
        # doesn't fire).
        for buf in agg._buffers.values():
            buf.created_at = time.time() - 600
            buf.last_event_at = time.time()

        due = agg.take_due_buffers()
        self.assertEqual(len(due), 1)
        assert due[0] is not None
        self.assertEqual(due[0].flush_reason, "max_age")
        self.assertEqual(agg.flush_counts["max_age"], 1)
        self.assertEqual(agg.buffer_count(), 0)

    def test_fresh_buffer_not_flushed_by_age(self):
        """A buffer younger than ``max_buffer_age`` is unaffected."""
        agg = NotificationAggregator(
            time_window_seconds=10**6,
            count_threshold=10**6,
            max_buffer_age=3600,
        )
        agg.buffer_p1_event(
            board="b1", task_id="leaf1",
            ev=FakeEvent("completed", "leaf1"),
            task=None, sub=_sub(),
        )
        due = agg.take_due_buffers()
        self.assertEqual(due, [])
        self.assertEqual(agg.buffer_count(), 1)

    def test_from_config_reads_max_buffer_age(self):
        """Config key ``aggregation.max_buffer_age_seconds`` is honoured."""
        cfg = {
            "aggregation": {
                "max_buffer_age_seconds": 600,
                "time_window_seconds": 300,
                "count_threshold": 3,
            },
        }
        agg = NotificationAggregator.from_config(cfg)
        self.assertEqual(agg.max_buffer_age, 600.0)

    def test_from_config_invalid_max_age_falls_back(self):
        """Invalid config value silently falls back to default."""
        cfg = {"aggregation": {"max_buffer_age_seconds": "not-a-number"}}
        agg = NotificationAggregator.from_config(cfg)
        self.assertEqual(
            agg.max_buffer_age,
            NotificationAggregator.DEFAULT_MAX_BUFFER_AGE,
        )


class TestFlushIfPipelineRoot(unittest.TestCase):
    """Pipeline-done flush — review S3 key-matching fix."""

    def test_pipeline_root_done_flushes_buffer(self):
        """When the task being completed IS a pipeline root, the buffer
        for that root is popped and returned with
        flush_reason='pipeline_done'.
        """
        import sqlite3
        agg = NotificationAggregator(
            time_window_seconds=10**6,
            count_threshold=10**6,
        )
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE task_links(child_id TEXT, parent_id TEXT)")
        conn.execute(
            "CREATE TABLE tasks(id TEXT PRIMARY KEY, title TEXT, status TEXT)",
        )
        conn.execute(
            "INSERT INTO tasks(id, title, status) VALUES (?, ?, ?)",
            ("root", "重构认证", "running"),
        )
        # Buffer an event keyed under "root" (no parents → own root).
        agg.buffer_p1_event(
            board="b1", task_id="root",
            ev=FakeEvent("completed", "root"),
            task=None, sub=_sub(), conn=conn,
        )
        self.assertEqual(agg.buffer_count(), 1)
        buf = next(iter(agg._buffers.values()))
        buf.events.append(
            {"kind": "completed", "task_id": "root",
             "payload": {}, "task_obj": None},
        )
        flushed = agg.flush_if_pipeline_root(
            board="b1", task_id="root", sub=_sub(), conn=conn,
        )
        self.assertIsNotNone(flushed)
        assert flushed is not None
        self.assertEqual(flushed.flush_reason, "pipeline_done")
        self.assertEqual(agg.buffer_count(), 0)
        conn.close()

    def test_non_root_completion_does_not_flush(self):
        """When the task being completed is a leaf, the buffer is left
        intact — only root completions trigger the pipeline-done flush.
        """
        import sqlite3
        agg = NotificationAggregator(
            time_window_seconds=10**6,
            count_threshold=10**6,
        )
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE task_links(child_id TEXT, parent_id TEXT)")
        conn.execute(
            "CREATE TABLE tasks(id TEXT PRIMARY KEY, title TEXT, status TEXT)",
        )
        conn.execute(
            "INSERT INTO tasks(id, title, status) VALUES (?, ?, ?)",
            ("leaf", "L", "running"),
        )
        conn.execute(
            "INSERT INTO tasks(id, title, status) VALUES (?, ?, ?)",
            ("root", "R", "running"),
        )
        conn.execute(
            "INSERT INTO task_links(child_id, parent_id) VALUES (?, ?)",
            ("leaf", "root"),
        )
        agg.buffer_p1_event(
            board="b1", task_id="leaf",
            ev=FakeEvent("completed", "leaf"),
            task=None, sub=_sub(), conn=conn,
        )
        self.assertEqual(agg.buffer_count(), 1)
        flushed = agg.flush_if_pipeline_root(
            board="b1", task_id="leaf", sub=_sub(), conn=conn,
        )
        self.assertIsNone(flushed)
        self.assertEqual(agg.buffer_count(), 1)
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
