"""
Unit tests for session_orchestration/watcher.py session-orchestration flow.

Coverage
--------
1. Core tick mechanics: state updates, lock handling, intent drain,
   unavailable-adapter skips, startup verification, and config gate.
2. Attention lifecycle: WAITING_USER / PAUSED_HANDOFF open, refresh, resolve,
   and direct user-attention transitions close the prior reason.
3. Stale/frozen lifecycle: deterministic eligibility, empty-capture safety,
   ambiguous RUNNING persistence, drive/progress resolution, and one nudge per
   stale episode.
4. Watcher-to-feed projection: digest reconciliation on attention entry,
   resolution, stale/frozen escalation, already-nudged stale episodes, and
   missing digest message recovery.

All tests use:
- An in-memory SQLite DB (via tmp_path fixture).
- A FakeAdapter that returns a fixed SessionLifecycle.
- A fake tmux_capture callable that records invocations.
- Injected probe_runner / probe_specs so no real binaries are invoked.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import session_orchestration.feed as feed_mod

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.adapters.verify import AdapterProbeSpec
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle
from session_orchestration.watcher import (
    SessionWatcher,
    _is_omp_nudge_checkin_eligible,
    _is_stale_frozen_eligible,
    _on_hang,
    _pane_hash,
    run_tick,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAdapter(AgentAdapter):
    """Adapter that returns a fixed lifecycle value and records detect() calls."""

    def __init__(
        self,
        lifecycle: SessionLifecycle = SessionLifecycle.RUNNING,
        idle_indicator_regex: re.Pattern[str] | None = None,
    ):
        self._lifecycle = lifecycle
        self._idle_indicator_regex = idle_indicator_regex
        self.detect_calls: List[SessionHandle] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(idle_indicator_regex=self._idle_indicator_regex)

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        raise NotImplementedError

    def drive(self, handle: SessionHandle, message: str) -> None:
        raise NotImplementedError

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        self.detect_calls.append(handle)
        return self._lifecycle

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        raise NotImplementedError

    def terminate(self, handle: SessionHandle) -> None:
        self.terminate_calls = getattr(self, "terminate_calls", [])
        self.terminate_calls.append(handle)


class FakeProbeRunner:
    """ProbeRunner that finds every binary and returns fixed help text."""

    def __init__(self, help_text: str = ""):
        self._help = help_text

    def which(self, binary: str) -> str | None:
        return f"/usr/local/bin/{binary}"

    def help_text(self, binary: str) -> str:
        return self._help


class FakeCapture:
    """Fake tmux_capture that returns fixed text and records calls."""

    def __init__(self, text: str = "some pane output"):
        self._text = text
        self.calls: List[str] = []

    def __call__(self, pane: str) -> str:
        self.calls.append(pane)
        return self._text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path: Path) -> SessionOrchestrationRegistry:
    return SessionOrchestrationRegistry(db_path=db_path)


def _fake_probe_spec(adapter: FakeAdapter) -> AdapterProbeSpec:
    """Return a probe spec whose flags are all None (always pass)."""
    return AdapterProbeSpec(binary="fake-agent")


def _make_watcher(
    registry: SessionOrchestrationRegistry,
    adapter: FakeAdapter,
    capture: FakeCapture,
    *,
    adapter_name: str = "fake",
) -> SessionWatcher:
    """Build a SessionWatcher with injected fakes."""
    watcher = SessionWatcher(
        registry=registry,
        adapters={adapter_name: adapter},
        tmux_capture=capture,
        # Fake tmux sessions are always "alive" so the dead-tmux reap (which
        # checks real tmux liveness) does not spuriously mark them ERROR.
        tmux_liveness_fn=lambda _s: True,
    )
    # Inject probe fakes so no real binaries are touched
    probe_runner = FakeProbeRunner()
    probe_specs = {type(adapter).__name__: _fake_probe_spec(adapter)}
    watcher.startup_verify(probe_runner=probe_runner, probe_specs=probe_specs)
    return watcher


def _seed_row(
    registry: SessionOrchestrationRegistry,
    task_id: str = "task-001",
    agent: str = "fake",
    state: str = "RUNNING",
    tmux_session: str = "hermes-fake-001",
) -> None:
    """Insert a minimal registry row.

    The registry schema has ``tmux_session`` but no ``pane`` column.
    The watcher derives the pane as ``<tmux_session>:0.0`` from the row.
    """
    registry.upsert(
        task_id,
        agent=agent,
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        repo=f"repo-{uuid.uuid4().hex[:8]}",
        state=state,
        tmux_session=tmux_session,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTickWritesState:
    """Tick iterates seeded rows and writes state to the registry."""

    def test_state_written_for_active_row(self, registry, db_path):
        task_id = "t-state-001"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        capture = FakeCapture("❯ ")
        watcher = _make_watcher(registry, adapter, capture)

        with (
            patch(
                "session_orchestration.feed.reconcile_attention_digest",
                return_value={"status": "posted"},
            ),
            patch("session_orchestration.feed.push_turn_change", return_value=False),
        ):
            processed = watcher.tick()

        assert processed == 1, "expected exactly one row processed"
        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "WAITING_USER"

    def test_menu_question_and_options_persisted_from_pane(self, registry, db_path):
        """omp emits no markers: the watcher parses the pane and persists the
        question, ordered options, and last_input_kind='menu' on WAITING_USER."""
        task_id = "t-menu-001"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        menu_pane = (
            " Apply the proposed edit?\n"
            "────────────────────────────\n"
            "│ Accept (Recommended)     │\n"
            "│    Apply it now.         │\n"
            "│ Defer                    │\n"
            "────────────────────────────\n"
            " up/down navigate  enter select  esc cancel"
        )
        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        capture = FakeCapture(menu_pane)
        watcher = _make_watcher(registry, adapter, capture)

        with (
            patch(
                "session_orchestration.feed.reconcile_attention_digest",
                return_value={"status": "posted"},
            ),
            patch("session_orchestration.feed.push_turn_change", return_value=False),
        ):
            watcher.tick()

        row = registry.get(task_id)
        assert row["state"] == "WAITING_USER"
        assert row["last_input_kind"] == "menu"
        assert json.loads(row["last_options"]) == ["Accept (Recommended)", "Defer"]
        assert "Apply the proposed edit?" in (row["last_question"] or "")

    def test_input_kind_cleared_when_leaving_waiting(self, registry, db_path):
        """A stale 'menu' marker must not survive the return to RUNNING."""
        task_id = "t-menu-002"
        _seed_row(registry, task_id=task_id, state="WAITING_USER")
        registry.upsert(task_id, agent="fake", last_input_kind="menu",
                        last_options=json.dumps(["A", "B"]))

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("⠹ working…")
        watcher = _make_watcher(registry, adapter, capture)
        with patch("session_orchestration.feed.push_turn_change", return_value=False):
            watcher.tick()

        row = registry.get(task_id)
        assert row["state"] == "RUNNING"
        assert row["last_input_kind"] == ""

    def test_session_end_while_waiting_reaps_attention_and_pings(
        self, registry, db_path
    ):
        """A session that ends (DONE) while WAITING_USER must resolve its
        attention item (drops from the feed digest) and clear the ping
        debounce so no further @-pings fire."""
        import session_orchestration.feed as feed_mod

        task_id = "t-reap-end"
        _seed_row(registry, task_id=task_id, state="WAITING_USER")
        registry.open_attention_item(task_id, "WAITING_USER", priority=100)
        feed_mod._last_notified[task_id] = "WAITING_USER"

        # Attention item is open before the tick.
        assert any(
            it["reason"] == "WAITING_USER"
            for it in registry.list_unresolved_attention_items()
        )

        adapter = FakeAdapter(SessionLifecycle.DONE)
        watcher = _make_watcher(registry, adapter, FakeCapture("done."))
        with patch(
            "session_orchestration.feed.reconcile_attention_digest",
            return_value={"status": "edited"},
        ):
            watcher.tick()

        # Attention item resolved → falls out of the digest.
        assert not [
            it
            for it in registry.list_unresolved_attention_items()
            if it["reason"] == "WAITING_USER"
        ]
        # Ping debounce cleared → no lingering @-ping state.
        assert task_id not in feed_mod._last_notified

    def test_terminate_intent_reaps_attention_and_marks_terminal(
        self, registry, db_path
    ):
        """A terminate intent (thread archived/closed) kills the session AND
        reaps its attention item + feed line + ping debounce, even though the
        now-terminal row is never re-iterated by the active loop."""
        import session_orchestration.feed as feed_mod

        task_id = "t-archive-term"
        _seed_row(registry, task_id=task_id, state="WAITING_USER")
        registry.open_attention_item(task_id, "WAITING_USER", priority=100)
        feed_mod._last_notified[task_id] = "WAITING_USER"
        registry.enqueue_terminate(task_id)

        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        watcher = _make_watcher(registry, adapter, FakeCapture("menu…"))
        with patch(
            "session_orchestration.feed.reconcile_attention_digest",
            return_value={"status": "edited"},
        ):
            watcher.tick()

        row = registry.get(task_id)
        assert row["state"] in ("DONE", "ERROR"), "session marked terminal"
        assert not registry.list_unresolved_attention_items(), "attention reaped"
        assert task_id not in feed_mod._last_notified, "ping debounce cleared"

    def test_omp_stable_idle_promotes_running_to_waiting_user(self, registry, db_path):
        """omp free-form wait: a detect()=RUNNING pane that idle_waiting() flags
        is promoted to WAITING_USER only after it is STABLE across a tick (guards
        against flapping). First tick stays RUNNING (no prior hash); second tick
        with an identical pane promotes."""
        class IdleAdapter(FakeAdapter):
            def idle_waiting(self, pane_text):  # opt-in signal, always idle here
                return True

        task_id = "t-idle-omp"
        _seed_row(registry, task_id=task_id, state="RUNNING")
        adapter = IdleAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture(" What do you want me to do?\n╭── omp ──╮\n╰─  ─╯")
        watcher = _make_watcher(registry, adapter, capture)

        with (
            patch("session_orchestration.feed.push_turn_change", return_value=False),
            patch(
                "session_orchestration.feed.reconcile_attention_digest",
                return_value={"status": "posted"},
            ),
        ):
            watcher.tick()
            row_after_1 = registry.get(task_id)
            watcher.tick()
            row_after_2 = registry.get(task_id)

        assert row_after_1["state"] == "RUNNING", "1st tick: not yet stable"
        assert row_after_2["state"] == "WAITING_USER", "2nd tick: stable idle → waiting"
        assert row_after_2["last_input_kind"] == "prompt"
        assert "What do you want me to do?" in (row_after_2["last_question"] or "")

    def test_no_spurious_notification_on_same_state(self, registry, db_path):
        """Tick must NOT raise or produce side-effects when state unchanged."""
        task_id = "t-state-002"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("working…")
        watcher = _make_watcher(registry, adapter, capture)

        # Should not raise; processed count reflects the single row
        processed = watcher.tick()
        assert processed == 1

        row = registry.get(task_id)
        assert row["state"] == "RUNNING"

    def test_terminal_rows_not_processed(self, registry, db_path):
        """Rows in DONE/ERROR state are skipped."""
        for tid, state in [("t-done", "DONE"), ("t-error", "ERROR")]:
            _seed_row(registry, task_id=tid, state=state)

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)

        processed = watcher.tick()
        assert processed == 0
        # capture was never called for terminal rows
        assert capture.calls == []



class TestAttentionLifecycle:
    """Watcher-owned attention items mirror lifecycle without new states."""

    def test_waiting_user_attention_open_refresh_resolve(self, registry, db_path):
        task_id = "t-attn-waiting"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        watcher = _make_watcher(registry, adapter, FakeCapture("❯ "))

        with (
            patch(
                "session_orchestration.feed.reconcile_attention_digest",
                return_value={"status": "edited"},
            ) as reconcile_digest,
            patch("session_orchestration.feed.push_turn_change", return_value=True) as thread_notice,
            patch("session_orchestration.watcher._on_heartbeat_tick") as heartbeat_tick,
        ):
            watcher.tick()
            reconcile_digest.assert_called_once_with(registry)
            thread_notice.assert_called_once()
            first_items = registry.list_unresolved_attention_items()
            waiting_items = [
                item for item in first_items if item["reason"] == "WAITING_USER"
            ]
            assert len(waiting_items) == 1
            first_id = waiting_items[0]["id"]

            watcher.tick()
            assert reconcile_digest.call_count == 1
            refreshed_items = registry.list_unresolved_attention_items()
            waiting_items = [
                item for item in refreshed_items if item["reason"] == "WAITING_USER"
            ]
            assert [item["id"] for item in waiting_items] == [first_id]

            adapter._lifecycle = SessionLifecycle.RUNNING
            watcher.tick()
            assert reconcile_digest.call_count == 2
            heartbeat_tick.assert_not_called()

        assert not [
            item
            for item in registry.list_unresolved_attention_items()
            if item["reason"] == "WAITING_USER"
        ]

    def test_paused_handoff_attention_open_refresh_resolve(self, registry, db_path):
        task_id = "t-attn-handoff"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        adapter = FakeAdapter(SessionLifecycle.PAUSED_HANDOFF)
        watcher = _make_watcher(registry, adapter, FakeCapture("HERMES_HANDOFF\n❯ "))

        with (
            patch(
                "session_orchestration.feed.reconcile_attention_digest",
                return_value={"status": "edited"},
            ) as reconcile_digest,
            patch("session_orchestration.feed.push_turn_change", return_value=True) as thread_notice,
            patch("session_orchestration.watcher._on_heartbeat_tick") as heartbeat_tick,
        ):
            watcher.tick()
            reconcile_digest.assert_called_once_with(registry)
            thread_notice.assert_called_once()
            first_items = registry.list_unresolved_attention_items()
            handoff_items = [
                item for item in first_items if item["reason"] == "PAUSED_HANDOFF"
            ]
            assert len(handoff_items) == 1
            first_id = handoff_items[0]["id"]

            watcher.tick()
            assert reconcile_digest.call_count == 1
            refreshed_items = registry.list_unresolved_attention_items()
            handoff_items = [
                item for item in refreshed_items if item["reason"] == "PAUSED_HANDOFF"
            ]
            assert [item["id"] for item in handoff_items] == [first_id]

            adapter._lifecycle = SessionLifecycle.RUNNING
            watcher.tick()
            assert reconcile_digest.call_count == 2
            heartbeat_tick.assert_not_called()

        assert not [
            item
            for item in registry.list_unresolved_attention_items()
            if item["reason"] == "PAUSED_HANDOFF"
        ]

    def test_leaving_user_attention_state_rearms_turn_change_debounce(
        self, registry, db_path
    ):
        import session_orchestration.feed as feed_mod

        task_id = "t-attn-debounce"
        _seed_row(registry, task_id=task_id, state="WAITING_USER")
        feed_mod._last_notified[task_id] = "WAITING_USER"

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture("working"))
        with patch(
            "session_orchestration.feed.reconcile_attention_digest",
            return_value={"status": "edited"},
        ) as reconcile_digest:
            watcher.tick()

        reconcile_digest.assert_called_once_with(registry)

        assert task_id not in feed_mod._last_notified

    def test_stale_frozen_attention_opens_on_running_row(self, registry, db_path):
        task_id = "t-attn-stale-open"
        pane_text = "unchanged frozen pane"
        pane_hash = _pane_hash(pane_text)
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=pane_hash,
            last_output_ts=time.time() - 999,
            idle_ticks=3,
            nudge_count=1,
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture(pane_text))
        watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "RUNNING"
        assert [
            item
            for item in registry.list_unresolved_attention_items()
            if item["reason"] == "STALE_FROZEN"
        ]

    def test_stale_frozen_escalation_reconciles_attention_digest(
        self, registry, db_path
    ):
        task_id = "t-attn-stale-digest"
        pane_text = "unchanged frozen pane for digest"
        pane_hash = _pane_hash(pane_text)
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=pane_hash,
            last_output_ts=time.time() - 999,
            idle_ticks=3,
            nudge_count=0,
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture(pane_text))

        with (
            patch(
                "session_orchestration.feed.reconcile_attention_digest",
                return_value={"status": "edited"},
            ) as reconcile_digest,
            patch("session_orchestration.feed.push_hang_notification", return_value=False) as thread_notice,
            patch("session_orchestration.watcher._on_heartbeat_tick") as heartbeat_tick,
            patch("session_orchestration.watcher._send_auto_nudge") as auto_nudge,
        ):
            watcher.tick()

        reconcile_digest.assert_called_once_with(registry)
        thread_notice.assert_called_once()
        auto_nudge.assert_called_once()
        heartbeat_tick.assert_not_called()
        assert [
            item
            for item in registry.list_unresolved_attention_items()
            if item["reason"] == "STALE_FROZEN"
        ]

    def test_stale_frozen_not_resolved_by_default_running_detect(
        self, registry, db_path
    ):
        task_id = "t-attn-stale-running"
        pane_text = "same pane without enough stale evidence"
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=_pane_hash(pane_text),
            idle_ticks=0,
        )
        stale_item = registry.open_attention_item(task_id, "STALE_FROZEN")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture(pane_text))
        watcher.tick()

        unresolved = registry.list_unresolved_attention_items()
        stale_ids = [
            item["id"] for item in unresolved if item["reason"] == "STALE_FROZEN"
        ]
        assert stale_ids == [stale_item["id"]]

    def test_empty_capture_does_not_resolve_stale_frozen_or_reset_counters(
        self, registry, db_path
    ):
        task_id = "t-attn-stale-empty-capture"
        old_text = "old frozen pane"
        old_hash = _pane_hash(old_text)
        old_last_output_ts = time.time() - 999
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=old_hash,
            last_output_ts=old_last_output_ts,
            idle_ticks=3,
            nudge_count=1,
        )
        stale_item = registry.open_attention_item(task_id, "STALE_FROZEN")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture(""))
        watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["last_pane_hash"] == old_hash
        assert row["last_output_ts"] == pytest.approx(old_last_output_ts)
        assert row["idle_ticks"] == 3
        assert row["nudge_count"] == 1
        stale_ids = [
            item["id"]
            for item in registry.list_unresolved_attention_items()
            if item["reason"] == "STALE_FROZEN"
        ]
        assert stale_ids == [stale_item["id"]]

    def test_stale_frozen_resolves_on_drive_signal_when_represented(
        self, registry, db_path
    ):
        task_id = "t-attn-stale-drive"
        pane_text = "still frozen but user replied"
        pane_hash = _pane_hash(pane_text)
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=pane_hash,
            last_output_ts=time.time() - 999,
            idle_ticks=3,
            nudge_count=0,
        )
        registry.open_attention_item(task_id, "STALE_FROZEN")
        registry.enqueue_intent(
            "drive",
            task_id=task_id,
            payload={"task_id": task_id},
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture(pane_text))
        with (
            patch("session_orchestration.watcher._on_hang") as on_hang,
            patch(
                "session_orchestration.feed.reconcile_attention_digest",
                return_value={"status": "edited"},
            ) as reconcile_digest,
        ):
            watcher.tick()
            on_hang.assert_not_called()

        reconcile_digest.assert_called_once_with(registry)

        assert not [
            item
            for item in registry.list_unresolved_attention_items()
            if item["reason"] == "STALE_FROZEN"
        ]

    def test_stale_frozen_resolves_on_pane_hash_change(self, registry, db_path):
        task_id = "t-attn-stale-resolve"
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=_pane_hash("old frozen pane"),
            last_output_ts=time.time() - 999,
            idle_ticks=3,
        )
        registry.open_attention_item(task_id, "STALE_FROZEN")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture("new active output"))
        with patch(
            "session_orchestration.feed.reconcile_attention_digest",
            return_value={"status": "edited"},
        ) as reconcile_digest:
            watcher.tick()

        reconcile_digest.assert_called_once_with(registry)

        assert not [
            item
            for item in registry.list_unresolved_attention_items()
            if item["reason"] == "STALE_FROZEN"
        ]


    def test_waiting_user_digest_removes_item_after_progress(self, registry, db_path):
        task_id = "t-attn-digest-progress"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        posted: List[str] = []
        edited: List[str] = []

        def fake_post(channel_id: str, content: str, *, token=None):
            posted.append(content)
            return "digest-msg-1"

        def fake_edit(channel_id: str, message_id: str, content: str, *, token=None):
            edited.append(content)
            return feed_mod._EDIT_OK

        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        watcher = _make_watcher(registry, adapter, FakeCapture("waiting at prompt"))
        with (
            patch("session_orchestration.feed._get_feed_channel_id", return_value="feed-1"),
            patch("session_orchestration.feed._post_discord_message", side_effect=fake_post),
            patch("session_orchestration.feed._edit_discord_message", side_effect=fake_edit),
            patch("session_orchestration.watcher._on_heartbeat_tick") as heartbeat_tick,
        ):
            watcher.tick()
            assert len(posted) == 1
            assert task_id in posted[0]
            assert "reason `WAITING_USER`" in posted[0]
            assert registry.get_projection(
                "feed-1", feed_mod._ATTENTION_DIGEST_PROJECTION_NAME
            )["payload"] == {"item_count": 1, "task_ids": [task_id]}

            adapter._lifecycle = SessionLifecycle.RUNNING
            watcher._tmux_capture = FakeCapture("agent made progress")
            watcher.tick()

        assert edited == ["**Hermes action feed**\nNo unresolved attention items."]
        assert registry.list_unresolved_attention_items() == []
        assert registry.get_projection(
            "feed-1", feed_mod._ATTENTION_DIGEST_PROJECTION_NAME
        )["payload"] == {"item_count": 0, "task_ids": []}
        heartbeat_tick.assert_not_called()

    def test_stale_frozen_digest_persists_across_running_then_drive_removes(
        self, registry, db_path
    ):
        task_id = "t-attn-stale-digest-drive"
        pane_text = "unchanged ambiguous running pane"
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=_pane_hash(pane_text),
            idle_ticks=0,
        )
        registry.open_attention_item(
            task_id,
            "STALE_FROZEN",
            priority=50,
            detail="existing unresolved stale/frozen item",
        )

        posted: List[str] = []
        edited: List[str] = []

        def fake_post(channel_id: str, content: str, *, token=None):
            posted.append(content)
            return "digest-msg-1"

        def fake_edit(channel_id: str, message_id: str, content: str, *, token=None):
            edited.append(content)
            return feed_mod._EDIT_OK

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture(pane_text))
        with (
            patch("session_orchestration.feed._get_feed_channel_id", return_value="feed-1"),
            patch("session_orchestration.feed._post_discord_message", side_effect=fake_post),
            patch("session_orchestration.feed._edit_discord_message", side_effect=fake_edit),
            patch("session_orchestration.watcher._on_heartbeat_tick"),
        ):
            feed_mod.reconcile_attention_digest(registry)
            assert len(posted) == 1
            assert task_id in posted[0]
            assert "reason `STALE_FROZEN`" in posted[0]

            watcher.tick()
            unresolved = registry.list_unresolved_attention_items()
            assert [item["reason"] for item in unresolved] == ["STALE_FROZEN"]
            assert edited == []

            registry.enqueue_intent("drive", task_id=task_id, payload={"task_id": task_id})
            watcher.tick()

        assert edited == ["**Hermes action feed**\nNo unresolved attention items."]
        assert registry.list_unresolved_attention_items() == []

    def test_missing_digest_message_recreated_with_current_attention_item(
        self, registry, db_path
    ):
        task_id = "t-attn-missing-digest"
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.open_attention_item(task_id, "WAITING_USER", priority=100)
        registry.upsert_projection(
            "feed-1",
            feed_mod._ATTENTION_DIGEST_PROJECTION_NAME,
            message_id="missing-digest-msg",
            content_hash="stale-hash",
            payload={"item_count": 0, "task_ids": []},
        )

        posted: List[str] = []

        def fake_post(channel_id: str, content: str, *, token=None):
            posted.append(content)
            return "replacement-digest-msg"

        with (
            patch(
                "session_orchestration.feed._edit_discord_message",
                return_value=feed_mod._EDIT_MISSING,
            ) as edit_message,
            patch("session_orchestration.feed._post_discord_message", side_effect=fake_post),
        ):
            result = feed_mod.reconcile_attention_digest(
                registry,
                feed_channel_id="feed-1",
                now="2026-06-29T02:00:00+00:00",
            )

        assert result["status"] == "recreated"
        edit_message.assert_called_once()
        assert len(posted) == 1
        assert task_id in posted[0]
        projection = registry.get_projection(
            "feed-1", feed_mod._ATTENTION_DIGEST_PROJECTION_NAME
        )
        assert projection["message_id"] == "replacement-digest-msg"
        assert projection["payload"] == {"item_count": 1, "task_ids": [task_id]}

    def test_already_nudged_stale_episode_still_reconciles_digest(
        self, registry, db_path
    ):
        task_id = "t-attn-stale-already-nudged-digest"
        pane_text = "unchanged stale pane after earlier nudge"
        pane_hash = _pane_hash(pane_text)
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=pane_hash,
            last_output_ts=time.time() - 999,
            idle_ticks=3,
            nudge_count=1,
        )

        posted: List[str] = []

        def fake_post(channel_id: str, content: str, *, token=None):
            posted.append(content)
            return "digest-msg-1"

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture(pane_text))
        with (
            patch("session_orchestration.feed._get_feed_channel_id", return_value="feed-1"),
            patch("session_orchestration.feed._post_discord_message", side_effect=fake_post),
            patch("session_orchestration.feed.push_hang_notification") as hang_notice,
            patch("session_orchestration.watcher._send_auto_nudge") as auto_nudge,
            patch("session_orchestration.watcher._on_heartbeat_tick") as heartbeat_tick,
        ):
            watcher.tick()

        assert len(posted) == 1
        assert task_id in posted[0]
        assert "reason `STALE_FROZEN`" in posted[0]
        assert registry.get_projection(
            "feed-1", feed_mod._ATTENTION_DIGEST_PROJECTION_NAME
        )["payload"] == {"item_count": 1, "task_ids": [task_id]}
        hang_notice.assert_not_called()
        auto_nudge.assert_not_called()
        heartbeat_tick.assert_not_called()

    def test_direct_user_attention_transition_closes_prior_reason_and_updates_digest(
        self, registry, db_path
    ):
        task_id = "t-attn-direct-transition"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        posted: List[str] = []
        edited: List[str] = []

        def fake_post(channel_id: str, content: str, *, token=None):
            posted.append(content)
            return "digest-msg-1"

        def fake_edit(channel_id: str, message_id: str, content: str, *, token=None):
            edited.append(content)
            return feed_mod._EDIT_OK

        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        watcher = _make_watcher(registry, adapter, FakeCapture("waiting"))
        with (
            patch("session_orchestration.feed._get_feed_channel_id", return_value="feed-1"),
            patch("session_orchestration.feed._post_discord_message", side_effect=fake_post),
            patch("session_orchestration.feed._edit_discord_message", side_effect=fake_edit),
            patch("session_orchestration.watcher._on_heartbeat_tick") as heartbeat_tick,
        ):
            watcher.tick()
            adapter._lifecycle = SessionLifecycle.PAUSED_HANDOFF
            watcher._tmux_capture = FakeCapture("handoff")
            watcher.tick()

        assert len(posted) == 1
        assert "reason `WAITING_USER`" in posted[0]
        assert len(edited) == 1
        assert "reason `PAUSED_HANDOFF`" in edited[0]
        assert "reason `WAITING_USER`" not in edited[0]
        unresolved = registry.list_unresolved_attention_items()
        assert [item["reason"] for item in unresolved] == ["PAUSED_HANDOFF"]
        heartbeat_tick.assert_not_called()

class TestLockSkipsCapture:
    """A tick whose target session is locked skips capture entirely."""

    def test_capture_not_called_when_locked(self, registry, db_path):
        task_id = "t-lock-001"
        _seed_row(registry, task_id=task_id, tmux_session="locked-session")

        # Acquire the lock from an "external" holder (simulates relay)
        external_holder = "relay:pid:9999:1234567890.000"
        acquired = registry.acquire_lock(
            task_id, external_holder, ttl_seconds=300.0
        )
        assert acquired, "setup: external holder should have acquired lock"

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)

        processed = watcher.tick()

        # Row was skipped — no capture call, processed count is 0
        assert processed == 0
        assert capture.calls == [], (
            "capture-pane must NOT be called while lock is held by relay"
        )
        # adapter.detect should not have been called either
        assert adapter.detect_calls == []

    def test_capture_called_when_unlocked(self, registry, db_path):
        task_id = "t-lock-002"
        _seed_row(registry, task_id=task_id, tmux_session="free-session")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("running…")
        watcher = _make_watcher(registry, adapter, capture)

        processed = watcher.tick()

        assert processed == 1
        assert any("free-session" in c for c in capture.calls)

    def test_lock_released_after_capture(self, registry, db_path):
        """Lock must be released after capture so relay can acquire it next."""
        task_id = "t-lock-003"
        _seed_row(registry, task_id=task_id, tmux_session="rel-session")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)

        watcher.tick()

        # After tick, relay must be able to acquire the lock
        row = registry.get(task_id)
        assert row is not None
        # lock_holder should be None (released)
        assert row.get("lock_holder") is None, (
            "lock_holder must be released after capture"
        )


class TestIntentQueueDrain:
    """Intent-queue entries are drained and applied during the tick."""

    def test_adopt_intent_creates_row(self, registry, db_path):
        """An 'adopt' intent enqueued before the tick creates a registry row."""
        task_id = "t-intent-adopt-001"
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        repo = f"repo-{uuid.uuid4().hex[:8]}"

        registry.enqueue_intent(
            "adopt",
            task_id=task_id,
            run_id=run_id,
            repo=repo,
            payload={"agent": "fake", "run_id": run_id, "repo": repo, "task_id": task_id},
        )

        # No rows yet
        assert registry.get(task_id) is None

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)

        # Tick drains the intent → row appears
        watcher.tick()

        row = registry.get(task_id)
        assert row is not None, "adopt intent should have created a registry row"
        assert row["agent"] == "fake"

    def test_update_intent_modifies_existing_row(self, registry, db_path):
        """An 'update' intent modifies fields on an existing row."""
        task_id = "t-intent-update-001"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        # Enqueue an update intent to change the project field
        registry.enqueue_intent(
            "update",
            task_id=task_id,
            payload={"task_id": task_id, "project": "my-project"},
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row.get("project") == "my-project"

    def test_queue_empty_after_drain(self, registry, db_path):
        """Queue is empty after the tick drains it."""
        task_id = "t-intent-drain-001"
        registry.enqueue_intent(
            "update",
            task_id=task_id,
            payload={"task_id": task_id, "project": "x"},
        )

        # Seed the row so the update has somewhere to land
        _seed_row(registry, task_id=task_id)

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        # Second tick: intents drained = empty
        intents = registry.drain_intents()
        assert intents == [], "queue should be empty after the tick drained it"


class TestUnavailableAdapterSkipped:
    """Rows whose agent name is not in available adapters are skipped."""

    def test_unknown_agent_row_skipped(self, registry, db_path):
        task_id = "t-unavail-001"
        _seed_row(registry, task_id=task_id, agent="unknown-agent")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        # Watcher only has "fake", not "unknown-agent"
        watcher = _make_watcher(registry, adapter, capture, adapter_name="fake")

        processed = watcher.tick()

        assert processed == 0
        assert capture.calls == []

    def test_available_agent_processed_while_unknown_skipped(self, registry, db_path):
        """One row available, one not — only the available one is processed."""
        _seed_row(registry, task_id="t-avail", agent="fake", state="RUNNING")
        _seed_row(registry, task_id="t-unavail", agent="mystery-agent", state="RUNNING")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture, adapter_name="fake")

        processed = watcher.tick()

        assert processed == 1


class TestStartupVerifyIdempotent:
    """startup_verify is a no-op when called multiple times."""

    def test_second_call_is_noop(self, registry, db_path):
        adapter = FakeAdapter()
        probe_runner = FakeProbeRunner()
        probe_specs = {type(adapter).__name__: _fake_probe_spec(adapter)}

        call_count = [0]
        original_verify = __import__(
            "session_orchestration.adapters.verify", fromlist=["verify_adapters"]
        ).verify_adapters

        watcher = SessionWatcher(
            registry=registry,
            adapters={"fake": adapter},
        )
        watcher.startup_verify(probe_runner=probe_runner, probe_specs=probe_specs)
        first_available = dict(watcher._available_adapters)

        # Second call should be a no-op (same result, no re-probe)
        watcher.startup_verify(probe_runner=FakeProbeRunner("different"), probe_specs=probe_specs)
        second_available = dict(watcher._available_adapters)

        # Available set unchanged
        assert set(first_available.keys()) == set(second_available.keys())


class TestPaneHash:
    """_pane_hash returns consistent values for change detection."""

    def test_same_text_same_hash(self):
        assert _pane_hash("hello world") == _pane_hash("hello world")

    def test_different_text_different_hash(self):
        assert _pane_hash("hello") != _pane_hash("world")

    def test_empty_string(self):
        h = _pane_hash("")
        assert isinstance(h, str)
        assert len(h) == 16


class TestIdleTickIncrement:
    """idle_ticks increments when pane text is unchanged."""

    def test_idle_ticks_increment_on_unchanged_pane(self, registry, db_path):
        task_id = "t-idle-001"
        _seed_row(registry, task_id=task_id, tmux_session="idle-session")

        static_text = "working… (same every tick)"
        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture(static_text)
        watcher = _make_watcher(registry, adapter, capture)

        # First tick: establishes pane_hash
        watcher.tick()
        row1 = registry.get(task_id)
        idle1 = row1.get("idle_ticks", 0)

        # Second tick: same pane text → idle_ticks should increase
        watcher.tick()
        row2 = registry.get(task_id)
        idle2 = row2.get("idle_ticks", 0)

        assert idle2 > idle1, (
            f"idle_ticks should increase when pane text is unchanged "
            f"(got {idle1} then {idle2})"
        )

    def test_idle_ticks_reset_on_changed_pane(self, registry, db_path):
        task_id = "t-idle-002"
        _seed_row(registry, task_id=task_id, tmux_session="active-session")

        texts = ["first output", "second output"]
        call_n = [0]

        def changing_capture(pane: str) -> str:
            idx = min(call_n[0], len(texts) - 1)
            call_n[0] += 1
            return texts[idx]

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = SessionWatcher(
            registry=registry,
            adapters={"fake": adapter},
            tmux_capture=changing_capture,
        )
        probe_runner = FakeProbeRunner()
        probe_specs = {type(adapter).__name__: _fake_probe_spec(adapter)}
        watcher.startup_verify(probe_runner=probe_runner, probe_specs=probe_specs)

        # Tick 1 — establishes first hash
        watcher.tick()

        # Tick 2 — different text → idle_ticks should be reset to 0
        watcher.tick()
        row = registry.get(task_id)
        assert (row.get("idle_ticks") or 0) == 0

    def test_pane_change_resets_stale_episode_action_counter(self, registry, db_path):
        task_id = "t-idle-003"
        old_text = "old frozen output"
        _seed_row(registry, task_id=task_id, tmux_session="reset-session")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=_pane_hash(old_text),
            last_output_ts=time.time() - 999,
            idle_ticks=5,
            nudge_count=1,
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("new output after progress")
        watcher = _make_watcher(registry, adapter, capture)

        watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["nudge_count"] == 0


class TestRunTickEntrypoint:
    """run_tick remains a single tick-driven cron entrypoint."""

    def test_run_tick_processes_one_cron_tick(self, registry, db_path):
        task_id = "t-run-tick-001"
        _seed_row(registry, task_id=task_id, state="RUNNING")
        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("single tick output")

        processed = run_tick(
            registry=registry,
            adapters={"fake": adapter},
            tmux_capture=capture,
            probe_runner=FakeProbeRunner(),
            probe_specs={type(adapter).__name__: _fake_probe_spec(adapter)},
            tmux_liveness_fn=lambda _s: True,
        )

        assert processed == 1
        assert len(capture.calls) == 1


class TestStaleFrozenGuards:
    """Deterministic stale/frozen and OMP action eligibility guards."""

    def test_eligible_with_static_hash_idle_threshold_stale_output_and_no_activity(self):
        now = time.time()
        pane_text = "unchanged pane output"
        pane_hash = _pane_hash(pane_text)
        row = {
            "idle_ticks": 3,
            "last_output_ts": now - 301,
            "nudge_count": 0,
        }

        assert _is_stale_frozen_eligible(
            row,
            previous_pane_hash=pane_hash,
            current_pane_hash=pane_hash,
            pane_text=pane_text,
            adapter=FakeAdapter(SessionLifecycle.RUNNING),
            hang_idle_ticks=3,
            hang_stale_seconds=300,
            now=now,
        )
        assert _is_omp_nudge_checkin_eligible(
            row,
            previous_pane_hash=pane_hash,
            current_pane_hash=pane_hash,
            pane_text=pane_text,
            adapter=FakeAdapter(SessionLifecycle.RUNNING),
            hang_idle_ticks=3,
            hang_stale_seconds=300,
            now=now,
        )

    def test_not_eligible_when_active_regex_matches(self):
        now = time.time()
        pane_text = "Running tool: cargo build"
        pane_hash = _pane_hash(pane_text)
        row = {
            "idle_ticks": 3,
            "last_output_ts": now - 301,
            "nudge_count": 0,
        }
        adapter = FakeAdapter(
            SessionLifecycle.RUNNING,
            idle_indicator_regex=re.compile(r"Running tool"),
        )

        assert not _is_stale_frozen_eligible(
            row,
            previous_pane_hash=pane_hash,
            current_pane_hash=pane_hash,
            pane_text=pane_text,
            adapter=adapter,
            hang_idle_ticks=3,
            hang_stale_seconds=300,
            now=now,
        )

    def test_not_eligible_when_output_is_fresh(self):
        now = time.time()
        pane_text = "unchanged but recent"
        pane_hash = _pane_hash(pane_text)
        row = {
            "idle_ticks": 3,
            "last_output_ts": now - 30,
            "nudge_count": 0,
        }

        assert not _is_stale_frozen_eligible(
            row,
            previous_pane_hash=pane_hash,
            current_pane_hash=pane_hash,
            pane_text=pane_text,
            adapter=FakeAdapter(SessionLifecycle.RUNNING),
            hang_idle_ticks=3,
            hang_stale_seconds=300,
            now=now,
        )

    def test_not_eligible_without_last_output_timestamp(self):
        now = time.time()
        pane_text = "unchanged but undated"
        pane_hash = _pane_hash(pane_text)
        row = {
            "idle_ticks": 3,
            "nudge_count": 0,
        }

        assert not _is_stale_frozen_eligible(
            row,
            previous_pane_hash=pane_hash,
            current_pane_hash=pane_hash,
            pane_text=pane_text,
            adapter=FakeAdapter(SessionLifecycle.RUNNING),
            hang_idle_ticks=3,
            hang_stale_seconds=300,
            now=now,
        )

    def test_tick_does_not_fire_stale_actions_without_last_output_timestamp(
        self, registry, db_path
    ):
        task_id = "t-stale-no-output-ts"
        pane_text = "unchanged but no timestamp"
        _seed_row(registry, task_id=task_id, state="RUNNING")
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_pane_hash=_pane_hash(pane_text),
            idle_ticks=3,
            nudge_count=0,
        )
        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = _make_watcher(registry, adapter, FakeCapture(pane_text))
        notifications: List[str] = []
        nudges: List[str] = []

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=type(
                    "Cfg",
                    (),
                    {"hang_idle_ticks": 3, "hang_stale_seconds": 300},
                )(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda tid, *_a, **_kw: notifications.append(tid),
            ),
            patch(
                "session_orchestration.watcher._send_auto_nudge",
                side_effect=lambda tid, *_a, **_kw: nudges.append(tid),
            ),
        ):
            watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["nudge_count"] == 0
        assert notifications == []
        assert nudges == []

    def test_not_eligible_when_pane_hash_changed(self):
        now = time.time()
        row = {
            "idle_ticks": 3,
            "last_output_ts": now - 301,
            "nudge_count": 0,
        }

        assert not _is_stale_frozen_eligible(
            row,
            previous_pane_hash=_pane_hash("old output"),
            current_pane_hash=_pane_hash("new output"),
            pane_text="new output",
            adapter=FakeAdapter(SessionLifecycle.RUNNING),
            hang_idle_ticks=3,
            hang_stale_seconds=300,
            now=now,
        )

    def test_omp_action_not_eligible_when_episode_already_nudged(self):
        now = time.time()
        pane_text = "unchanged pane output"
        pane_hash = _pane_hash(pane_text)
        row = {
            "idle_ticks": 3,
            "last_output_ts": now - 301,
            "nudge_count": 1,
        }

        assert _is_stale_frozen_eligible(
            row,
            previous_pane_hash=pane_hash,
            current_pane_hash=pane_hash,
            pane_text=pane_text,
            adapter=FakeAdapter(SessionLifecycle.RUNNING),
            hang_idle_ticks=3,
            hang_stale_seconds=300,
            now=now,
        )
        assert not _is_omp_nudge_checkin_eligible(
            row,
            previous_pane_hash=pane_hash,
            current_pane_hash=pane_hash,
            pane_text=pane_text,
            adapter=FakeAdapter(SessionLifecycle.RUNNING),
            hang_idle_ticks=3,
            hang_stale_seconds=300,
            now=now,
        )

    def test_on_hang_skips_actions_but_reconciles_when_episode_already_nudged(
        self, registry
    ):
        now = time.time()
        pane_text = "unchanged pane output"
        pane_hash = _pane_hash(pane_text)
        row = {
            "task_id": "t-stale-already-nudged",
            "agent": "fake",
            "state": "RUNNING",
            "idle_ticks": 3,
            "last_output_ts": now - 301,
            "last_pane_hash": pane_hash,
            "nudge_count": 1,
        }
        notifications: List[str] = []
        nudges: List[str] = []

        with (
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda tid, *_a, **_kw: notifications.append(tid),
            ),
            patch(
                "session_orchestration.watcher._send_auto_nudge",
                side_effect=lambda tid, *_a, **_kw: nudges.append(tid),
            ),
            patch(
                "session_orchestration.feed.reconcile_attention_digest",
                return_value={"status": "unchanged"},
            ) as reconcile_digest,
        ):
            _on_hang(
                "t-stale-already-nudged",
                row,
                registry=registry,
                adapter=FakeAdapter(SessionLifecycle.RUNNING),
                pane_text=pane_text,
                previous_pane_hash=pane_hash,
                current_pane_hash=pane_hash,
            )

        assert notifications == []
        assert nudges == []
        reconcile_digest.assert_called_once_with(registry)
