"""Tests for execution-owner slash commands: /now, /blocked, /next."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str = "/now") -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._session_db = None

    mock_session_entry = MagicMock()
    mock_session_entry.session_id = "sess-123"
    mock_session_entry.session_key = "discord:u1:c1"
    mock_session_entry.created_at = MagicMock(strftime=lambda _: "2026-03-12 00:00")
    mock_session_entry.updated_at = MagicMock(strftime=lambda _: "2026-03-12 00:01")
    mock_session_entry.total_tokens = 42

    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    runner.session_store = mock_store
    runner._pending_messages = {}
    runner._ops_window_seconds = 24 * 3600
    runner._ops_stall_threshold_seconds = 180
    runner._ops_runs = deque()
    runner._ops_active_runs = {}
    runner._ops_recent_debug_limit = 5
    runner._ops_alerted_run_ids = set()
    runner._ops_stall_recovery_notices_enabled = True
    runner._ops_stall_alert_home_fanout_main_enabled = False

    return runner


def _seed_artifacts(root):
    reports = root / "reports"
    kb = root / "kb"
    reports.mkdir(parents=True, exist_ok=True)
    kb.mkdir(parents=True, exist_ok=True)

    (reports / "github_sync_latest.json").write_text(
        json.dumps(
            {
                "counts": {
                    "owner_open_prs": 2,
                    "review_requested_prs": 1,
                    "assigned_open_issues": 0,
                },
                "ingest": {"event_source": "polling"},
            }
        ),
        encoding="utf-8",
    )

    (kb / "twitter_bookmarks_state.json").write_text(
        json.dumps(
            {
                "fetched_items": 20,
                "inserted": 1,
                "updated": 2,
                "freshness": {"stale_minutes": 31, "sla_met": False, "alert_triggered": True},
                "error": "rate-limited",
            }
        ),
        encoding="utf-8",
    )

    (kb / "twitter_vector_state.json").write_text(
        json.dumps({"errors": 1}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_now_blocked_next_handlers(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    with patch("gateway.run._hermes_home", tmp_path):
        now_text = await runner._handle_now_command(_make_event("/now"))
        blocked_text = await runner._handle_blocked_command(_make_event("/blocked"))
        next_text = await runner._handle_next_command(_make_event("/next"))

    assert "Execution Owner" in now_text
    assert "Cross-app bridge suggestions" in now_text
    assert "`source=polling" in now_text
    assert "polling fallback" in blocked_text.lower()
    assert "Bookmarks" in blocked_text
    assert "`source=polling" in blocked_text
    assert "Next Actions" in next_text
    assert "`source=polling" in next_text


@pytest.mark.asyncio
async def test_ops_handler_reports_live_metrics(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    now_ts = time.time()
    runner._ops_runs.extend(
        [
            {"ts": now_ts - 20, "outcome": "success", "latency_ms": 900, "thread": False},
            {"ts": now_ts - 10, "outcome": "success", "latency_ms": 1200, "thread": True},
            {"ts": now_ts - 5, "outcome": "error", "latency_ms": 500, "thread": False},
        ]
    )
    runner._ops_active_runs = {
        "main": {"started_at": now_ts - 30, "thread": False},
        "thread": {"started_at": now_ts - 240, "thread": True},
    }
    runner._pending_messages = {"s1": "queued work"}

    with patch("gateway.run._hermes_home", tmp_path):
        ops_text = await runner._handle_ops_command(_make_event("/ops"))

    assert "Ops Board" in ops_text
    assert "Started: 3" in ops_text
    assert "Completed: 2" in ops_text
    assert "Failed: 1" in ops_text
    assert "Queue depth: 1" in ops_text
    assert "Active stalled (>180s): 1" in ops_text
    assert "Snapshot:" in ops_text


@pytest.mark.asyncio
async def test_ops_handler_supports_debug_run_lookup(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    now_ts = time.time()
    runner._ops_runs.extend(
        [
            {
                "ts": now_ts - 12,
                "started_at": now_ts - 20,
                "finished_at": now_ts - 12,
                "outcome": "success",
                "latency_ms": 820,
                "thread": False,
                "run_id": "run-s1",
                "session_key": "discord:u1:c1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": None,
            }
        ]
    )

    with patch("gateway.run._hermes_home", tmp_path):
        debug_text = await runner._handle_ops_command(_make_event("/ops debug run-s1"))

    assert "Ops Run Debug" in debug_text
    assert "run-s1" in debug_text
    assert "Outcome: success" in debug_text


@pytest.mark.asyncio
async def test_ops_handler_supports_debug_latest_alias(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    now_ts = time.time()
    runner._ops_runs.extend(
        [
            {
                "ts": now_ts - 20,
                "started_at": now_ts - 27,
                "finished_at": now_ts - 20,
                "outcome": "success",
                "latency_ms": 400,
                "thread": False,
                "run_id": "run-old",
                "session_key": "discord:u1:c1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": None,
            },
            {
                "ts": now_ts - 5,
                "started_at": now_ts - 11,
                "finished_at": now_ts - 5,
                "outcome": "error",
                "latency_ms": 900,
                "thread": True,
                "run_id": "run-new",
                "session_key": "discord:u1:t1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": "t1",
            },
        ]
    )

    with patch("gateway.run._hermes_home", tmp_path):
        debug_text = await runner._handle_ops_command(_make_event("/ops debug latest"))

    assert "Ops Run Debug" in debug_text
    assert "run-new" in debug_text
    assert "Outcome: error" in debug_text


@pytest.mark.asyncio
async def test_ops_handler_supports_debug_latest_main_alias(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    now_ts = time.time()
    runner._ops_runs.extend(
        [
            {
                "ts": now_ts - 20,
                "started_at": now_ts - 28,
                "finished_at": now_ts - 20,
                "outcome": "success",
                "latency_ms": 450,
                "thread": False,
                "run_id": "run-main-old",
                "session_key": "discord:u1:c1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": None,
            },
            {
                "ts": now_ts - 9,
                "started_at": now_ts - 16,
                "finished_at": now_ts - 9,
                "outcome": "success",
                "latency_ms": 500,
                "thread": False,
                "run_id": "run-main-new",
                "session_key": "discord:u1:c1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": None,
            },
            {
                "ts": now_ts - 4,
                "started_at": now_ts - 7,
                "finished_at": now_ts - 4,
                "outcome": "error",
                "latency_ms": 900,
                "thread": True,
                "run_id": "run-thread-newest",
                "session_key": "discord:u1:t1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": "t1",
            },
        ]
    )

    with patch("gateway.run._hermes_home", tmp_path):
        debug_text = await runner._handle_ops_command(_make_event("/ops debug latest-main"))

    assert "Ops Run Debug" in debug_text
    assert "run-main-new" in debug_text
    assert "Thread: no" in debug_text


@pytest.mark.asyncio
async def test_ops_handler_supports_debug_latest_thread_alias(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    now_ts = time.time()
    runner._ops_runs.extend(
        [
            {
                "ts": now_ts - 20,
                "started_at": now_ts - 28,
                "finished_at": now_ts - 20,
                "outcome": "success",
                "latency_ms": 450,
                "thread": False,
                "run_id": "run-main-old",
                "session_key": "discord:u1:c1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": None,
            },
            {
                "ts": now_ts - 9,
                "started_at": now_ts - 16,
                "finished_at": now_ts - 9,
                "outcome": "error",
                "latency_ms": 910,
                "thread": True,
                "run_id": "run-thread-new",
                "session_key": "discord:u1:t1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": "t1",
            },
        ]
    )

    with patch("gateway.run._hermes_home", tmp_path):
        debug_text = await runner._handle_ops_command(_make_event("/ops debug latest-thread"))

    assert "Ops Run Debug" in debug_text
    assert "run-thread-new" in debug_text
    assert "Thread: yes" in debug_text


@pytest.mark.asyncio
async def test_ops_handler_latest_main_prefers_active_run(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    now_ts = time.time()
    runner._ops_runs.extend(
        [
            {
                "ts": now_ts - 14,
                "started_at": now_ts - 20,
                "finished_at": now_ts - 14,
                "outcome": "success",
                "latency_ms": 500,
                "thread": False,
                "run_id": "run-main-finished",
                "session_key": "discord:u1:c1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": None,
            }
        ]
    )
    runner._ops_active_runs = {
        "discord:u1:c1": {
            "run_id": "run-main-active",
            "session_key": "discord:u1:c1",
            "started_at": now_ts - 3,
            "platform": "discord",
            "chat_id": "c1",
            "thread": False,
            "thread_id": None,
        }
    }

    with patch("gateway.run._hermes_home", tmp_path):
        debug_text = await runner._handle_ops_command(_make_event("/ops debug latest-main"))

    assert "Ops Run Debug" in debug_text
    assert "run-main-active" in debug_text
    assert "Status: active" in debug_text


@pytest.mark.asyncio
async def test_ops_handler_latest_thread_reports_no_matching_runs(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    now_ts = time.time()
    runner._ops_runs.extend(
        [
            {
                "ts": now_ts - 5,
                "started_at": now_ts - 10,
                "finished_at": now_ts - 5,
                "outcome": "success",
                "latency_ms": 420,
                "thread": False,
                "run_id": "run-main-only",
                "session_key": "discord:u1:c1",
                "platform": "discord",
                "chat_id": "c1",
                "thread_id": None,
            }
        ]
    )

    with patch("gateway.run._hermes_home", tmp_path):
        debug_text = await runner._handle_ops_command(_make_event("/ops debug latest-thread"))

    assert "No thread runs available" in debug_text


@pytest.mark.asyncio
async def test_ops_handler_includes_recent_run_ids(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    now_ts = time.time()
    runner._ops_runs.extend(
        [
            {"ts": now_ts - 20, "outcome": "success", "latency_ms": 900, "thread": False, "run_id": "run-a"},
            {"ts": now_ts - 10, "outcome": "success", "latency_ms": 1200, "thread": True, "run_id": "run-b"},
        ]
    )

    with patch("gateway.run._hermes_home", tmp_path):
        ops_text = await runner._handle_ops_command(_make_event("/ops"))

    assert "Recent runs" in ops_text
    assert "run-a" in ops_text
    assert "run-b" in ops_text


@pytest.mark.asyncio
async def test_snapshot_reports_stale_artifact_age(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    reports_file = tmp_path / "reports" / "github_sync_latest.json"
    kb_file = tmp_path / "kb" / "twitter_bookmarks_state.json"
    vector_file = tmp_path / "kb" / "twitter_vector_state.json"

    stale_epoch = time.time() - (31 * 60)
    os.utime(reports_file, (stale_epoch, stale_epoch))
    os.utime(kb_file, (stale_epoch, stale_epoch))
    os.utime(vector_file, (stale_epoch, stale_epoch))

    with patch("gateway.run._hermes_home", tmp_path):
        blocked_text = await runner._handle_blocked_command(_make_event("/blocked"))
        next_text = await runner._handle_next_command(_make_event("/next"))

    assert "snapshot is stale" in blocked_text.lower()
    assert "Fix delayed sync job" in next_text


@pytest.mark.asyncio
async def test_snapshot_reports_missing_artifacts(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    (tmp_path / "kb" / "twitter_vector_state.json").unlink()

    with patch("gateway.run._hermes_home", tmp_path):
        blocked_text = await runner._handle_blocked_command(_make_event("/blocked"))
        next_text = await runner._handle_next_command(_make_event("/next"))

    assert "missing/unreadable artifacts" in blocked_text.lower()
    assert "twitter_vector_state.json" in blocked_text
    assert "Repair missing artifact generation" in next_text


@pytest.mark.asyncio
async def test_snapshot_flags_ingest_reliability_contract_gaps(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    (tmp_path / "reports" / "github_sync_latest.json").write_text(
        json.dumps(
            {
                "counts": {"owner_open_prs": 0, "review_requested_prs": 0, "assigned_open_issues": 0},
                "ingest": {
                    "event_source": "webhook",
                    "replay_backlog": 3,
                    "dlq_depth": 2,
                    "checkpoint_age_minutes": 29,
                    "idempotency": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )

    with patch("gateway.run._hermes_home", tmp_path):
        blocked_text = await runner._handle_blocked_command(_make_event("/blocked"))
        next_text = await runner._handle_next_command(_make_event("/next"))

    assert "Replay backlog pending: 3 events" in blocked_text
    assert "DLQ has pending failed events: 2" in blocked_text
    assert "Enable ingest idempotency keys" in next_text
    assert "Process DLQ items" in next_text


@pytest.mark.asyncio
async def test_snapshot_handles_malformed_numeric_fields(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    (tmp_path / "reports" / "github_sync_latest.json").write_text(
        json.dumps(
            {
                "counts": {
                    "owner_open_prs": "n/a",
                    "review_requested_prs": "unknown",
                    "assigned_open_issues": None,
                },
                "ingest": {"event_source": "polling"},
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "kb" / "twitter_bookmarks_state.json").write_text(
        json.dumps(
            {
                "fetched_items": 20,
                "inserted": "oops",
                "updated": "oops",
                "freshness": {"stale_minutes": 31, "sla_met": False, "alert_triggered": True},
            }
        ),
        encoding="utf-8",
    )

    with patch("gateway.run._hermes_home", tmp_path):
        now_text = await runner._handle_now_command(_make_event("/now"))
        next_text = await runner._handle_next_command(_make_event("/next"))

    assert "Execution Owner" in now_text
    assert "Next Actions" in next_text


@pytest.mark.asyncio
async def test_snapshot_flags_projects_v2_field_drift(tmp_path):
    runner = _make_runner()
    _seed_artifacts(tmp_path)

    (tmp_path / "reports" / "github_sync_latest.json").write_text(
        json.dumps(
            {
                "counts": {"owner_open_prs": 0, "review_requested_prs": 0, "assigned_open_issues": 0},
                "ingest": {
                    "event_source": "webhook",
                    "processor_checkpoint": "ckpt-1",
                    "last_event_id": "evt-99",
                    "checkpoint_age_minutes": 1,
                    "idempotency": {"enabled": True},
                },
                "projects_v2": {
                    "health": "degraded",
                    "field_drift_count": 2,
                    "missing_fields": ["Priority", "Size"],
                    "unmapped_labels": 4,
                    "canonical_fields": ["Status", "Priority", "Size"],
                },
            }
        ),
        encoding="utf-8",
    )

    with patch("gateway.run._hermes_home", tmp_path):
        now_text = await runner._handle_now_command(_make_event("/now"))
        blocked_text = await runner._handle_blocked_command(_make_event("/blocked"))
        next_text = await runner._handle_next_command(_make_event("/next"))

    assert "Projects v2: health=degraded, field_drift=2, unmapped_labels=4" in now_text
    assert "Projects v2 field drift detected" in blocked_text
    assert "Label taxonomy drift: 4 unmapped labels" in blocked_text
    assert "Apply canonical Projects v2 field schema" in next_text
    assert "Map legacy labels into Projects v2 fields" in next_text


@pytest.mark.asyncio
async def test_ops_stall_alerts_emit_once_per_run():
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    runner.adapters = {Platform.DISCORD: adapter}

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
    )

    await runner._send_ops_stall_alert(run_id="run-stall", session_key="discord:u1:c1", source=source, elapsed_seconds=190)
    await runner._send_ops_stall_alert(run_id="run-stall", session_key="discord:u1:c1", source=source, elapsed_seconds=210)

    assert adapter.send.await_count == 1


@pytest.mark.asyncio
async def test_ops_stall_alert_fanout_to_home_channel_for_main_runs_only():
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._ops_stall_alert_home_fanout_main_enabled = True

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="chan-main",
        user_name="tester",
    )

    with patch.dict(os.environ, {"DISCORD_HOME_CHANNEL": "chan-home"}, clear=False):
        await runner._send_ops_stall_alert(run_id="run-main", session_key="discord:u1:main", source=source, elapsed_seconds=190)

    sent_chat_ids = [call.kwargs.get("chat_id") for call in adapter.send.await_args_list]
    assert sent_chat_ids == ["chan-main", "chan-home"]


@pytest.mark.asyncio
async def test_ops_stall_alert_does_not_fanout_thread_runs():
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._ops_stall_alert_home_fanout_main_enabled = True

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="chan-main",
        thread_id="thread-9",
        chat_type="thread",
        user_name="tester",
    )

    with patch.dict(os.environ, {"DISCORD_HOME_CHANNEL": "chan-home"}, clear=False):
        await runner._send_ops_stall_alert(run_id="run-thread", session_key="discord:u1:thread", source=source, elapsed_seconds=190)

    sent_chat_ids = [call.kwargs.get("chat_id") for call in adapter.send.await_args_list]
    assert sent_chat_ids == ["chan-main"]


@pytest.mark.asyncio
async def test_ops_stall_recovery_notice_emits_and_clears_alert_state():
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    runner.adapters = {Platform.DISCORD: adapter}

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
    )

    runner._ops_alerted_run_ids = {"run-stall"}

    await runner._send_ops_stall_recovered_notice(
        run_id="run-stall",
        session_key="discord:u1:c1",
        source=source,
        elapsed_seconds=240,
        outcome="success",
    )

    assert "run-stall" not in runner._ops_alerted_run_ids
    assert adapter.send.await_count == 1
    assert "recovered" in adapter.send.await_args.kwargs.get("content", "").lower()


@pytest.mark.asyncio
async def test_ops_stall_recovery_notice_fanout_to_home_channel_for_main_runs_only():
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._ops_stall_alert_home_fanout_main_enabled = True

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="chan-main",
        user_name="tester",
    )

    runner._ops_alerted_run_ids = {"run-main"}

    with patch.dict(os.environ, {"DISCORD_HOME_CHANNEL": "chan-home"}, clear=False):
        await runner._send_ops_stall_recovered_notice(
            run_id="run-main",
            session_key="discord:u1:main",
            source=source,
            elapsed_seconds=220,
            outcome="success",
        )

    sent_chat_ids = [call.kwargs.get("chat_id") for call in adapter.send.await_args_list]
    assert sent_chat_ids == ["chan-main", "chan-home"]


@pytest.mark.asyncio
async def test_ops_stall_recovery_notice_does_not_fanout_thread_runs():
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._ops_stall_alert_home_fanout_main_enabled = True

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="chan-main",
        thread_id="thread-1",
        chat_type="thread",
        user_name="tester",
    )

    runner._ops_alerted_run_ids = {"run-thread"}

    with patch.dict(os.environ, {"DISCORD_HOME_CHANNEL": "chan-home"}, clear=False):
        await runner._send_ops_stall_recovered_notice(
            run_id="run-thread",
            session_key="discord:u1:thread",
            source=source,
            elapsed_seconds=220,
            outcome="success",
        )

    sent_chat_ids = [call.kwargs.get("chat_id") for call in adapter.send.await_args_list]
    assert sent_chat_ids == ["chan-main"]


@pytest.mark.asyncio
async def test_help_includes_ops_command():
    runner = _make_runner()
    help_text = await runner._handle_help_command(_make_event("/help"))
    assert "`/ops`" in help_text


def test_help_and_known_commands_include_exec_owner_commands():
    from gateway.run import GatewayRunner
    import inspect

    source = inspect.getsource(GatewayRunner._handle_message)
    assert '"ops"' in source
    assert '"now"' in source
    assert '"blocked"' in source
    assert '"next"' in source
