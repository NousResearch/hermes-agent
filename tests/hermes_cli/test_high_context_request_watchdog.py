import json
from types import SimpleNamespace


def _agent(*, queued_steer_count=0):
    return SimpleNamespace(
        session_id="session-high-context",
        model="gpt-5.5",
        provider="openai-codex",
        platform="cli",
        _pending_steer_status_count=lambda: queued_steer_count,
    )


def _thresholds():
    from agent.request_watchdog import RequestWatchdogThresholds

    return RequestWatchdogThresholds(
        normal_alert_seconds=90.0,
        high_context_alert_seconds=10.0,
        terminal_recovery_seconds=30.0,
        poll_interval_seconds=999.0,
        high_context_tokens=200_000,
    )


def test_watchdog_status_is_written_for_long_high_context_request(monkeypatch):
    from agent.request_watchdog import (
        poll_request_watchdog,
        start_request_watchdog,
    )

    updates = []
    monkeypatch.setattr(
        "hermes_cli.active_sessions.update_active_session_metadata",
        lambda **kwargs: updates.append(kwargs) or 1,
    )

    record = start_request_watchdog(
        _agent(),
        request_id="turn-1:api:1",
        api_call_count=1,
        estimated_context_tokens=240_000,
        now=100.0,
        thresholds=_thresholds(),
        start_monitor=False,
    )
    status = poll_request_watchdog(
        record,
        now=111.0,
        thresholds=_thresholds(),
        write_status=True,
    )

    assert status["status"] == "waiting"
    assert status["high_context"] is True
    assert "active model request waiting" in status["message"]
    assert updates[-1]["session_id"] == "session-high-context"
    assert updates[-1]["metadata"]["model_request_status"] == "waiting"
    assert updates[-1]["metadata"]["model_request_high_context"] is True


def test_stream_events_keep_request_from_being_classified_as_stuck():
    from agent.request_watchdog import (
        mark_request_watchdog_event,
        poll_request_watchdog,
        start_request_watchdog,
    )

    record = start_request_watchdog(
        _agent(),
        request_id="turn-1:api:2",
        api_call_count=2,
        estimated_context_tokens=260_000,
        now=100.0,
        thresholds=_thresholds(),
        start_monitor=False,
    )
    mark_request_watchdog_event(record, now=108.0, byte_count=42)

    status = poll_request_watchdog(record, now=111.0, thresholds=_thresholds())

    assert status["status"] == "healthy"
    assert status["seconds_since_event"] == 3.0


def test_stream_events_clear_previous_waiting_status(monkeypatch):
    from agent.request_watchdog import (
        mark_request_watchdog_event,
        poll_request_watchdog,
        start_request_watchdog,
    )

    updates = []
    monkeypatch.setattr(
        "hermes_cli.active_sessions.update_active_session_metadata",
        lambda **kwargs: updates.append(kwargs) or 1,
    )

    record = start_request_watchdog(
        _agent(),
        request_id="turn-1:api:2b",
        api_call_count=2,
        estimated_context_tokens=260_000,
        now=100.0,
        thresholds=_thresholds(),
        start_monitor=False,
    )
    waiting = poll_request_watchdog(
        record,
        now=111.0,
        thresholds=_thresholds(),
        write_status=True,
    )
    mark_request_watchdog_event(record, now=112.0, byte_count=42)
    healthy = poll_request_watchdog(
        record,
        now=113.0,
        thresholds=_thresholds(),
        write_status=True,
    )

    assert waiting["status"] == "waiting"
    assert healthy["status"] == "healthy"
    assert updates[-1]["metadata"]["model_request_status"] is None


def test_queued_steer_status_explains_it_cannot_land_before_tool_boundary(monkeypatch):
    from agent.request_watchdog import poll_request_watchdog, start_request_watchdog

    updates = []
    monkeypatch.setattr(
        "hermes_cli.active_sessions.update_active_session_metadata",
        lambda **kwargs: updates.append(kwargs) or 1,
    )
    record = start_request_watchdog(
        _agent(queued_steer_count=2),
        request_id="turn-1:api:3",
        api_call_count=3,
        estimated_context_tokens=250_000,
        now=100.0,
        thresholds=_thresholds(),
        start_monitor=False,
    )

    status = poll_request_watchdog(
        record,
        now=111.0,
        thresholds=_thresholds(),
        write_status=True,
    )

    assert status["queued_steer_count"] == 2
    assert "steer will queue until tool boundary" in status["message"]
    assert "private" not in json.dumps(updates)
    assert updates[-1]["metadata"]["model_request_queued_steer_count"] == 2


def test_terminal_recovery_writes_compact_state_without_mass_closing_sessions(tmp_path):
    from agent.request_watchdog import (
        poll_request_watchdog,
        start_request_watchdog,
        write_recoverable_turn_state,
    )

    record = start_request_watchdog(
        _agent(),
        request_id="turn-1:api:4",
        api_call_count=4,
        estimated_context_tokens=250_000,
        now=100.0,
        thresholds=_thresholds(),
        start_monitor=False,
    )
    status = poll_request_watchdog(record, now=131.0, thresholds=_thresholds())
    recovery = write_recoverable_turn_state(
        record,
        status=status,
        directory=tmp_path,
    )

    data = json.loads(recovery["path"].read_text(encoding="utf-8"))
    assert status["status"] == "terminal_recovery_needed"
    assert recovery["session_id"] == "session-high-context"
    assert recovery["mass_close_sessions"] is False
    assert recovery["end_session_reason"] == "high_context_request_stalled"
    assert recovery["recommended_action"] == "bounded_recovery_prompt"
    assert data["session_id"] == "session-high-context"
    assert data["recommended_action"] == "bounded_recovery_prompt"
    assert "resume_prompt" in data
    assert "inspect current DB/log/repo state first" in data["resume_prompt"]
    assert "Do not replay broad history" in data["resume_prompt"]
    assert "terminal-only write/test mode" in data["resume_prompt"]
    assert "compact final/blocked answer mode" in data["resume_prompt"]
    assert "Keep gpt-5.5" in data["resume_prompt"]
    assert "mass-close" not in data["resume_prompt"].lower()
    assert "all_sessions" not in data


def test_repeated_stale_closeout_recovery_is_policy_aware(tmp_path):
    from agent.request_watchdog import (
        poll_request_watchdog,
        start_request_watchdog,
        write_recoverable_turn_state,
    )

    record = start_request_watchdog(
        _agent(),
        request_id="turn-1:api:5",
        api_call_count=5,
        estimated_context_tokens=275_000,
        now=100.0,
        thresholds=_thresholds(),
        start_monitor=False,
    )
    status = poll_request_watchdog(record, now=131.0, thresholds=_thresholds())
    status["closeout_only"] = True
    status["repeated_stale_call_count"] = 2
    status["fixed_model_policy"] = True
    recovery = write_recoverable_turn_state(
        record,
        status=status,
        directory=tmp_path,
    )

    data = json.loads(recovery["path"].read_text(encoding="utf-8"))

    assert recovery["recommended_action"] == "compact_finalization_prompt"
    assert data["recommended_action"] == "compact_finalization_prompt"
    assert data["repeated_stale_call_count"] == 2
    assert "Do not retry the same huge request unchanged" in data["resume_prompt"]
    assert "model switch requires explicit user approval" in data["resume_prompt"]
    assert "gpt-5.4" not in data["resume_prompt"]


def test_closeout_phase_uses_bounded_mode_before_another_high_context_request():
    from agent.request_watchdog import should_use_bounded_closeout_mode

    assert should_use_bounded_closeout_mode(
        estimated_context_tokens=220_000,
        remaining_work_kind="review/report/merge monitoring",
        high_context_tokens=200_000,
    )
    assert not should_use_bounded_closeout_mode(
        estimated_context_tokens=220_000,
        remaining_work_kind="implementation",
        high_context_tokens=200_000,
    )
    assert not should_use_bounded_closeout_mode(
        estimated_context_tokens=50_000,
        remaining_work_kind="review",
        high_context_tokens=200_000,
    )


def test_active_session_and_dashboard_status_stay_value_free():
    from hermes_cli.active_sessions import _safe_status_metadata_update
    from hermes_cli.web_server import _attach_session_status_evidence

    safe, _remove = _safe_status_metadata_update(
        {
            "model_request_status": "waiting",
            "model_request_status_message": "active model request waiting; steer will queue until tool boundary.",
            "model_request_high_context": True,
            "model_request_queued_steer_count": 2,
            "queued_steer_text": "do not expose this",
        }
    )
    assert safe["model_request_status"] == "waiting"
    assert safe["model_request_queued_steer_count"] == 2
    assert "queued_steer_text" not in safe

    row = _attach_session_status_evidence(
        {"id": "session-high-context", "session_id": "session-high-context"},
        registry_by_session={
            "session-high-context": {
                "session_id": "session-high-context",
                "metadata": safe,
            }
        },
    )
    assert row["model_request_status"] == "waiting"
    assert "tool boundary" in row["model_request_status_message"]
    assert "queued_steer_text" not in row

    leaky_row = _attach_session_status_evidence(
        {"id": "leaky-session", "session_id": "leaky-session"},
        registry_by_session={
            "leaky-session": {
                "session_id": "leaky-session",
                "metadata": {
                    "model_request_status": "waiting",
                    "model_request_status_message": "do not expose private queued steer text",
                },
            }
        },
    )
    assert leaky_row["model_request_status"] == "waiting"
    assert "model_request_status_message" not in leaky_row
