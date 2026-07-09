import json
from types import SimpleNamespace


def test_control_plane_degraded_ws_defers_restart_when_worker_progress_exists():
    from hermes_cli.control_plane import build_control_plane_status

    report = build_control_plane_status(
        registry_status={
            "checked": 1,
            "live": 1,
            "stale": 0,
            "entries": [
                {
                    "session_id": "session-live",
                    "runtime_status": "live",
                    "metadata": {
                        "model_request_status": "waiting",
                        "model_request_high_context": True,
                        "last_activity_age_seconds": 7,
                        "queued_steer_text": "do not expose this private steer",
                    },
                }
            ],
        },
        runtime_status={"gateway_state": "running", "active_agents": 1},
        listener_alive=True,
        ws_probe={
            "ok": False,
            "error_type": "TimeoutError",
            "raw_error": "do not expose this socket detail",
        },
        ws_health={"active_clients": 0, "stale_closed_clients": 3, "send_failures": 1},
        close_wait_count=5,
        now=1_000.0,
    )

    assert report["status"] == "degraded"
    assert report["listener_alive"] is True
    assert report["websocket"]["probe_ok"] is False
    assert report["websocket"]["stale_closed_clients"] == 3
    assert report["close_wait_count"] == 5
    assert report["active_worker_progress"] is True
    assert report["restart_guidance"] == "defer_restart_active_worker_progress"
    dumped = json.dumps(report)
    assert "private steer" not in dumped
    assert "socket detail" not in dumped


def test_control_plane_reports_queued_steer_boundary_without_prompt_text():
    from hermes_cli.control_plane import build_control_plane_status

    report = build_control_plane_status(
        registry_status={
            "checked": 1,
            "live": 1,
            "stale": 0,
            "entries": [
                {
                    "session_id": "session-steer",
                    "runtime_status": "live",
                    "metadata": {
                        "pending_steer_count": 2,
                        "pending_steer_queued": True,
                        "model_request_status": "waiting",
                        "model_request_status_message": (
                            "active model request waiting; steer will queue until tool boundary."
                        ),
                        "queued_steer_text": "secret queued message",
                    },
                }
            ],
        },
        runtime_status={"gateway_state": "running", "active_agents": 1},
        listener_alive=True,
        ws_probe={"ok": True},
        now=1_000.0,
    )

    session = report["sessions"][0]
    assert session["session_id"] == "session-steer"
    assert session["queued_steer_count"] == 2
    assert session["steer_boundary"] == "cannot_steer_until_current_tool_boundary"
    assert "secret queued message" not in json.dumps(report)


def test_control_plane_status_counts_cli_control_queue_without_prompt_text(tmp_path, monkeypatch):
    from hermes_cli import active_sessions
    import hermes_cli.control_plane as control_plane

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(control_plane, "_read_model_policy_config", lambda: {})
    monkeypatch.setattr(control_plane, "_read_session_db_models", lambda _ids: {})
    monkeypatch.setattr(control_plane, "_read_session_db_lifecycle", lambda _ids: {})

    lease, message = active_sessions.try_acquire_active_session(
        session_id="cli-queued-status",
        surface="cli",
        config={},
    )
    assert message is None
    assert lease is not None
    try:
        queued = control_plane.queue_control_plane_steer(
            agent=None,
            session_id="cli-queued-status",
            text="private queued correction",
        )
        assert queued["status"] == "queued"

        report = control_plane.build_control_plane_status(
            registry_status=active_sessions.active_session_registry_status(),
            runtime_status={"gateway_state": "stopped", "active_agents": 0},
            listener_alive=False,
            ws_probe={"ok": True},
            close_wait_count=0,
            now=1_000.0,
        )
    finally:
        lease.release()

    session = report["sessions"][0]
    assert session["session_id"] == "cli-queued-status"
    assert session["queued_steer_count"] == 1
    assert session["steer_boundary"] == "cannot_steer_until_current_tool_boundary"
    assert "private queued correction" not in json.dumps(report)


def test_control_plane_steer_queues_without_echoing_text():
    from hermes_cli.control_plane import queue_control_plane_steer

    calls = []

    class Agent:
        def steer(self, text):
            calls.append(text)
            self._pending_steer_count = 1
            return True

    result = queue_control_plane_steer(
        agent=Agent(),
        text="please inspect this private log line",
    )

    assert calls == ["please inspect this private log line"]
    assert result["status"] == "queued"
    assert result["delivery"] == "after_next_tool_boundary"
    assert result["steer_boundary"] == "cannot_steer_until_current_tool_boundary"
    assert "private log" not in json.dumps(result)


def test_control_plane_fingerprints_gateway_agent_session_ids():
    from hermes_cli.control_plane import build_control_plane_status, queue_control_plane_steer

    raw_session_id = "agent:main:telegram:dm:secret-chat"
    report = build_control_plane_status(
        registry_status={
            "checked": 1,
            "live": 1,
            "stale": 0,
            "entries": [
                {
                    "session_id": raw_session_id,
                    "runtime_status": "live",
                    "metadata": {"last_activity_age_seconds": 1},
                }
            ],
        },
        runtime_status={"gateway_state": "running", "active_agents": 0},
        listener_alive=True,
        ws_probe={"ok": True},
        now=1_000.0,
    )

    session = report["sessions"][0]
    assert "session_id" not in session
    assert len(session["session_id_fingerprint"]) == 16
    assert "secret-chat" not in json.dumps(report)

    steer = queue_control_plane_steer(
        agent=None,
        session_id=raw_session_id,
        text="safe resume",
    )
    assert "session_id" not in steer
    assert len(steer["session_id_fingerprint"]) == 16
    assert "secret-chat" not in json.dumps(steer)


def test_control_plane_flags_fixed_model_violation_from_active_metadata(monkeypatch):
    import hermes_cli.control_plane as control_plane

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model_policy": {
                "fixed_model": "gpt-5.5",
                "forbid_lower_fallback": True,
            }
        },
    )

    report = control_plane.build_control_plane_status(
        registry_status={
            "checked": 1,
            "live": 1,
            "stale": 0,
            "entries": [
                {
                    "session_id": "session-policy",
                    "runtime_status": "live",
                    "metadata": {
                        "model_request_status": "waiting",
                        "model_request_model": "gpt-5.4-mini",
                        "queued_steer_text": "do not expose this queued text",
                        "provider_payload": "do not expose this payload",
                    },
                }
            ],
        },
        runtime_status={"gateway_state": "running", "active_agents": 1},
        listener_alive=True,
        ws_probe={"ok": True},
        now=1_000.0,
    )

    session = report["sessions"][0]
    assert report["model_policy_violation"] is True
    assert report["required_model"] == "gpt-5.5"
    assert report["model_policy_recommended_action"] == "interrupt_and_restore_fixed_model"
    assert session["model_policy_violation"] is True
    assert session["required_model"] == "gpt-5.5"
    dumped = json.dumps(report)
    assert "queued text" not in dumped
    assert "payload" not in dumped


def test_control_plane_flags_fixed_model_violation_from_session_db(monkeypatch):
    from hermes_state import SessionDB
    import hermes_cli.control_plane as control_plane

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model_policy": {
                "fixed_model": "gpt-5.5",
                "forbid_lower_fallback": True,
            }
        },
    )
    db = SessionDB()
    try:
        db.create_session(session_id="db-policy", source="cli", model="gpt-5.4-mini")
    finally:
        db.close()

    report = control_plane.build_control_plane_status(
        registry_status={
            "checked": 1,
            "live": 1,
            "stale": 0,
            "entries": [
                {
                    "session_id": "db-policy",
                    "runtime_status": "live",
                    "metadata": {"last_activity_age_seconds": 1},
                }
            ],
        },
        runtime_status={"gateway_state": "running", "active_agents": 0},
        listener_alive=True,
        ws_probe={"ok": True},
        now=1_000.0,
    )

    session = report["sessions"][0]
    assert report["model_policy_violation"] is True
    assert session["model_policy_violation"] is True
    assert session["required_model"] == "gpt-5.5"


def test_control_plane_recommends_repair_for_ended_session_with_live_evidence(monkeypatch):
    import hermes_cli.control_plane as control_plane

    monkeypatch.setattr(control_plane, "_read_model_policy_config", lambda: {})
    monkeypatch.setattr(control_plane, "_read_session_db_models", lambda _ids: {})
    monkeypatch.setattr(
        control_plane,
        "_read_session_db_lifecycle",
        lambda _ids: {
            "ended-live": {
                "ended": True,
                "end_reason": "agent_close",
                "messages_after_end": True,
            }
        },
        raising=False,
    )

    report = control_plane.build_control_plane_status(
        registry_status={
            "checked": 1,
            "live": 1,
            "stale": 0,
            "entries": [
                {
                    "session_id": "ended-live",
                    "runtime_status": "live",
                    "metadata": {"last_activity_age_seconds": 1},
                }
            ],
        },
        runtime_status={"gateway_state": "running", "active_agents": 0},
        listener_alive=True,
        ws_probe={"ok": True},
        now=1_000.0,
    )

    session = report["sessions"][0]
    assert session["db_lifecycle_status"] == "ended_with_live_runtime_evidence"
    assert session["db_lifecycle_evidence"] == [
        "active_runtime_lease",
        "messages_after_end",
    ]
    assert session["repair_recommendation"] == "inspect_before_reopen_or_close"


def test_runtime_control_plane_status_cli_is_value_free(capsys):
    from hermes_cli import runtime_cli

    rc = runtime_cli._cmd_control_plane_status(
        SimpleNamespace(
            session_id=None,
            json=False,
            ws_probe_url=None,
            _control_plane_status={
                "status": "degraded",
                "listener_alive": True,
                "close_wait_count": 4,
                "active_worker_progress": True,
                "restart_guidance": "defer_restart_active_worker_progress",
                "model_policy_violation": True,
                "required_model": "gpt-5.5",
                "model_policy_recommended_action": "interrupt_and_restore_fixed_model",
                "websocket": {
                    "probe_status": "failed",
                    "probe_ok": False,
                    "stale_closed_clients": 2,
                },
                "sessions": [
                    {
                        "session_id": "session-cli",
                        "runtime_status": "live",
                        "queued_steer_count": 1,
                        "steer_boundary": "cannot_steer_until_current_tool_boundary",
                        "model_policy_violation": True,
                        "required_model": "gpt-5.5",
                        "model_policy_recommended_action": "interrupt_and_restore_fixed_model",
                        "raw_prompt": "do not expose",
                    },
                    {
                        "session_id_fingerprint": "abcd1234abcd1234",
                        "runtime_status": "live",
                        "queued_steer_count": 0,
                    }
                ],
            },
        )
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "control-plane: status=degraded" in out
    assert "restart_guidance=defer_restart_active_worker_progress" in out
    assert "queued_steer_count=1" in out
    assert "model_policy_violation=True" in out
    assert "required_model=gpt-5.5" in out
    assert "recommended_action=interrupt_and_restore_fixed_model" in out
    assert "session_id_fingerprint=abcd1234abcd1234" in out
    assert "do not expose" not in out
