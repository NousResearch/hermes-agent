import json
from types import SimpleNamespace



def test_runtime_control_plane_steer_queues_for_active_cli_session(tmp_path, monkeypatch, capsys):
    from hermes_cli import active_sessions, runtime_cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    lease, message = active_sessions.try_acquire_active_session(
        session_id="cli-control-session",
        surface="cli",
        config={},
    )
    assert message is None
    assert lease is not None

    try:
        rc = runtime_cli._cmd_control_plane_steer(
            SimpleNamespace(
                session_id="cli-control-session",
                message="please adjust the active turn",
            )
        )
        out = json.loads(capsys.readouterr().out)
    finally:
        lease.release()

    assert rc == 0
    assert out["status"] == "queued"
    assert out["delivery"] == "queued_until_next_boundary"
    assert out["steer_boundary"] == "cannot_steer_until_current_tool_boundary"
    assert out["queued_steer_count"] == 1
    assert out.get("reason") != "no_live_agent_control_channel"
    assert "please adjust" not in json.dumps(out)



def test_cli_control_queue_consumes_value_for_target_session_only(tmp_path, monkeypatch):
    from hermes_cli import active_sessions
    from hermes_cli.control_plane import (
        consume_control_plane_steers,
        queue_control_plane_steer,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    cli_lease, message = active_sessions.try_acquire_active_session(
        session_id="cli-control-session",
        surface="cli",
        config={},
    )
    other_lease, other_message = active_sessions.try_acquire_active_session(
        session_id="other-cli-session",
        surface="cli",
        config={},
    )
    assert message is None
    assert other_message is None

    try:
        first = queue_control_plane_steer(
            agent=None,
            session_id="cli-control-session",
            text="first correction",
        )
        second = queue_control_plane_steer(
            agent=None,
            session_id="cli-control-session",
            text="second correction",
        )
        queue_control_plane_steer(
            agent=None,
            session_id="other-cli-session",
            text="other correction",
        )

        assert first["status"] == "queued"
        assert second["status"] == "queued"
        assert second["queued_steer_count"] == 2
        assert consume_control_plane_steers("cli-control-session") == [
            "first correction",
            "second correction",
        ]
        assert consume_control_plane_steers("cli-control-session") == []
        assert consume_control_plane_steers("other-cli-session") == ["other correction"]
    finally:
        cli_lease.release()
        other_lease.release()
