from pathlib import Path
from unittest.mock import patch

from agent.job_callbacks import deliver_completion
from agent.job_protocol import build_completion_envelope


def test_deliver_completion_uses_cron_delivery_for_platform_callbacks():
    envelope = build_completion_envelope(
        kind="background",
        task_id="bg_1",
        status="succeeded",
        callback={"type": "platform", "target": {"platform": "telegram", "chat_id": "456", "thread_id": "12"}},
        summary="done",
        final_output="all good",
    )

    with patch("cron.scheduler._deliver_result", return_value=None) as deliver_mock:
        error = deliver_completion(envelope)

    assert error is None
    job = deliver_mock.call_args.args[0]
    assert job["id"] == "bg_1"
    assert job["deliver"] == "telegram:456:12"
    assert job["origin"] == {"platform": "telegram", "chat_id": "456", "thread_id": "12"}
    assert deliver_mock.call_args.args[1] == "all good"


def test_deliver_completion_returns_none_for_noop_callback():
    envelope = build_completion_envelope(
        kind="delegation",
        task_id="delegate_1",
        status="queued",
        callback={"type": "none"},
        summary="queued",
    )

    assert deliver_completion(envelope) is None


def test_deliver_completion_session_callback_appends_to_transcript_and_delivers(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "sessions.json").write_text(
        """
        {
          "telegram:user:123": {
            "session_key": "telegram:user:123",
            "session_id": "sess_1",
            "created_at": "2026-04-21T00:00:00",
            "updated_at": "2026-04-21T00:00:00",
            "platform": "telegram",
            "chat_type": "dm",
            "origin": {
              "platform": "telegram",
              "chat_id": "456",
              "user_id": "123",
              "thread_id": "12"
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    envelope = build_completion_envelope(
        kind="delegation",
        task_id="delegate_1",
        status="succeeded",
        callback={"type": "session", "session_id": "sess_1", "platform": "telegram"},
        summary="done",
        final_output="delegated result",
    )

    with patch("agent.job_callbacks.get_hermes_home", return_value=tmp_path), \
         patch("hermes_state.SessionDB") as mock_db_cls, \
         patch("cron.scheduler._deliver_result", return_value=None) as deliver_mock:
        error = deliver_completion(envelope)

    assert error is None
    transcript = (sessions_dir / "sess_1.jsonl").read_text(encoding="utf-8")
    assert "delegated result" in transcript
    mock_db = mock_db_cls.return_value
    mock_db.ensure_session.assert_called_once_with("sess_1", source="callback")
    mock_db.append_message.assert_called_once()
    job = deliver_mock.call_args.args[0]
    assert job["deliver"] == "telegram:456:12"
    assert deliver_mock.call_args.args[1] == "delegated result"


def test_deliver_completion_session_callback_without_origin_only_reconciles_locally(tmp_path):
    envelope = build_completion_envelope(
        kind="delegation",
        task_id="delegate_1",
        status="succeeded",
        callback={"type": "session", "session_id": "sess_local", "platform": "cli"},
        summary="done",
        final_output="delegated result",
    )

    with patch("agent.job_callbacks.get_hermes_home", return_value=tmp_path), \
         patch("hermes_state.SessionDB"):
        error = deliver_completion(envelope)

    assert error is None
    transcript = (tmp_path / "sessions" / "sess_local.jsonl").read_text(encoding="utf-8")
    assert "delegated result" in transcript


def test_deliver_completion_session_callback_requires_session_id():
    envelope = build_completion_envelope(
        kind="delegation",
        task_id="delegate_1",
        status="failed",
        callback={"type": "session"},
        error="boom",
    )

    error = deliver_completion(envelope)
    assert error == "session callback missing session_id"
