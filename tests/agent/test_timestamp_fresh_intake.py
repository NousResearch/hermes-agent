"""Fresh non-gateway timestamp intake follows the pinned gateway admission contract."""

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest


T0 = datetime(2026, 8, 20, 12, 0, tzinfo=ZoneInfo("UTC")).timestamp()
T1 = T0 + 61


def _chat_response(captured):
    def respond(kwargs):
        captured.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop",
            )],
            model="test-model", usage=None,
        )
    return respond


@pytest.mark.parametrize("enabled", [False, True])
def test_staged_cli_admission_timestamp_is_reused_for_wire_sidecar_and_persistence(
    tmp_path, monkeypatch, enabled,
):
    """A real staged CLI dict keeps its accepted time when the loop starts later."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    captured = []
    try:
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="chat_completions", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, session_db=db, session_id=f"staged-cli-{enabled}",
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent._disable_streaming = True
        # This is the exact CLI staging shape; run_conversation intentionally receives no timestamp.
        agent._pending_cli_user_message = {"role": "user", "content": "current question", "timestamp": T0}
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
            "message_timestamps": {"enabled": enabled},
        })
        monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
        monkeypatch.setattr("agent.turn_context.time.time", lambda: T1)
        monkeypatch.setattr("agent.turn_context._collect_pre_llm_call_context", lambda *args, **kwargs: "plugin context")
        monkeypatch.setattr(agent, "_interruptible_api_call", _chat_response(captured))

        result = agent.run_conversation("current question", task_id="staged-cli")

        rendered = "[Thu 2026-08-20 12:00:00 UTC] current question" if enabled else "current question"
        expected_sidecar = rendered + "\n\nplugin context"
        assert result["completed"] is True
        assert [m["content"] for m in captured[0]["messages"] if m["role"] == "user"] == [expected_sidecar]
        current = next(message for message in result["messages"] if message["role"] == "user")
        assert current["content"] == "current question"
        assert current["timestamp"] == T0
        assert current["api_content"] == expected_sidecar
    finally:
        db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        stored = reopened.get_messages_as_conversation(f"staged-cli-{enabled}")
        current = next(message for message in stored if message["role"] == "user")
        assert current["content"] == "current question"
        assert current["timestamp"] == T0
        assert current["api_content"] == expected_sidecar
    finally:
        reopened.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_fresh_prefixed_text_uses_supplied_time_and_persists_clean_body(tmp_path, monkeypatch, enabled):
    """Fresh intake matches gateway prefix stripping without historical-note cleanup."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    supplied = T0 + 120
    incoming = "[2026-08-20T12:00:00+00:00] actual question"
    clean = "actual question"
    db = SessionDB(db_path=tmp_path / "state.db")
    captured = []
    try:
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="chat_completions", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, session_db=db, session_id=f"fresh-prefix-{enabled}",
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent._disable_streaming = True
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
            "message_timestamps": {"enabled": enabled},
        })
        monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
        monkeypatch.setattr(agent, "_interruptible_api_call", _chat_response(captured))

        result = agent.run_conversation(incoming, task_id="fresh-prefix", persist_user_timestamp=supplied)

        wire = "[Thu 2026-08-20 12:02:00 UTC] actual question" if enabled else clean
        assert [m["content"] for m in captured[0]["messages"] if m["role"] == "user"] == [wire]
        current = next(message for message in result["messages"] if message["role"] == "user")
        assert current["content"] == clean
        assert current["timestamp"] == supplied
        stored = db.get_messages_as_conversation(f"fresh-prefix-{enabled}")
        persisted = next(message for message in stored if message["role"] == "user")
        assert persisted["content"] == clean
        assert persisted["timestamp"] == supplied
    finally:
        db.close()
