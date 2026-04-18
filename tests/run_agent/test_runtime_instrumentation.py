from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from run_agent import AIAgent
from agent.runtime_events import (
    ARTIFACT_CREATED,
    FINAL_RESPONSE_DELIVERED,
    INTERRUPTION_CREATED,
    INTERRUPTION_RESUMED,
    STEP_COMPLETED,
    STEP_STARTED,
    TOOL_CALL_COMPLETED,
    TOOL_CALL_STARTED,
)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_runtime_helpers_persist_run_lifecycle(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="runtime-helper-session",
            )

        agent._runtime_start_turn("finish the task", "task-123")
        run_id = agent._active_run_id
        assert run_id

        agent._runtime_mark_waiting_for_tool()
        agent._runtime_finish_turn(final_response="done", completed=True, interrupted=False)

        row = db._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert row["session_id"] == "runtime-helper-session"
        assert row["state"] == "completed"
        assert row["final_status"] == "completed"
    finally:
        db.close()


def test_run_conversation_persists_completed_runtime_row(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="runtime-run-session",
            )
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _mock_response(content="Final answer")

        result = agent.run_conversation("hello runtime")

        assert result["final_response"] == "Final answer"
        row = db._conn.execute(
            "SELECT * FROM runs WHERE session_id = ? ORDER BY started_at DESC LIMIT 1",
            ("runtime-run-session",),
        ).fetchone()
        assert row is not None
        assert row["state"] == "completed"
        assert row["final_status"] == "completed"

        step_rows = db._conn.execute(
            "SELECT step_type, status FROM run_steps WHERE run_id = ? ORDER BY step_index ASC",
            (row["id"],),
        ).fetchall()
        assert [(r["step_type"], r["status"]) for r in step_rows] == [
            ("model_call", "completed"),
            ("finalization", "completed"),
        ]

        event_rows = db._conn.execute(
            "SELECT event_type FROM run_events WHERE run_id = ? ORDER BY timestamp ASC",
            (row["id"],),
        ).fetchall()
        event_types = [r["event_type"] for r in event_rows]
        assert STEP_STARTED in event_types
        assert STEP_COMPLETED in event_types
        assert FINAL_RESPONSE_DELIVERED in event_types
    finally:
        db.close()


class _ToolCall:
    def __init__(self, tool_id: str, name: str, arguments: str = "{}"):
        self.id = tool_id
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)



def test_run_conversation_persists_tool_step_and_events(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent.handle_function_call", return_value='{"ok": true}'),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="runtime-tool-session",
            )
            agent.client = MagicMock()
            agent.client.chat.completions.create.side_effect = [
                _mock_response(content="", tool_calls=[_ToolCall("call-1", "web_search")]),
                _mock_response(content="done"),
            ]

            result = agent.run_conversation("search runtime")

        assert result["final_response"] == "done"
        row = db._conn.execute(
            "SELECT * FROM runs WHERE session_id = ? ORDER BY started_at DESC LIMIT 1",
            ("runtime-tool-session",),
        ).fetchone()
        assert row is not None

        step_rows = db._conn.execute(
            "SELECT step_type, status, tool_name FROM run_steps WHERE run_id = ? ORDER BY step_index ASC",
            (row["id"],),
        ).fetchall()
        assert [(r["step_type"], r["status"], r["tool_name"]) for r in step_rows] == [
            ("model_call", "completed", None),
            ("tool_execution", "completed", "web_search"),
            ("model_call", "completed", None),
            ("finalization", "completed", None),
        ]

        event_rows = db._conn.execute(
            "SELECT event_type FROM run_events WHERE run_id = ? ORDER BY timestamp ASC",
            (row["id"],),
        ).fetchall()
        event_types = [r["event_type"] for r in event_rows]
        assert TOOL_CALL_STARTED in event_types
        assert TOOL_CALL_COMPLETED in event_types
        assert FINAL_RESPONSE_DELIVERED in event_types

        timeline = db.get_run_timeline(row["id"])
        assert timeline is not None
        assert timeline["run"]["id"] == row["id"]
        assert [item["event_type"] for item in timeline["timeline"]].count(TOOL_CALL_STARTED) == 1
        assert [item["event_type"] for item in timeline["timeline"]].count(TOOL_CALL_COMPLETED) == 1
        tool_event = next(item for item in timeline["timeline"] if item["event_type"] == TOOL_CALL_COMPLETED)
        assert tool_event["step_type"] == "tool_execution"
        assert tool_event["tool_name"] == "web_search"
    finally:
        db.close()


def test_clarify_tool_persists_waiting_human_interruption(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("clarify")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="runtime-clarify-session",
                clarify_callback=lambda question, choices: "Blue",
            )

        agent._runtime_start_turn("need a choice", "task-clarify")
        run_id = agent._active_run_id
        assistant_message = SimpleNamespace(
            tool_calls=[_ToolCall("call-clarify", "clarify", '{"question":"What color?","choices":["Blue","Red"]}')]
        )
        messages = []

        agent._execute_tool_calls(assistant_message, messages, "task-clarify")

        interruption_rows = db._conn.execute(
            "SELECT reason_type, waiting_on, status FROM interruptions WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        assert [(r["reason_type"], r["waiting_on"], r["status"]) for r in interruption_rows] == [
            ("waiting_user", "clarify", "resumed"),
        ]

        run_row = db._conn.execute("SELECT state, next_step FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["state"] == "executing"
        assert run_row["next_step"] == "run_again"

        event_rows = db._conn.execute(
            "SELECT event_type FROM run_events WHERE run_id = ? ORDER BY timestamp ASC",
            (run_id,),
        ).fetchall()
        event_types = [r["event_type"] for r in event_rows]
        assert INTERRUPTION_CREATED in event_types
        assert INTERRUPTION_RESUMED in event_types
    finally:
        db.close()


def test_run_conversation_registers_trajectory_artifact_before_run_finishes(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent._save_trajectory_to_file", return_value=str(tmp_path / "trajectory_samples.jsonl")),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="runtime-trajectory-session",
                save_trajectories=True,
            )
            agent.client = MagicMock()
            agent.client.chat.completions.create.return_value = _mock_response(content="Final answer")

            result = agent.run_conversation("hello trajectory artifact")

        assert result["final_response"] == "Final answer"
        run_row = db._conn.execute(
            "SELECT * FROM runs WHERE session_id = ? ORDER BY started_at DESC LIMIT 1",
            ("runtime-trajectory-session",),
        ).fetchone()
        assert run_row is not None

        artifact_rows = db._conn.execute(
            "SELECT artifact_type, path_or_ref, produced_by, purpose, is_final, delivered FROM artifacts WHERE run_id = ? ORDER BY created_at ASC",
            (run_row["id"],),
        ).fetchall()
        assert [(r["artifact_type"], r["path_or_ref"], r["produced_by"], r["purpose"], r["is_final"], r["delivered"]) for r in artifact_rows] == [
            ("trajectory", str(tmp_path / "trajectory_samples.jsonl"), "assistant", "conversation_trajectory", 1, 0),
        ]

        event_rows = db._conn.execute(
            "SELECT event_type FROM run_events WHERE run_id = ? ORDER BY timestamp ASC",
            (run_row["id"],),
        ).fetchall()
        event_types = [r["event_type"] for r in event_rows]
        assert ARTIFACT_CREATED in event_types
    finally:
        db.close()
