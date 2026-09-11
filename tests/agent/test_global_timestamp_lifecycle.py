"""Non-gateway timestamps reach the real Responses request builder."""

from copy import deepcopy
from datetime import datetime
import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest


STAMP = datetime(2026, 8, 20, 12, 0, tzinfo=ZoneInfo("UTC")).timestamp()


@pytest.mark.parametrize("enabled", [False, True])
def test_direct_agent_global_timestamp_reaches_responses_request(tmp_path, monkeypatch, enabled):
    """Replace only the provider call; persistence and request conversion are real."""
    from hermes_state import SessionDB
    from openai.types.responses import Response
    from run_agent import AIAgent

    captured = []
    db = SessionDB(db_path=tmp_path / "state.db")

    def respond(kwargs, **unused):
        captured.append(deepcopy(kwargs))
        return Response.model_validate({
            "id": "resp_test", "object": "response", "created_at": STAMP,
            "model": "test-model", "status": "completed",
            "output": [{
                "type": "message", "id": "msg_done", "role": "assistant",
                "status": "completed", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }],
            "usage": None, "error": None, "incomplete_details": None,
            "instructions": None, "metadata": {}, "parallel_tool_calls": True,
            "temperature": None, "tool_choice": "auto", "tools": [], "top_p": None,
        })

    try:
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="codex_responses", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
            skip_memory=True, save_trajectories=False, session_db=db, session_id="timestamp-direct",
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
            "message_timestamps": {"enabled": enabled},
        })
        monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", respond)
        monkeypatch.setattr(agent, "_interruptible_api_call", respond)

        result = agent.run_conversation(
            "current question",
            conversation_history=[
                {"role": "user", "content": "[System note: Your previous turn was interrupted. Continue.] earlier question", "timestamp": STAMP},
                {"role": "assistant", "content": "earlier answer"},
            ],
            task_id="timestamp-direct",
            persist_user_timestamp=STAMP,
        )

        assert result["completed"] is True
        historical_wire = "[Thu 2026-08-20 12:00:00 UTC] earlier question" if enabled else "earlier question"
        current_wire = "[Thu 2026-08-20 12:00:00 UTC] current question" if enabled else "current question"
        assert captured[0]["input"][0]["content"] == historical_wire
        assert captured[0]["input"][-1]["content"] == current_wire
        # Rendering is confined to the request: callers retain clean canonical rows
        # for the next turn and any later persistence/reload lifecycle.
        assert result["messages"][0]["content"] == "[System note: Your previous turn was interrupted. Continue.] earlier question"
        assert result["messages"][-2]["content"] == "current question"
        persisted = db.get_messages_as_conversation("timestamp-direct")
        persisted_user = next(message for message in reversed(persisted) if message["role"] == "user")
        assert persisted_user["content"] == "current question"
        assert persisted_user["timestamp"] == STAMP
    finally:
        db.close()


def test_direct_agent_keeps_fresh_recovery_shaped_input_at_provider_boundary(tmp_path, monkeypatch):
    """Recovery cleanup is historical replay only, never fresh user intake."""
    from hermes_state import SessionDB
    from openai.types.responses import Response
    from run_agent import AIAgent

    captured = []
    db = SessionDB(db_path=tmp_path / "state.db")

    def respond(kwargs, **unused):
        captured.append(deepcopy(kwargs))
        return Response.model_validate({
            "id": "resp_fresh", "object": "response", "created_at": STAMP,
            "model": "test-model", "status": "completed",
            "output": [{
                "type": "message", "id": "msg_fresh", "role": "assistant",
                "status": "completed", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }],
            "usage": None, "error": None, "incomplete_details": None,
            "instructions": None, "metadata": {}, "parallel_tool_calls": True,
            "temperature": None, "tool_choice": "auto", "tools": [], "top_p": None,
        })

    fresh = "[System note: Your previous turn was interrupted. Continue.] actual question"
    try:
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="codex_responses", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
            skip_memory=True, save_trajectories=False, session_db=db, session_id="timestamp-fresh",
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
            "message_timestamps": {"enabled": True},
        })
        monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", respond)
        monkeypatch.setattr(agent, "_interruptible_api_call", respond)

        result = agent.run_conversation(fresh, task_id="timestamp-fresh", persist_user_timestamp=STAMP)

        assert result["completed"] is True
        assert captured[0]["input"][-1]["content"] == (
            "[Thu 2026-08-20 12:00:00 UTC] " + fresh
        )
        assert result["messages"][-2]["content"] == fresh
    finally:
        db.close()


def test_direct_agent_global_timestamp_reaches_chat_completions_request(tmp_path, monkeypatch):
    """The Chat Completions wire copy is rendered while durable rows stay canonical."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    captured = []
    db = SessionDB(db_path=tmp_path / "state.db")

    def respond(kwargs):
        captured.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop",
            )],
            model="test-model", usage=None,
        )

    try:
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="chat_completions", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
            skip_memory=True, save_trajectories=False, session_db=db, session_id="timestamp-chat",
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent._disable_streaming = True
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
            "message_timestamps": {"enabled": True},
        })
        monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
        monkeypatch.setattr(agent, "_interruptible_api_call", respond)

        result = agent.run_conversation(
            "current question",
            conversation_history=[
                {"role": "user", "content": "earlier question", "timestamp": STAMP},
                {"role": "assistant", "content": "earlier answer"},
            ],
            task_id="timestamp-chat",
            persist_user_timestamp=STAMP,
        )

        assert result["completed"] is True
        wire_users = [message["content"] for message in captured[0]["messages"] if message["role"] == "user"]
        assert wire_users == [
            "[Thu 2026-08-20 12:00:00 UTC] earlier question",
            "[Thu 2026-08-20 12:00:00 UTC] current question",
        ]
        assert result["messages"][0]["content"] == "earlier question"
        assert result["messages"][-2]["content"] == "current question"
        persisted = db.get_messages_as_conversation("timestamp-chat")
        persisted_users = [message for message in persisted if message["role"] == "user"]
        assert [message["content"] for message in persisted_users] == ["current question"]
        assert persisted_users[0]["timestamp"] == STAMP
    finally:
        db.close()


def test_direct_agent_timestamp_projection_survives_tool_continuation_and_db_reopen(tmp_path, monkeypatch):
    """Only the provider boundary is replaced; cached and durable histories stay canonical."""
    from hermes_state import SessionDB
    from openai.types.responses import Response
    from run_agent import AIAgent

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    captured, outputs = [], []
    tool_file = tmp_path / "tool.txt"
    tool_file.write_text("tool fixture", encoding="utf-8")

    def respond(kwargs, **unused):
        captured.append(deepcopy(kwargs))
        return Response.model_validate({
            "id": f"resp_{len(captured)}", "object": "response", "created_at": STAMP,
            "model": "test-model", "status": "completed", "output": outputs.pop(0),
            "usage": None, "error": None, "incomplete_details": None,
            "instructions": None, "metadata": {}, "parallel_tool_calls": True,
            "temperature": None, "tool_choice": "auto", "tools": [], "top_p": None,
        })

    def make_agent(session_id):
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="codex_responses", max_iterations=4,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, session_db=db, session_id=session_id,
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent.valid_tool_names = {"read_file"}
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", respond)
        monkeypatch.setattr(agent, "_interruptible_api_call", respond)
        return agent

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "message_timestamps": {"enabled": True},
    })
    monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
    try:
        outputs.extend([
            [{
                "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file",
                "arguments": json.dumps({"path": str(tool_file)}),
            }],
            [{
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "phase": "final_answer", "content": [{"type": "output_text", "text": "first", "annotations": []}],
            }],
        ])
        first_agent = make_agent("timestamp-lifecycle")
        first = first_agent.run_conversation("first question", task_id="first", persist_user_timestamp=STAMP)

        assert first["completed"] is True
        assert len(captured) == 2
        assert captured[1]["input"][:len(captured[0]["input"])] == captured[0]["input"]
        assert first["messages"][0]["content"] == "first question"

        outputs.append([{
            "type": "message", "id": "msg_2", "role": "assistant", "status": "completed",
            "phase": "final_answer", "content": [{"type": "output_text", "text": "cached", "annotations": []}],
        }])
        cached = first_agent.run_conversation(
            "second question", conversation_history=first["messages"], task_id="cached",
            persist_user_timestamp=STAMP + 30,
        )
        assert cached["completed"] is True
        assert captured[-1]["input"][0]["content"] == "[Thu 2026-08-20 12:00:00 UTC] first question"
        assert cached["messages"][0]["content"] == "first question"

        reopened = SessionDB(db_path=db_path)
        try:
            durable_history = reopened.get_messages_as_conversation("timestamp-lifecycle")
        finally:
            reopened.close()
        outputs.append([{
            "type": "message", "id": "msg_3", "role": "assistant", "status": "completed",
            "phase": "final_answer", "content": [{"type": "output_text", "text": "reopened", "annotations": []}],
        }])
        reopened_agent = make_agent("timestamp-lifecycle")
        resumed = reopened_agent.run_conversation(
            "third question", conversation_history=durable_history, task_id="reopened",
            persist_user_timestamp=STAMP + 60,
        )
        assert resumed["completed"] is True
        assert captured[-1]["input"][0]["content"] == "[Thu 2026-08-20 12:00:00 UTC] first question"
        assert resumed["messages"][0]["content"] == "first question"
    finally:
        db.close()


def test_timestamped_history_survives_actual_tool_pruning_persistence_and_restart(tmp_path, monkeypatch):
    """A real prune commit must retain canonical timestamp rows and compatible sidecars."""
    from agent.context_compressor import _estimate_msg_budget_tokens
    from hermes_state import SessionDB
    from openai.types.responses import Response
    from run_agent import AIAgent

    db_path = tmp_path / "state.db"
    session_id = "timestamp-prune-restart"
    timestamped_sidecar = "[Thu 2026-08-20 12:00:30 UTC] retained question\n\nremember this context"
    db = SessionDB(db_path=db_path)
    captured = []

    history = [
        {"role": "user", "content": "first question", "timestamp": STAMP},
        {
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"/tmp/synthetic"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "x" * 24_000},
        {"role": "assistant", "content": "large result received"},
        {
            "role": "user", "content": "retained question", "timestamp": STAMP + 30,
            "api_content": timestamped_sidecar,
        },
        {"role": "assistant", "content": "retained answer"},
        {"role": "user", "content": "tail question", "timestamp": STAMP + 60},
        {"role": "assistant", "content": "tail answer"},
    ]

    def response(kwargs, **unused):
        captured.append(deepcopy(kwargs))
        return Response.model_validate({
            "id": "resp_prune", "object": "response", "created_at": STAMP,
            "model": "test-model", "status": "completed",
            "output": [{
                "type": "message", "id": "msg_prune", "role": "assistant",
                "status": "completed", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "resumed", "annotations": []}],
            }],
            "usage": None, "error": None, "incomplete_details": None,
            "instructions": None, "metadata": {}, "parallel_tool_calls": True,
            "temperature": None, "tool_choice": "auto", "tools": [], "top_p": None,
        })

    def agent_for(database):
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="codex_responses", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, session_db=database, session_id=session_id,
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent.context_compressor.proactive_prune_tokens = 48_000
        agent.context_compressor.proactive_prune_min_result_chars = 8_000
        agent.context_compressor.proactive_prune_min_reclaim_tokens = 4_096
        agent.context_compressor.protect_first_n = 0
        agent.context_compressor.protect_last_n = 2
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", response)
        monkeypatch.setattr(agent, "_interruptible_api_call", response)
        return agent

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "message_timestamps": {"enabled": True},
    })
    monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
    try:
        db.create_session(session_id, source="cli")
        db.append_messages_batch(session_id, history)
        first_agent = agent_for(db)
        before = db.get_messages_as_conversation(session_id)
        pruned, prune_count = first_agent.context_compressor.prune_tool_results_only(
            before, current_tokens=120_000,
        )

        assert prune_count >= 1
        assert len(pruned[2]["content"]) < len(history[2]["content"])
        assert sum(_estimate_msg_budget_tokens(message) for message in pruned) < sum(
            _estimate_msg_budget_tokens(message) for message in before
        )
    finally:
        db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        durable = reopened.get_messages_as_conversation(session_id)
        retained = next(message for message in durable if message.get("content") == "retained question")
        assert retained["timestamp"] == STAMP + 30
        assert retained["api_content"] == timestamped_sidecar
        resumed_agent = agent_for(reopened)
        resumed = resumed_agent.run_conversation(
            "resume question", conversation_history=durable, task_id="prune-restart",
            persist_user_timestamp=STAMP + 90,
        )
        assert resumed["completed"] is True
        assert any(message.get("content") == timestamped_sidecar for message in captured[0]["input"])
        retained_returned = next(message for message in resumed["messages"] if message.get("content") == "retained question")
        assert retained_returned["timestamp"] == STAMP + 30
        assert retained_returned["api_content"] == timestamped_sidecar
    finally:
        reopened.close()


def test_timestamped_retained_tail_survives_full_compaction_persistence_and_restart(tmp_path, monkeypatch):
    """Full production compaction may summarize history but not timestamped retained rows."""
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX
    from agent.conversation_compression import compress_context
    from hermes_state import SessionDB
    from openai.types.responses import Response
    from run_agent import AIAgent

    db_path = tmp_path / "state.db"
    session_id = "timestamp-compaction-restart"
    retained_sidecar = "[Thu 2026-08-20 12:00:30 UTC] retained tail question\n\nretained context"
    db = SessionDB(db_path=db_path)
    captured = []
    history = []
    for index in range(4):
        history.extend([
            {
                "role": "user", "content": f"old question {index} " + "x" * 4_000,
                "timestamp": STAMP + index,
            },
            {"role": "assistant", "content": f"old answer {index} " + "y" * 4_000},
        ])
    history.extend([
        {
            "role": "user", "content": "retained tail question", "timestamp": STAMP + 30,
            "api_content": retained_sidecar,
        },
        {"role": "assistant", "content": "retained tail answer"},
    ])

    def response(kwargs, **unused):
        captured.append(deepcopy(kwargs))
        return Response.model_validate({
            "id": "resp_compaction", "object": "response", "created_at": STAMP,
            "model": "test-model", "status": "completed",
            "output": [{
                "type": "message", "id": "msg_compaction", "role": "assistant",
                "status": "completed", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "continued", "annotations": []}],
            }],
            "usage": None, "error": None, "incomplete_details": None,
            "instructions": None, "metadata": {}, "parallel_tool_calls": True,
            "temperature": None, "tool_choice": "auto", "tools": [], "top_p": None,
        })

    def agent_for(database):
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="codex_responses", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, session_db=database, session_id=session_id,
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent._compression_feasibility_checked = True
        agent.compression_in_place = True
        agent.context_compressor.protect_first_n = 0
        agent.context_compressor.protect_last_n = 2
        agent.context_compressor.tail_token_budget = 80
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", response)
        monkeypatch.setattr(agent, "_interruptible_api_call", response)
        return agent

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "message_timestamps": {"enabled": True},
    })
    monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
    try:
        db.create_session(session_id, source="cli")
        db.append_messages_batch(session_id, history)
        first_agent = agent_for(db)
        with pytest.MonkeyPatch.context() as summary_patch:
            summary_patch.setattr(
                first_agent.context_compressor,
                "_generate_summary",
                lambda *args, **kwargs: f"{SUMMARY_PREFIX}\nSynthetic compacted history",
            )
            compacted, _ = compress_context(
                first_agent, db.get_messages_as_conversation(session_id),
                "Stable synthetic system prompt.", approx_tokens=100_000, force=True,
            )

        assert len(compacted) < len(history)
        assert any(message.get(COMPRESSED_SUMMARY_METADATA_KEY) for message in compacted)
        retained_compacted = next(
            message for message in compacted if message.get("content") == "retained tail question"
        )
        assert retained_compacted["timestamp"] == STAMP + 30
        assert retained_compacted["api_content"] == retained_sidecar
    finally:
        db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        durable = reopened.get_messages_as_conversation(session_id)
        retained = next(message for message in durable if message.get("content") == "retained tail question")
        assert retained["timestamp"] == STAMP + 30
        assert retained["api_content"] == retained_sidecar
        resumed_agent = agent_for(reopened)
        resumed = resumed_agent.run_conversation(
            "resume after compaction", conversation_history=durable, task_id="compaction-restart",
            persist_user_timestamp=STAMP + 90,
        )
        assert resumed["completed"] is True
        assert any(message.get("content") == retained_sidecar for message in captured[0]["input"])
        retained_returned = next(
            message for message in resumed["messages"] if message.get("content") == "retained tail question"
        )
        assert retained_returned["timestamp"] == STAMP + 30
        assert retained_returned["api_content"] == retained_sidecar
    finally:
        reopened.close()


def test_timestamped_turn_recovers_clean_canonical_history_after_provider_failure(tmp_path, monkeypatch):
    """A failed direct turn persists clean data that a new agent can timestamp on resume."""
    from hermes_state import SessionDB
    from openai.types.responses import Response
    from run_agent import AIAgent

    db_path = tmp_path / "state.db"
    session_id = "timestamp-failure-restart"
    db = SessionDB(db_path=db_path)
    captured = []

    def agent_for(database):
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="codex_responses", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, session_db=database, session_id=session_id,
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        return agent

    def recovered_response(kwargs, **unused):
        captured.append(deepcopy(kwargs))
        return Response.model_validate({
            "id": "resp_recovered", "object": "response", "created_at": STAMP,
            "model": "test-model", "status": "completed",
            "output": [{
                "type": "message", "id": "msg_recovered", "role": "assistant",
                "status": "completed", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "recovered", "annotations": []}],
            }],
            "usage": None, "error": None, "incomplete_details": None,
            "instructions": None, "metadata": {}, "parallel_tool_calls": True,
            "temperature": None, "tool_choice": "auto", "tools": [], "top_p": None,
        })

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "message_timestamps": {"enabled": True},
    })
    monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
    try:
        failing_agent = agent_for(db)
        monkeypatch.setattr(
            failing_agent, "_interruptible_streaming_api_call",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic provider failure")),
        )
        monkeypatch.setattr(
            failing_agent, "_interruptible_api_call",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic provider failure")),
        )
        failed = failing_agent.run_conversation(
            "failed question", task_id="failure", persist_user_timestamp=STAMP,
        )
        assert failed["completed"] is False
    finally:
        db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        durable = reopened.get_messages_as_conversation(session_id)
        failed_user = next(message for message in durable if message.get("role") == "user")
        assert failed_user["content"] == "failed question"
        assert failed_user["timestamp"] == STAMP
        recovered_agent = agent_for(reopened)
        monkeypatch.setattr(recovered_agent, "_interruptible_streaming_api_call", recovered_response)
        monkeypatch.setattr(recovered_agent, "_interruptible_api_call", recovered_response)
        recovered = recovered_agent.run_conversation(
            "resume question", conversation_history=durable, task_id="recovery",
            persist_user_timestamp=STAMP + 30,
        )
        assert recovered["completed"] is True
        # Failure leaves no assistant row, so the normal alternation repair merges
        # the resumed input into the retained user row. The original logical input
        # must stay once with its original timestamp rather than be restamped.
        resumed_wire = captured[0]["input"][0]["content"]
        assert resumed_wire.startswith("[Thu 2026-08-20 12:00:00 UTC] failed question")
        assert resumed_wire.count("failed question") == 1
        assert resumed_wire.count("resume question") == 1
        # The ordinary alternation repair is retained history policy, not the
        # timestamp projection; it remains clean and contains each logical ask once.
        assert recovered["messages"][0]["content"] == "failed question\n\nresume question"
    finally:
        reopened.close()
