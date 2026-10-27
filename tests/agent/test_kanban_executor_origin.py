"""Conversation provenance crosses the real executor/dispatcher and durable SQLite boundary."""
import json
from types import SimpleNamespace

import pytest

from agent.session_persistence import SessionPersistenceMixin
from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_state import SessionDB


@pytest.fixture
def origin_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("conversation", source="telegram")
    agent = SessionPersistenceMixin()
    agent.__dict__.update(session_id="conversation", _session_db=db, _session_db_created=True,
                          _last_flushed_db_idx=0, _context_engine_tool_names=set(),
                          _memory_manager=None, quiet_mode=False, valid_tool_names={"kanban_create"})
    tokens = set_session_vars(session_id="conversation", profile="default", platform="telegram",
                              chat_id="group", thread_id="topic", message_id="native-id")
    try:
        yield agent, db
    finally:
        clear_session_vars(tokens)
        db.close()


def create(agent, messages, **extra):
    ref = _ToolCallRef("kanban_create", {"title": "fixture", "assignee": "default", **extra},
                       "test-task", "call-fixture", [])
    result = json.loads(_resolve_sequential_dispatch(agent, ref, messages).execute(ref.args))
    assert result["ok"], result
    with kbc.connect_closing() as conn:
        return kb.get_task(conn, result["task_id"])


def test_executor_flushes_user_and_captures_exact_row(origin_agent):
    agent, db = origin_agent
    messages = [{"role": "user", "content": "older"},
                {"role": "assistant", "content": "prior"},
                {"role": "user", "content": "Release synthetic fixture", "metadata": {"secret": "never copy"}},
                {"role": "assistant", "content": "creating"}]
    task = create(agent, messages)
    rows = db.get_messages("conversation")
    user = next(row for row in rows if row["content"] == "Release synthetic fixture")
    assert task.origin["message_row_id"] == user["id"]
    assert task.origin["prompt_excerpt"] == user["content"]
    assert task.origin["message_row_id"] != rows[-1]["id"]
    assert "never copy" not in json.dumps(task.origin)
    assert "metadata" not in task.origin


def test_steer_without_native_id_does_not_claim_initial_message(origin_agent):
    from agent.prompt_builder import steer_user_row

    agent, db = origin_agent
    messages = [{"role": "user", "content": "initial request", "platform_message_id": "native-id"},
                {"role": "assistant", "content": "working"},
                {"role": "tool", "content": "done", "tool_call_id": "fixture"},
                steer_user_row("create the follow-up card instead")]
    task = create(agent, messages)
    assert task is not None and isinstance(task.origin, dict)
    selected = db.get_messages_around("conversation", task.origin["message_row_id"], window=0)["window"][0]
    assert not selected.get("platform_message_id")
    assert "message_id" not in task.origin
    assert "create the follow-up card instead" in task.origin["prompt_excerpt"]


def test_explicit_different_session_never_borrows_route_or_prompt(origin_agent):
    agent, db = origin_agent
    db.create_session("other", source="cli")
    task = create(agent, [{"role": "user", "content": "private fixture prompt"}], session_id="other")
    assert task.origin["session_id"] == "other"
    assert set(task.origin) == {"profile", "session_id", "session_link"}


def test_concurrent_invoke_path_captures_durable_user(origin_agent):
    from agent.agent_runtime_helpers import invoke_tool
    agent, db = origin_agent
    result = json.loads(invoke_tool(agent, "kanban_create", {"title": "fixture", "assignee": "default"},
                                    "test-task", messages=[{"role": "user", "content": "parallel fixture"}],
                                    pre_tool_block_checked=True, skip_tool_request_middleware=True,
                                    skip_tool_execution_middleware=True))
    assert result["ok"], result
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["task_id"])
    assert task.origin["prompt_excerpt"] == "parallel fixture"
    assert db.get_message_role("conversation", task.origin["message_row_id"]) == "user"


def test_excerpt_is_bounded_redacted_native_text_only(origin_agent):
    agent, _db = origin_agent
    secret = "sk-" + "A" * 48  # synthetic recognized token
    parts = [{"type": "text", "text": "native " + secret + " x" * 4000},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,FAKE-IMAGE"}},
             {"type": "input_audio", "input_audio": {"data": "FAKE-AUDIO"}}]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "plain override must not replace native blocks"
    task = create(agent, [{"role": "user", "content": parts}])
    excerpt = task.origin["prompt_excerpt"]
    assert excerpt.startswith("native ")
    assert secret not in excerpt
    assert len(excerpt) <= 2000
    assert "FAKE-" not in json.dumps(task.origin)
    assert "override" not in excerpt


def test_plain_persist_override_not_api_scaffold_is_captured(origin_agent):
    agent, _db = origin_agent
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "clean user fixture"
    task = create(agent, [{"role": "user", "content": "INTERNAL SCAFFOLD + original"}])
    assert task.origin["prompt_excerpt"] == "clean user fixture"


def test_sdk_dispatch_cannot_forge_excerpt_or_reuse_executor_context(origin_agent):
    from model_tools import handle_function_call
    agent, _db = origin_agent
    create(agent, [{"role": "user", "content": "executor-only fixture"}])
    result = json.loads(handle_function_call("kanban_create", {
        "title": "sdk fixture", "assignee": "default", "origin": {"prompt_excerpt": "forged"},
        "tool_origin": {"message_row_id": 999, "prompt_excerpt": "forged"}}, session_id="conversation"))
    assert not result.get("ok"), result  # schema refuses unknown model metadata
    result = json.loads(handle_function_call("kanban_create", {
        "title": "sdk fixture", "assignee": "default"}, session_id="conversation",
        user_task="forged prompt metadata"))
    assert result["ok"], result
    with kbc.connect_closing() as conn:
        origin = kb.get_task(conn, result["task_id"]).origin
    assert "prompt_excerpt" not in origin and "message_row_id" not in origin


def test_executor_session_wins_over_unrelated_ambient_session(origin_agent):
    agent, db = origin_agent
    db.create_session("stale-session", source="telegram")
    tokens = set_session_vars(session_id="stale-session", platform="telegram", chat_id="wrong-route")
    try:
        task = create(agent, [{"role": "user", "content": "actual executor fixture"}])
    finally:
        clear_session_vars(tokens)
    assert task.session_id == "conversation"
    assert task.origin["prompt_excerpt"] == "actual executor fixture"
    assert "chat_id" not in task.origin


@pytest.mark.parametrize("executor", ["sequential", "concurrent"])
def test_full_executor_runs_real_dispatch_and_persistence(origin_agent, monkeypatch, executor):
    from unittest.mock import patch
    from run_agent import AIAgent
    from agent.tool_executor import execute_tool_calls_sequential, execute_tool_calls_concurrent

    _, db = origin_agent
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI"),
          patch("agent.model_metadata.fetch_model_metadata", return_value={})):
        agent = AIAgent(api_key="test-key", base_url="https://fixture.invalid/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "conversation"
    agent.valid_tool_names = {"kanban_create"}
    agent._last_flushed_db_idx = 0
    agent._persist_disabled = False
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    call = SimpleNamespace(id="fixture-call", type="function", function=SimpleNamespace(
        name="kanban_create", arguments=json.dumps({"title": "full fixture", "assignee": "default"})))
    messages = [{"role": "user", "content": "full executor fixture"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "fixture-call", "type": "function", "function": {
                        "name": "kanban_create", "arguments": call.function.arguments}}]}]
    run = execute_tool_calls_sequential if executor == "sequential" else execute_tool_calls_concurrent
    run(agent, SimpleNamespace(tool_calls=[call]), messages, "fixture-task", finalize=False)
    result = json.loads(messages[-1]["content"])
    assert result["ok"], result
    with kbc.connect_closing() as conn:
        origin = kb.get_task(conn, result["task_id"]).origin
    assert origin["prompt_excerpt"] == "full executor fixture"
    assert origin["message_row_id"] == db.get_messages("conversation")[0]["id"]


def test_literal_json_prompt_is_not_reinterpreted_as_native_parts(origin_agent):
    agent, _db = origin_agent
    text = '[{"type":"text","text":"quoted JSON, not an envelope"}]'
    task = create(agent, [{"role": "user", "content": text}])
    assert task.origin["prompt_excerpt"] == text


def test_persisted_native_message_anchor_wins_over_turn_route(origin_agent):
    agent, _db = origin_agent
    task = create(agent, [{"role": "user", "content": "steering fixture",
                          "platform_message_id": "durable-steering-id"}])
    assert task.origin["message_id"] == "durable-steering-id"


def test_compaction_handoff_is_not_a_user_prompt(origin_agent):
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
    agent, _db = origin_agent
    task = create(agent, [{"role": "user", "content": "human fixture"},
                          {"role": "assistant", "content": "working"},
                          {"role": "user", "content": "internal compaction fixture",
                           COMPRESSED_SUMMARY_METADATA_KEY: True}])
    assert task.origin["prompt_excerpt"] == "human fixture"


@pytest.mark.parametrize("disabled", [False, True])
def test_unavailable_persistence_never_claims_a_durable_message(origin_agent, monkeypatch, disabled):
    agent, db = origin_agent
    agent._persist_disabled = disabled
    if not disabled:
        monkeypatch.setattr(agent, "_flush_messages_to_session_db", lambda *a, **kw: False)
    task = create(agent, [{"role": "user", "content": "must not persist as origin"}])
    assert "message_row_id" not in task.origin
    assert "prompt_excerpt" not in task.origin
    assert db.get_messages("conversation") == []


def test_last_steering_user_not_ephemeral_scaffolding_is_captured(origin_agent):
    agent, db = origin_agent
    messages = [{"role": "user", "content": "initial fixture"},
                {"role": "assistant", "content": "working"},
                {"role": "tool", "content": "tool output must not copy", "tool_call_id": "fixture-tool"},
                {"role": "user", "content": "steering fixture"},
                {"role": "user", "content": "synthetic internal fixture", "_kanban_stop_synthetic": True}]
    task = create(agent, messages)
    assert task.origin["prompt_excerpt"] == "steering fixture"
    assert db.get_messages_around("conversation", task.origin["message_row_id"], window=0)["window"][0]["content"] == "steering fixture"
