"""Real plugin dispatch, turn loop, and temporary SQLite; provider I/O is simulated."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import hermes_yaml as yaml

from hermes_cli.plugins import PluginManager
from hermes_state import SessionDB
from run_agent import AIAgent

QUESTION = "¿Cuál es el principal riesgo posoperatorio temprano tras ritidectomía?"
ANSWER = "El hematoma es una complicación temprana tras ritidectomía."
BLOCKED = "UNCERTIFIED_SENTINEL: administra cefazolina 9 mg cada 3 h."


def completion(text=None, tool=False):
    calls = [SimpleNamespace(id="evidence-1", type="function", function=SimpleNamespace(
        name="web_search", arguments=json.dumps({"query": "synthetic fixture"}),
    ))] if tool else None
    message = SimpleNamespace(content=text, tool_calls=calls, reasoning=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool else "stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15), model="fake/model",
    )


@pytest.fixture
def delivery_agent(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["clinical-evidence-guard"]},
        "display": {"file_mutation_verifier": False, "turn_completion_explainer": False},
    }))
    manager = PluginManager()
    manager.discover_and_load()
    assert manager.has_hook("pre_delivery")
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    db = SessionDB(db_path=home / "state.db")
    schema = {"type": "function", "function": {"name": "web_search", "description": "Fixture evidence",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}
    with (patch("model_tools.get_tool_definitions", return_value=[schema]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(
            api_key="fake-test-key", base_url="https://example.invalid/v1", model="fake/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli",
            session_id="delivery-integration", session_db=db, max_iterations=12,
        )
    agent.client = MagicMock()
    agent._file_mutation_verifier_enabled = lambda: False
    agent._turn_completion_explainer_enabled = lambda: False
    delivered = []
    agent.stream_delta_callback = delivered.append
    agent.reasoning_callback = lambda text: delivered.append("reasoning:" + text)
    agent.tool_progress_callback = lambda *args, **kwargs: delivered.append("tool:" + str(args))
    agent._save_trajectory = MagicMock()
    monkeypatch.setattr("agent.turn_context.start_deferred_title_upgrade", lambda *args: None)
    monkeypatch.setattr("agent.title_generator.generate_title", lambda *args, **kwargs: "Synthetic test")
    yield agent, db, delivered, manager
    manager.unload()
    db.close()


def test_blocked_candidate_repaired_using_tool_evidence_before_release_and_persistence(delivery_agent, monkeypatch):
    agent, db, delivered, _manager = delivery_agent
    responses = iter([completion(BLOCKED), completion(tool=True), completion(ANSWER)])
    requests = []

    def create(**kwargs):
        requests.append([dict(row) for row in kwargs["messages"]])
        agent._fire_stream_delta(BLOCKED)
        agent._fire_reasoning_delta(BLOCKED)
        return next(responses)

    agent.client.chat.completions.create = create
    monkeypatch.setattr("model_tools.handle_function_call", lambda *args, **kwargs: json.dumps({
        "results": [{"content": ANSWER, "url": "https://example.invalid/synthetic-source"}],
    }))
    result = agent.run_conversation(QUESTION)

    assert result["final_response"] == ANSWER
    assert result["completed"] is True
    assert result["api_calls"] == 3
    assert result["clinical_delivery_trace"]["repair_count"] == 1
    assert delivered == [ANSWER]
    assert result["last_reasoning"] is None
    assert result["pre_transform_response"] is None
    assert "[PRE_DELIVERY_REPAIR]" in json.dumps(requests[1], ensure_ascii=False)
    stored = db.get_messages("delivery-integration")
    assert [row["content"] for row in stored if row["role"] == "assistant" and row["content"]] == [ANSWER]
    for rows in (stored, result["messages"], agent._save_trajectory.call_args.args[0]):
        rendered = json.dumps(rows, ensure_ascii=False, default=str)
        assert "UNCERTIFIED_SENTINEL" not in rendered
        assert "[PRE_DELIVERY_REPAIR]" not in rendered
    # Every request retains the exact original system bytes.
    system_rows = [[row for row in rows if row["role"] == "system"] for rows in requests]
    assert all(rows == system_rows[0] for rows in system_rows)


def test_repair_exhaustion_has_finite_calls_and_no_blocked_history(delivery_agent):
    from agent.turn_stop_gates import _MAX_PRE_DELIVERY_REPAIR_ATTEMPTS, _PRE_DELIVERY_TERMINAL_RESPONSE
    agent, db, delivered, _manager = delivery_agent
    agent.client.chat.completions.create.side_effect = [
        completion(BLOCKED) for _ in range(_MAX_PRE_DELIVERY_REPAIR_ATTEMPTS + 1)
    ]
    result = agent.run_conversation(QUESTION)
    assert result["api_calls"] == _MAX_PRE_DELIVERY_REPAIR_ATTEMPTS + 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["final_response"] == _PRE_DELIVERY_TERMINAL_RESPONSE
    assert delivered == [_PRE_DELIVERY_TERMINAL_RESPONSE]
    assert "UNCERTIFIED_SENTINEL" not in json.dumps(db.get_messages("delivery-integration"), default=str)
    assert "[PRE_DELIVERY_REPAIR]" not in json.dumps(result["messages"], default=str)


def _finalize_direct(agent, messages, candidate):
    from agent.turn_finalizer import finalize_turn
    agent._pre_delivery_gate_active = True
    agent._current_turn_user_idx = 0
    agent._current_turn_id = "direct-protected"
    return finalize_turn(
        agent, final_response=candidate, api_call_count=1, interrupted=False, failed=False,
        messages=messages, conversation_history=None, effective_task_id="task", turn_id="direct-protected",
        user_message=QUESTION, original_user_message=QUESTION, _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


def test_cleanup_exception_cannot_save_uncertified_candidate(delivery_agent):
    agent, _db, delivered, _manager = delivery_agent
    def broken_cleanup(_messages):
        raise RuntimeError("synthetic cleanup failure")
    agent._drop_trailing_empty_response_scaffolding = broken_cleanup
    messages = [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": BLOCKED}]
    result = _finalize_direct(agent, messages, BLOCKED)
    assert result["completed"] is False
    assert result["failed"] is True
    assert not result["final_response"]
    assert not delivered
    assert "UNCERTIFIED_SENTINEL" not in json.dumps(result["messages"], default=str)
    assert "UNCERTIFIED_SENTINEL" not in json.dumps(agent._save_trajectory.call_args.args[0], default=str)


def test_rejected_host_suffix_marks_failed_before_any_persistence(delivery_agent, monkeypatch):
    agent, db, delivered, manager = delivery_agent
    manager.invoke_hook("pre_llm_call", session_id=agent.session_id, turn_id="direct-protected", user_message=QUESTION)
    manager.invoke_hook("post_tool_call", session_id=agent.session_id, turn_id="direct-protected",
                        tool_name="web_search", result={"content": ANSWER})
    agent._pre_delivery_output_cache = {"response_text": ANSWER, "transformed": False, "pre_transform": None}
    monkeypatch.setattr("agent.turn_finalizer._append_file_mutation_footer",
                        lambda _agent, text, _logger: text + "\n" + BLOCKED)
    messages = [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": ANSWER}]
    result = _finalize_direct(agent, messages, ANSWER)
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "pre_delivery_blocked"
    assert not result["final_response"]
    assert not delivered
    assert "UNCERTIFIED_SENTINEL" not in json.dumps(db.get_messages(agent.session_id), default=str)
