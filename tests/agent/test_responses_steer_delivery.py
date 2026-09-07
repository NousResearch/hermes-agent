"""Normalized pairing repair preserves results and guidance at the wire boundary."""

import copy
import json
import threading
from types import SimpleNamespace as NS

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results, repair_message_sequence
from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.chat_completion_helpers import _assistant_tool_call_dict
from agent.codex_responses_adapter import (
    _derive_responses_function_call_id,
    _split_responses_tool_id,
)
from agent.interrupt_control import InterruptControlMixin
from agent.message_sanitization import uniquify_tool_call_ids
from agent.transports.codex import ResponsesApiTransport
from agent.transports.types import ToolCall
from agent.turn_iteration_prep import _inject_steer_into_newest_tool_result

SENTINEL = "Please include the violet sentinel in the answer."


class _ProjectionAgent(InterruptControlMixin):
    _split_responses_tool_id = staticmethod(_split_responses_tool_id)
    _derive_responses_function_call_id = staticmethod(_derive_responses_function_call_id)

    def __init__(self):
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()


def _response(*, tools=True, duplicate=True):
    output = [
        NS(type="function_call", id=f"fc_item{i}",
           call_id="call_shared" if duplicate else f"call_{i}",
           name="web_search", arguments=json.dumps({"query": f"probe{i}"}))
        for i in range(2)
    ] if tools else [NS(type="message", role="assistant", status="completed",
                       content=[NS(type="output_text", text="Finished.")])]
    return NS(output=output, status="completed", model="test/model",
              usage=NS(input_tokens=12, output_tokens=4, total_tokens=16))


@pytest.mark.parametrize("wire", ["responses", "messages"])
@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("boundary", ["post_tool", "pre_request"])
@pytest.mark.parametrize("multimodal", [False, True])
def test_repaired_results_retain_steering_through_projection(wire, duplicate, boundary, multimodal):
    agent = _ProjectionAgent()
    transport = ResponsesApiTransport()
    calls = transport.normalize_response(_response(duplicate=duplicate)).tool_calls
    if wire == "messages":
        calls = [ToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in calls]
    uniquify_tool_call_ids(calls)
    serialized = [_assistant_tool_call_dict(agent, call, i) for i, call in enumerate(calls)]
    messages = [
        {"role": "user", "content": "Run both probes."},
        {"role": "assistant", "content": "", "tool_calls": serialized},
    ]
    prefix = copy.deepcopy(messages)
    for i, call in enumerate(serialized):
        content = [
            {"type": "text", "text": f"result{i}"},
            {"type": "image_url", "image_url": {"url": f"https://example.invalid/image{i}.png"}},
        ] if multimodal else f"result{i}"
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    assert agent.steer(SENTINEL)
    if boundary == "post_tool":
        apply_pending_steer_to_tool_results(agent, messages, 2)
    else:
        _inject_steer_into_newest_tool_result(agent, messages, agent._drain_pending_steer())
    repair_message_sequence(agent, messages)
    payload = (transport.convert_messages(messages, is_github_responses=True)
               if wire == "responses" else convert_messages_to_anthropic(messages))
    encoded = json.dumps(payload)
    assert encoded.count(SENTINEL) == 1
    assert all(f"result{i}" in encoded for i in range(2))
    if multimodal:
        assert all(f"https://example.invalid/image{i}.png" in encoded for i in range(2))
    assert len({call["id"] for call in serialized}) == 2
    assert messages[:2] == prefix
    assert agent._pending_steer is None
    snapshot = copy.deepcopy(messages)
    apply_pending_steer_to_tool_results(agent, messages, 2)
    repair_message_sequence(agent, messages)
    assert messages == snapshot


class _Stream:
    def __init__(self, response):
        self.response = response

    def __iter__(self):
        for item in self.response.output:
            yield NS(type="response.output_item.done", item=item)
        yield NS(type="response.completed", response=self.response)

    def close(self):
        pass


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("tool_name", ["terminal", "web_search"])
@pytest.mark.parametrize("arrival", ["during_tool", "between_requests"])
def test_loop_delivers_repaired_batch_and_resumes_pairing(
    monkeypatch, tmp_path, tool_name, arrival, retry, outcome="normal",
):
    from run_agent import AIAgent
    from hermes_state import SessionDB

    definitions = [{"type": "function", "function": {
        "name": tool_name, "description": "Inert fixture", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}},
        },
    }}]
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: definitions)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda *a, **k: {})
    monkeypatch.setattr("agent.title_generator._auto_title_enabled", lambda: False)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0)
    # No external socket may escape the recording SDK boundary.
    def no_network(*args, **kwargs):
        raise AssertionError("unexpected external connection in offline fixture")
    monkeypatch.setattr("socket.socket.connect", no_network)
    agent = AIAgent(model="gpt-5.4", provider="custom", api_mode="codex_responses",
                    api_key="test-key", base_url="https://example.invalid/v1",
                    quiet_mode=True, max_iterations=1 if outcome == "limit" else 4,
                    skip_context_files=True, skip_memory=True)
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.skip_background_review = True
    agent._cached_system_prompt = "Run the requested probes."
    agent._cleanup_task_resources = lambda *a: None
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    session_id = "pairing-fixture"
    db.create_session(session_id=session_id, source="cli", model="test/model")
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._persist_disabled = False
    captured, executed, accepted = [], [], []
    execution_ids = {}
    entered, release = threading.Event(), threading.Event()

    def dispatch(name, arguments, *args, **kwargs):
        executed.append(arguments["query"])
        execution_ids[arguments["query"]] = kwargs["tool_call_id"]
        entered.set()
        if not release.wait(10):
            raise RuntimeError("fixture controller did not release the tool")
        if arguments["query"] == "probe1":
            if outcome == "tool_error":
                raise RuntimeError("probe1 controlled tool failure")
            if outcome == "stop":
                agent.hard_interrupt()
            if outcome == "redirect":
                assert agent.redirect("Redirect fixture")
        return json.dumps({"result": arguments["query"]})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)

    def create(**kwargs):
        body = {**kwargs, **kwargs.get("extra_body", {})}
        captured.append(copy.deepcopy(body["input"]))
        if retry and len(captured) == 2:
            import httpx
            raise httpx.ReadTimeout("controlled provider retry")
        response = _response(tools=len(captured) == 1)
        if len(captured) == 1:
            for item in response.output:
                item.name = tool_name
        return _Stream(response) if kwargs.get("stream") else response

    agent.client = NS(responses=NS(create=create), close=lambda: None)
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: agent.client)
    if arrival == "between_requests":
        release.set()
        def on_step(count, previous):
            if previous:
                accepted.append(agent.steer(SENTINEL))
        agent.step_callback = on_step
    else:
        def steer_while_held():
            if entered.wait(10):
                accepted.append(agent.steer(SENTINEL))
            release.set()
        controller = threading.Thread(target=steer_while_held)
        controller.start()
    try:
        result = agent.run_conversation("Run both probes.")
    finally:
        release.set()
        if arrival == "during_tool":
            controller.join(10)
        db.close()
    assert accepted == [True]
    assert sorted(executed) == ["probe0", "probe1"]
    if outcome == "stop":
        assert result["interrupted"] is True
        assert len(captured) == 1
        eligible = ResponsesApiTransport().convert_messages(result["messages"], is_github_responses=True)
    else:
        assert result["final_response"] == "Finished."
        assert len(captured) == (3 if retry else 2)
        if outcome == "limit":
            assert result["completed"] is False
        if retry:
            assert captured[2] == captured[1]
            assert json.dumps(captured[2]).count(SENTINEL) == 1
        eligible = captured[1]
    outputs = [item for item in eligible if item.get("type") == "function_call_output"]
    calls = [item for item in eligible if item.get("type") == "function_call"]
    assert len(outputs) == len(calls) == 2, "wire dropped a legitimate tool result"
    assert {item["call_id"] for item in calls} == {item["call_id"] for item in outputs}
    assert len({item["call_id"] for item in outputs}) == 2
    for call in calls:
        query = json.loads(call["arguments"])["query"]
        output = next(item for item in outputs if item["call_id"] == call["call_id"])
        assert execution_ids[query] == call["call_id"]
        assert query in json.dumps(output["output"])
    # Upstream replays pairing IDs, not provider response-item IDs.
    assert all("id" not in call for call in calls)
    if outcome != "stop":
        assert json.dumps(captured[1]).count(SENTINEL) == 1
        assert not result.get("interrupted")
        assert not result.get("pending_steer")
    from agent.conversation_compression import _ensure_compressed_has_user_turn
    from agent.context_compressor import SUMMARY_PREFIX
    live = result["messages"]
    # Retained repaired tool tail needs no extra anchor; a dropped tail does.
    tool_block = next(i for i, msg in enumerate(live) if msg.get("tool_calls"))
    retained = [{"role": "user", "content": SUMMARY_PREFIX + " Earlier work."},
                *copy.deepcopy(live[tool_block:tool_block + 3])]
    assert _ensure_compressed_has_user_turn(live, retained) == "already_present"
    assert json.dumps(retained).count(SENTINEL) == 1
    dropped = [{"role": "user", "content": SUMMARY_PREFIX + " Earlier work."}]
    assert _ensure_compressed_has_user_turn(live, dropped) == "inserted"
    assert json.dumps(dropped).count(SENTINEL) == 1
    # A separate agent owns its own queue, regardless of repaired history.
    first_owner, second_owner = _ProjectionAgent(), _ProjectionAgent()
    assert first_owner.steer("Only the first agent owns this guidance.")
    assert second_owner._drain_pending_steer() is None
    assert first_owner._drain_pending_steer() == "Only the first agent owns this guidance."
    restarted = SessionDB(db_path=db_path)
    try:
        durable = restarted.get_messages_as_conversation(session_id)
    finally:
        restarted.close()
    results = [msg for msg in durable if msg["role"] == "tool"]
    durable_calls = [call for msg in durable for call in msg.get("tool_calls", [])]
    live_calls = [call for msg in live for call in msg.get("tool_calls", [])]
    assert len(durable_calls) == len(live_calls) == len(results) == 2
    assert {call["id"] for call in durable_calls} == {msg["tool_call_id"] for msg in results}
    assert durable_calls == live_calls
    assert {call["response_item_id"] for call in durable_calls} == {"fc_item0", "fc_item1"}
    assert {msg["tool_call_id"] for msg in results} == {item["call_id"] for item in outputs}
    assert all(any(query in json.dumps(msg["content"]) for msg in results) for query in executed)
    # Existing upstream limitation: post-flush content mutation is not a DB update.
    assert SENTINEL not in json.dumps(durable)
    resumed = ResponsesApiTransport().convert_messages(durable, is_github_responses=True)
    assert len([item for item in resumed if item.get("type") == "function_call_output"]) == 2


@pytest.mark.parametrize("outcome", ["tool_error", "limit", "stop", "redirect"])
def test_repaired_pairing_through_lifecycle_exits(monkeypatch, tmp_path, outcome):
    test_loop_delivers_repaired_batch_and_resumes_pairing(
        monkeypatch, tmp_path, "terminal", "during_tool", False, outcome,
    )


@pytest.mark.parametrize("persistence_error", [False, True])
def test_partial_exit_keeps_pairing_and_existing_pending_contract(monkeypatch, persistence_error):
    from agent.turn_tool_validation import _partial_exit

    agent = _ProjectionAgent()
    calls = ResponsesApiTransport().normalize_response(_response()).tool_calls
    uniquify_tool_call_ids(calls)
    serialized = [_assistant_tool_call_dict(agent, call, i) for i, call in enumerate(calls)]
    messages = [{"role": "user", "content": "Run probes."},
                {"role": "assistant", "content": "", "tool_calls": serialized}]
    messages.extend({"role": "tool", "tool_call_id": call["id"], "content": f"result{i}"}
                    for i, call in enumerate(serialized))
    assert agent.steer(SENTINEL)
    saved = []
    def persist(messages, history):
        if persistence_error:
            raise RuntimeError("controlled persistence error")
        saved.append(copy.deepcopy(messages))
    agent._persist_session = persist
    if persistence_error:
        with pytest.raises(RuntimeError, match="controlled persistence error"):
            _partial_exit(agent, messages, None, 2, "Partial fixture")
        assert not saved
    else:
        result = _partial_exit(agent, messages, None, 2, "Partial fixture")
        assert result["partial"] is True
        assert result["completed"] is False
        assert "pending_steer" not in result  # Existing upstream omission, not a new handoff.
        assert saved == [messages]
    assert agent._pending_steer == SENTINEL
    repair_message_sequence(agent, messages)
    outputs = [m for m in messages if m["role"] == "tool"]
    assert len(outputs) == 2
    assert {m["tool_call_id"] for m in outputs} == {c["id"] for c in serialized}
    assert [m["content"] for m in outputs] == ["result0", "result1"]


def test_recording_loop_detects_getter_only_mutation(monkeypatch, tmp_path):
    """Restoring the original property must fail at the recording wire boundary."""
    monkeypatch.setattr(ToolCall, "call_id", property(ToolCall.call_id.fget))
    with pytest.raises(AssertionError, match="wire dropped a legitimate tool result"):
        test_loop_delivers_repaired_batch_and_resumes_pairing(
            monkeypatch, tmp_path, "terminal", "during_tool", False,
        )
