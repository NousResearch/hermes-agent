# ABOUTME: Exercises completion policies through real plugin discovery and HTTP model transport.
# ABOUTME: Checks hidden candidates, durable history, bounded continuation, and resumed turns.
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


_PLUGIN = '''
import json
import time
from pathlib import Path

def register(ctx):
    root = Path(__file__).parent
    def review(**payload):
        mode = json.loads((root / "mode.json").read_text())
        with (root / "seen.jsonl").open("a") as output:
            output.write(json.dumps(payload) + "\\n")
        if mode == "none":
            return None
        if mode == "error":
            raise RuntimeError("PRIVATE_EXCEPTION")
        if mode == "timeout":
            time.sleep(3)
            return {"action": "allow"}
        if mode == "interrupt":
            time.sleep(1)
            return {"action": "allow"}
        if mode == "invalid":
            return {"action": "continue", "message": 42}
        if mode == "retrieve" and not any(row.get("role") == "tool" for row in payload["messages"]):
            return {"action": "continue", "message": "INTERNAL_REPAIR: read the evidence file."}
        if mode in {"fail", "reasoning", "truncated"}:
            return {"action": "fail", "message": "Review rejected this answer.", "code": "evidence_missing"}
        if mode in {"repeat", "budget", "veto"} or (mode == "continue" and payload.get("attempt", 0) == 0):
            return {"action": "continue", "message": "INTERNAL_REPAIR: verify the requested evidence."}
        if mode == "mutate":
            payload["messages"][0]["content"] = "CORRUPTED_REQUEST"
        return {"action": "allow"}
    ctx.register_hook("before_turn_end", review)
    def veto(**payload):
        if json.loads((root / "mode.json").read_text()) == "veto":
            return {"action": "fail", "message": "A second policy rejected this answer."}
    ctx.register_hook("before_turn_end", veto)
    def transform(response_text, **kwargs):
        if json.loads((root / "mode.json").read_text()) == "transform":
            return "Public transformed answer."
    ctx.register_hook("transform_llm_output", transform)
    def observe(**payload):
        with (root / "stream-observer.jsonl").open("a") as output:
            output.write(json.dumps(payload) + "\\n")
    ctx.register_hook("on_stream_delta", observe)
    ctx.register_hook("on_stream_end", observe)
'''


@pytest.fixture
def gate_runtime(tmp_path, monkeypatch):
    from hermes_cli import plugins
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)
    plugin = tmp_path / "plugins" / "completion-review"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: completion-review\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text(_PLUGIN)
    requests = []
    answer = {"text": "PRIVATE_DRAFT"}
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("The verified count is 42.\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith(("/chat/completions", "/messages", "/responses")):
                self.send_error(404)
                return
            requests.append(body)
            response = {
                "id": "completion-test", "object": "chat.completion", "created": 1,
                "model": "review-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer["text"]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            if json.loads((plugin / "mode.json").read_text()) == "truncated":
                response["choices"][0]["finish_reason"] = "length"
            if json.loads((plugin / "mode.json").read_text()) == "reasoning":
                response["choices"][0]["message"]["content"] = ""
                response["choices"][0]["message"]["reasoning_content"] = answer["text"]
            if json.loads((plugin / "mode.json").read_text()) == "retrieve" and len(requests) == 2:
                response["choices"][0]["message"] = {"role": "assistant", "content": None, "tool_calls": [
                    {"index": 0, "id": "read-evidence", "type": "function", "function": {
                        "name": "read_file", "arguments": json.dumps({"path": str(evidence)})}},
                ]}
                response["choices"][0]["finish_reason"] = "tool_calls"
            if self.path.endswith("/messages"):
                response = {"id": "msg-completion-test", "type": "message", "role": "assistant", "model": "review-model",
                            "content": [{"type": "text", "text": answer["text"]}], "stop_reason": "end_turn",
                            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
                if body.get("stream"):
                    events = [
                        {"type": "message_start", "message": {**response, "content": [], "stop_reason": None}},
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": answer["text"]}},
                        {"type": "content_block_stop", "index": 0},
                        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
                        {"type": "message_stop"},
                    ]
                    data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
                else:
                    data = json.dumps(response).encode()
                content_type = "text/event-stream" if body.get("stream") else "application/json"
            elif self.path.endswith("/responses"):
                item = {"id": "answer-test", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": answer["text"], "annotations": []}]}
                response = {"id": "response-test", "object": "response", "model": "review-model", "created_at": 1,
                            "status": "completed", "output": [item], "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
                if body.get("stream"):
                    events = [{"type": "response.created", "response": {**response, "output": []}},
                              {"type": "response.output_text.delta", "delta": answer["text"], "output_index": 0, "content_index": 0},
                              {"type": "response.output_item.done", "item": item, "output_index": 0},
                              {"type": "response.completed", "response": response}]
                    data = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
                else:
                    data = json.dumps(response).encode()
                content_type = "text/event-stream" if body.get("stream") else "application/json"
            elif body.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                data = f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n".encode()
                content_type = "text/event-stream"
            else:
                data = json.dumps(response).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            if json.loads((plugin / "mode.json").read_text()) != "truncated":
                answer["text"] = "Verified answer."

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    db = SessionDB(db_path=tmp_path / "state.db")

    def start(mode, *, iterations=4, streaming=True, enabled=True, api_mode="chat_completions"):
        (plugin / "mode.json").write_text(json.dumps(mode))
        (tmp_path / "config.yaml").write_text(
            "model:\n  context_length: 256000\n"
            "agent:\n  max_final_continuations: 2\n  verify_on_stop: false\n"
            "display:\n  turn_completion_explainer: false\n"
            "auxiliary:\n  title_generation:\n    enabled: false\n"
            f"plugins:\n  hook_callback_timeout: {2 if mode == 'interrupt' else 0.1}\n"
            f"  enabled: {'[completion-review]' if enabled else '[]'}\n"
        )
        plugins.discover_plugins(force=True)
        assert plugins.get_plugin_manager().home_path == tmp_path
        assert plugins.has_hook("before_turn_end") is enabled
        agent = AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{server.server_port}/v1",
            provider="custom", api_mode=api_mode, model="review-model",
            quiet_mode=True, skip_memory=True, skip_context_files=True, skip_background_review=True,
            enabled_toolsets=["file"] if mode == "retrieve" else [],
            max_iterations=iterations, session_id="completion-session", session_db=db,
        )
        agent.compression_enabled = False
        agent.save_trajectories = True
        agent._cached_system_prompt = "A stable instruction."
        delivered = []
        agent.stream_delta_callback = delivered.append if streaming else None
        agent.reasoning_callback = delivered.append if streaming else None
        agent.interim_assistant_callback = lambda text, **kwargs: delivered.append(text)
        return agent, delivered

    try:
        yield start, db, requests, plugin, tmp_path
    finally:
        from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher
        shutdown_plugin_stream_hook_dispatcher()
        plugins.get_plugin_manager().unload()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        db.close()


@pytest.mark.parametrize("mode,iterations,expected_calls", [
    ("continue", 4, 2), ("repeat", 4, 3), ("budget", 1, 1),
    ("fail", 4, 1), ("veto", 4, 1), ("error", 4, 1),
    ("invalid", 4, 1), ("timeout", 4, 1),
    ("retrieve", 4, 3),
    ("reasoning", 4, 1), ("truncated", 8, 4),
])
@pytest.mark.parametrize("streaming", [True, False])
def test_rejected_answers_never_escape(gate_runtime, mode, iterations, expected_calls, streaming, caplog):
    start, db, requests, plugin, root = gate_runtime
    agent, delivered = start(mode, iterations=iterations, streaming=streaming)
    result = agent.run_conversation("Check the evidence.")
    assert len(requests) == expected_calls
    if streaming:
        assert all(request.get("stream") for request in requests)
    assert not any(delivered), "Candidate text reached a streaming or interim callback before approval"
    assert "PRIVATE_DRAFT" not in result["final_response"]
    assert result["completed"] is (mode in {"continue", "retrieve"})
    assert result["failed"] is (mode not in {"continue", "retrieve"})
    history = json.dumps(result["messages"])
    durable = json.dumps(db.get_messages("completion-session"))
    trajectories = "".join(path.read_text() for path in root.glob("*trajector*.jsonl"))
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher
    shutdown_plugin_stream_hook_dispatcher()
    observers = (plugin / "stream-observer.jsonl").read_text() if (plugin / "stream-observer.jsonl").exists() else ""
    assert "PRIVATE_DRAFT" not in observers
    assert "Verified answer." not in observers
    for text in (history, durable, trajectories):
        assert "PRIVATE_DRAFT" not in text
        assert "INTERNAL_REPAIR" not in text
        assert "PRIVATE_EXCEPTION" not in text
    seen = [json.loads(line) for line in (plugin / "seen.jsonl").read_text().splitlines()]
    assert seen[0]["messages"][0]["content"] == "Check the evidence."
    assert seen[0]["model"] == "review-model"
    assert "effort" in seen[0] and "provider" in seen[0]
    assert seen[0]["effort"] == requests[0].get("reasoning_effort")
    if mode == "retrieve":
        assert any("42" in str(row.get("content")) for row in seen[-1]["messages"] if row.get("role") == "tool")
        assert [row["role"] for row in result["messages"]] == ["user", "assistant", "tool", "assistant"]
        assert [row["role"] for row in db.get_messages("completion-session")] == ["user", "assistant", "tool", "assistant"]
    if mode == "continue":
        assert result["final_response"] == "Verified answer."
        assert [entry["attempt"] for entry in seen] == [0, 1]
        assert requests[0]["messages"][0] == requests[1]["messages"][0]
        assert [row["role"] for row in result["messages"]] == ["user", "assistant"]
        # Resume through a fresh agent with real stored history, without replaying repair scaffolding.
        agent, _ = start("allow", streaming=False)
        resumed = agent.run_conversation("Continue our discussion.", conversation_history=db.get_messages("completion-session"))
        assert resumed["completed"] is True
        assert "INTERNAL_REPAIR" not in json.dumps(requests[-1])
    if mode == "fail":
        assert result["final_response"] == "Review rejected this answer."
        assert result["failure_reason"] == "evidence_missing"
    if mode in {"error", "timeout"}:
        assert any("before_turn_end" in record.message for record in caplog.records)


@pytest.mark.parametrize("mode,enabled", [("allow", True), ("none", True), ("mutate", True), ("allow", False), ("interrupt", True)])
def test_accepted_and_unregistered_turns_preserve_history(gate_runtime, mode, enabled):
    start, db, requests, _plugin, _root = gate_runtime
    agent, delivered = start(mode, enabled=enabled)
    cancel = None
    if mode == "interrupt":
        def interrupt_when_review_starts():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if (_plugin / "seen.jsonl").exists():
                    agent.hard_interrupt("The user stopped this turn.")
                    return
                time.sleep(0.01)
        cancel = threading.Thread(target=interrupt_when_review_starts)
        cancel.start()
    result = agent.run_conversation("Check the evidence.")
    if cancel is not None:
        cancel.join(timeout=6)
        assert not cancel.is_alive()
        assert result["interrupted"] is True
        assert result["completed"] is False
        assert not delivered
        assert "PRIVATE_DRAFT" not in json.dumps(db.get_messages("completion-session"))
        return
    assert result["completed"] is True
    assert result["final_response"] == "PRIVATE_DRAFT"
    assert len(requests) == 1
    assert [row["role"] for row in result["messages"]] == ["user", "assistant"]
    assert db.get_messages("completion-session")[0]["content"] == "Check the evidence."
    assert result["messages"][0]["content"] == "Check the evidence."
    if enabled:
        assert not delivered
    else:
        assert "PRIVATE_DRAFT" in "".join(delivered)


@pytest.mark.parametrize("api_mode", ["anthropic_messages", "codex_responses"])
@pytest.mark.parametrize("mode", ["allow", "continue", "fail"])
@pytest.mark.parametrize("streaming", [True, False])
def test_policy_contract_across_transports(gate_runtime, api_mode, mode, streaming):
    start, db, requests, plugin, _root = gate_runtime
    agent, delivered = start(mode, api_mode=api_mode, streaming=streaming)
    result = agent.run_conversation("Check the evidence.")
    assert len(requests) == (2 if mode == "continue" else 1)
    assert not any(delivered)
    assert result["completed"] is (mode != "fail")
    assert result["final_response"] == {"allow": "PRIVATE_DRAFT", "continue": "Verified answer.", "fail": "Review rejected this answer."}[mode]
    if mode != "allow":
        for output in (result, db.get_messages("completion-session")):
            assert "PRIVATE_DRAFT" not in json.dumps(output)
            assert "INTERNAL_REPAIR" not in json.dumps(output)
    seen = [json.loads(line) for line in (plugin / "seen.jsonl").read_text().splitlines()]
    assert seen[-1]["can_continue"] is True


def test_compression_excludes_internal_repair_context(gate_runtime):
    start, _db, requests, _plugin, _root = gate_runtime
    agent, _delivered = start("allow", streaming=False)
    messages = []
    for index in range(12):
        messages.extend([{"role": "user", "content": f"Earlier request {index}. " * 60},
                         {"role": "assistant", "content": f"Earlier answer {index}. " * 60}])
    messages.extend([{"role": "assistant", "content": "PRIVATE_DRAFT", "_turn_end_synthetic": True},
                     {"role": "user", "content": "INTERNAL_REPAIR", "_turn_end_synthetic": True}])
    compacted, _prompt = agent._compress_context(messages, system_message="A stable instruction.", force=True)
    assert requests, "The real summary transport did not run"
    assert "PRIVATE_DRAFT" not in json.dumps(requests)
    assert "INTERNAL_REPAIR" not in json.dumps(requests)
    assert "INTERNAL_REPAIR" not in json.dumps(compacted)


@pytest.mark.parametrize("case", ["transform", "native-runtime"])
def test_unreviewed_output_cannot_bypass_policy(gate_runtime, case):
    start, db, requests, plugin, _root = gate_runtime
    agent, delivered = start("transform" if case == "transform" else "fail")
    if case == "native-runtime":
        agent.api_mode = "codex_app_server"
    result = agent.run_conversation("Check the evidence.")
    assert not any(delivered)
    assert "PRIVATE_DRAFT" not in json.dumps(result)
    assert "PRIVATE_DRAFT" not in json.dumps(db.get_messages("completion-session"))
    if case == "transform":
        assert result["final_response"] == "Public transformed answer."
        seen = json.loads((plugin / "seen.jsonl").read_text().splitlines()[0])
        assert seen["final_response"] == result["final_response"]
    else:
        assert not requests
        assert result["failed"] is True
        assert result["failure_reason"] == "final_policy_unsupported_runtime"


@pytest.mark.parametrize("mode", ["allow", "fail", "invalid", "error", "timeout", "mutate"])
def test_async_dispatch_matches_policy_isolation(gate_runtime, mode):
    from hermes_cli import plugins
    start, _db, _requests, _plugin, _root = gate_runtime
    start(mode, streaming=False)
    payload = {"final_response": "candidate", "attempt": 0,
               "messages": [{"role": "user", "content": "Original request."}]}
    results = asyncio.run(plugins.get_plugin_manager().ainvoke_hook("before_turn_end", **payload))
    assert payload["messages"][0]["content"] == "Original request."
    if mode in {"error", "timeout"}:
        assert results[0]["action"] == "fail"
        assert "PRIVATE_EXCEPTION" not in json.dumps(results)
    elif mode == "fail":
        assert results[0]["action"] == "fail"
    elif mode != "invalid":
        assert results[0]["action"] == "allow"
