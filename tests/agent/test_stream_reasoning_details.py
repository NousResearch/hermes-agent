"""Opaque reasoning records survive both collectors and durable replay."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion

from agent.chat_completion_helpers import build_assistant_message, interruptible_streaming_api_call
from agent.chat_completion_helpers_relay import RelayChatAccumulator
from agent.transports.chat_completions import ChatCompletionsTransport
from persistence.conversation.messages import NewMessage
from persistence_assembly import open_persistence, profile_scope
from run_agent import AIAgent
from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_session


@pytest.mark.parametrize("collector", ["main", "relay"])
@pytest.mark.parametrize("delivery", ["final", "multiple", "absent"])
def test_stream_reasoning_details_survive_replay(tmp_path, collector, delivery):
    details = [
        {"type": "opaque.native", "version": 1, "messages": [
            {"type": "thinking", "signature": " signed+/=\n", "thinking": "α"},
            {"type": "text", "text": "hello"},
            {"type": "redacted_thinking", "data": "AA+/=="}],
         "projection": {"content": "hello world", "tool_calls": []}},
        {"type": "opaque.future", "index": 7, "unknown": [None, False, {"signature": "\r\n"}]},
    ] if delivery != "absent" else []
    deltas = [{"role": "assistant", "content": "hello"}, {"content": " world"}]
    if delivery == "multiple":
        deltas[0]["reasoning_details"] = details[:1]
        deltas.append({"reasoning_details": []})
        deltas.append({"reasoning_details": details[1:]})
    elif delivery == "final":
        deltas.append({"reasoning_details": details})
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not self.path.endswith("/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            requests.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index, delta in enumerate(deltas):
                chunk = {"id": "local-proof", "object": "chat.completion.chunk", "created": 1,
                         "model": "test-model", "choices": [{"index": 0, "delta": delta,
                         "finish_reason": "stop" if index == len(deltas) - 1 else None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    agent = AIAgent(api_key="local-test", base_url=base_url, provider="custom", model="test-model",
                    platform="subagent", quiet_mode=True, skip_memory=True, skip_context_files=True,
                    enabled_toolsets=[], save_trajectories=False)
    kwargs = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    try:
        if collector == "main":
            response = interruptible_streaming_api_call(agent, kwargs)
        else:
            acc = RelayChatAccumulator()
            with OpenAI(api_key="local-test", base_url=base_url) as client:
                with client.chat.completions.create(**kwargs, stream=True) as stream:
                    for chunk in stream:
                        acc.observe(chunk.model_dump(mode="json"))
            recorded = acc.finalize()
            recorded["choices"][0]["index"] = 0
            response = ChatCompletion.model_validate({"id": "local-proof", "created": 1,
                "object": "chat.completion", **recorded})
        assert requests and all(request["stream"] is True for request in requests)
        assert response.choices[0].message.content == "hello world"
        assert getattr(response.choices[0].message, "reasoning_details", None) == (details or None)
        normalized = ChatCompletionsTransport().normalize_response(response)
        assert normalized.reasoning_details == (details or None)
        message = build_assistant_message(agent, normalized, normalized.finish_reason)
        assert message.get("reasoning_details") == (details or None)
        with open_persistence(profile_scope(home=tmp_path)) as db:
            seed_session(db, "reasoning-proof", "cli")
            db.messages.append_batch("reasoning-proof", [NewMessage.from_live(message)])
        with open_persistence(profile_scope(home=tmp_path)) as db:
            replay = conversation(db, "reasoning-proof")
            assert replay[0].get("reasoning_details") == (details or None)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
