"""Reasoning streamed as typed ``thinking`` parts inside ``delta.content`` (Mistral) is kept."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI

from agent.auxiliary_client import _aggregate_chat_stream
from agent.chat_completion_helpers import interruptible_streaming_api_call
from agent.chat_completion_helpers_relay import RelayChatAccumulator
from run_agent import AIAgent

# Mistral's documented reasoning shape: ThinkChunks (``thinking`` is a list of TextChunks), then
# TextChunk lists, then plain strings, all in one stream.
_DELTAS = [
    {"role": "assistant", "content": ""},
    {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "Two plus "}]}]},
    {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "two is four."}]}]},
    {"content": [{"type": "text", "text": "The answer"}]},
    {"content": " is 4."},
]


@pytest.mark.parametrize("collector", ["main", "relay", "aux"])
def test_streamed_thinking_parts_become_reasoning_not_dropped(collector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index, delta in enumerate(_DELTAS):
                chunk = {"id": "mistral-proof", "object": "chat.completion.chunk", "created": 1,
                         "model": "mistral-large-4-0", "choices": [{"index": 0, "delta": delta,
                         "finish_reason": "stop" if index == len(_DELTAS) - 1 else None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    kwargs = {"model": "mistral-large-4-0", "messages": [{"role": "user", "content": "2+2?"}]}
    try:
        if collector == "main":
            agent = AIAgent(api_key="local-test", base_url=base_url, provider="custom",
                            model="mistral-large-4-0", platform="subagent", quiet_mode=True,
                            skip_memory=True, skip_context_files=True, enabled_toolsets=[],
                            save_trajectories=False)
            message = interruptible_streaming_api_call(agent, kwargs).choices[0].message
            content, reasoning = message.content, message.reasoning_content
        else:
            with OpenAI(api_key="local-test", base_url=base_url) as client:
                stream = client.chat.completions.create(**kwargs, stream=True)
                if collector == "aux":
                    message = _aggregate_chat_stream(stream, model="mistral-large-4-0").choices[0].message
                    content, reasoning = message.content, message.reasoning
                else:
                    acc = RelayChatAccumulator()
                    for chunk in stream:
                        acc.observe(chunk.model_dump(mode="json"))
                    message = acc.finalize()["choices"][0]["message"]
                    content, reasoning = message["content"], message["reasoning_content"]
        assert content == "The answer is 4."
        assert reasoning == "Two plus two is four."
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
