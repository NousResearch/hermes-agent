"""The iteration-limit summary request is billed like any turn and must be counted like one.

A fake OpenAI-compatible server on 127.0.0.1 prices every agent request at a fixed usage; the model
keeps calling a tool until the budget is spent, then the loop sends its one extra summary request.
Session counters and the state.db row must cover every request the provider billed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from run_agent import AIAgent

PROMPT_TOKENS, COMPLETION_TOKENS = 1000, 10


class _Provider(BaseHTTPRequestHandler):
    billed: list = []

    def log_message(self, *_args):
        pass

    def _reply(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"object": "list", "data": [{"id": "tool-model", "context_length": 131072}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        if not request.get("tools"):  # side calls (session title) are not agent turns
            return self._reply({"choices": [{"index": 0, "message": {"role": "assistant", "content": "t"},
                                             "finish_reason": "stop"}]})
        self.billed.append(request)
        summary = "maximum number of tool-calling iterations" in str(request["messages"][-1].get("content"))
        message = {"role": "assistant", "content": "Summary of the work."} if summary else {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"call_{len(self.billed)}", "type": "function", "function": {
                "name": "todo", "arguments": json.dumps({"todos": [{"id": "1", "content": "x", "status": "pending"}]})}}],
        }
        self._reply({"id": f"r{len(self.billed)}", "object": "chat.completion", "model": request["model"],
                     "choices": [{"index": 0, "message": message, "finish_reason": "stop" if summary else "tool_calls"}],
                     "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS,
                               "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS}})


@pytest.fixture
def provider_url():
    _Provider.billed = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_iteration_limit_summary_request_is_counted_in_session_usage(provider_url):
    from hermes_state import SessionDB

    db = SessionDB()
    agent = AIAgent(
        model="tool-model", provider="custom", base_url=provider_url, api_key="test-key", quiet_mode=True,
        enabled_toolsets=["todo"], skip_context_files=True, skip_memory=True, max_iterations=2,
        session_id="summary-usage", session_db=db,
    )
    agent._disable_streaming = True

    result = agent.run_conversation("do the thing")

    assert result["final_response"] == "Summary of the work."
    billed = len(_Provider.billed)
    assert billed == 3  # two tool turns + the summary request
    assert (agent.session_api_calls, agent.session_prompt_tokens, agent.session_completion_tokens) == (
        billed, billed * PROMPT_TOKENS, billed * COMPLETION_TOKENS)
    db.flush_token_counts()
    row = db.get_session("summary-usage")
    assert (row["api_call_count"], row["input_tokens"], row["output_tokens"]) == (
        billed, billed * PROMPT_TOKENS, billed * COMPLETION_TOKENS)
