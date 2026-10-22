"""Local wire-contract probe: which thinking blocks a real AIAgent replays to an
Anthropic-compatible relay. Synthetic SSE fixtures; NOT live-provider proof.

Run with the repo interpreter in an isolated HOME/HERMES_HOME:
    python evals/anthropic_proxy_thinking_replay.py <out.json>
"""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HISTORY = [
    {"role": "user", "content": "inspect the repo"},
    {"role": "assistant", "content": "checking", "reasoning_details": [
        {"type": "thinking", "thinking": "unsigned proxy reasoning"},
        {"type": "thinking", "thinking": "foreign signed", "signature": "sig-from-elsewhere"},
    ], "tool_calls": [{"id": "call_1", "type": "function",
                       "function": {"name": "terminal", "arguments": "{\"command\": \"ls\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
    {"role": "assistant", "content": "done"},
]


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def scenario(model, path="/anthropic"):
    import run_agent
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/messages"):  # skip the Ollama /api/show capability probe
                requests.append(payload)
            msg = {"id": "msg_local", "type": "message", "role": "assistant", "model": payload.get("model"),
                   "content": [], "stop_reason": None, "stop_sequence": None,
                   "usage": {"input_tokens": 10, "output_tokens": 0}}
            events = [
                {"type": "message_start", "message": msg},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "local fixture reply"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                 "usage": {"output_tokens": 3}},
                {"type": "message_stop"},
            ]
            raw = _sse(events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    agent = run_agent.AIAgent(
        model=model, provider="custom", base_url=f"http://127.0.0.1:{server.server_port}{path}",
        api_key="local-fixture-key", enabled_toolsets=["terminal"], max_iterations=1, quiet_mode=True,
        reasoning_config={"enabled": True, "effort": "high"}, skip_context_files=True, skip_memory=True,
        skip_background_review=True, save_trajectories=False, session_id=f"proxy-thinking-{model.replace('/', '-')}",
    )
    agent._api_max_retries = 1
    try:
        result = agent.run_conversation("reply briefly", system_message="Local wire probe.",
                                        conversation_history=[dict(m) for m in HISTORY])
    finally:
        agent.close()
        server.shutdown()
        server.server_close()
    wire = requests[0] if requests else {}
    assistant_turns = [m for m in wire.get("messages", []) if m.get("role") == "assistant"]
    thinking = [
        {k: b.get(k) for k in ("type", "thinking", "signature") if k in b}
        for m in assistant_turns for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if b.get("type") in ("thinking", "redacted_thinking")
    ]
    return {"model": model, "api_mode": agent.api_mode, "completed": result.get("completed"),
            "requests": len(requests), "wire_thinking_blocks": thinking,
            "thinking_param": wire.get("thinking"), "tool_count": len(wire.get("tools") or [])}


def main():
    out = {"surface": "real AIAgent + Anthropic SDK against a synthetic local relay; not provider proof",
           "scenarios": [scenario("deepseek-ai/deepseek-v4-pro"), scenario("some-vendor/other-model")]}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="proxy-thinking-") as home:
        os.environ.update(HOME=home, HERMES_HOME=home, HERMES_DISABLE_PLUGINS="1")
        main()
