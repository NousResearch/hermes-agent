"""Background wake turns and the model fallback notice (#137738).

A wake turn (no user message: a finished process, a plugin's scheduled check) shows no switch
notice, since most end silent and an outage would post one per wake. A hidden switch is never an
undisclosed one: when the fallback is still serving the next turn the user sends, that turn
announces it before it runs. Driven through real ``run_conversation`` turns against a loopback
provider whose primary model answers 402 (a quota failure keeps the fallback across turns).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from run_agent import AIAgent

WAKE = "internal_notification"


class _Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") == "primary-model":
            data = json.dumps({"error": {"message": "Insufficient credits", "type": "billing_error",
                                         "code": "insufficient_quota"}}).encode()
            self.send_response(402)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
        data = b"".join(b"data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "created": 0,
                                                "model": body.get("model"), **c}).encode() + b"\n\n"
                        for c in chunks) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def agent_and_notices():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    shown = []
    agent = AIAgent(
        model="primary-model", provider="custom", base_url=url, api_key="k", api_mode="chat_completions",
        fallback_model={"provider": "custom", "model": "backup-model", "base_url": url, "api_key": "k",
                        "api_mode": "chat_completions"},
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
        status_callback=lambda kind, text: shown.append(str(text)),
    )
    agent.compression_enabled = False
    yield agent, shown
    server.shutdown()
    server.server_close()


def _turn(agent, shown, kind=None):
    shown.clear()
    kwargs = {"persist_user_display_kind": kind} if kind else {}
    result = agent.run_conversation("check in", conversation_history=[], **kwargs)
    assert result.get("final_response") == "ok"
    return [line for line in shown if "Model fallback" in line or "Primary model restored" in line]


def test_wake_turns_stay_quiet_and_the_next_user_turn_discloses_the_fallback(agent_and_notices):
    agent, shown = agent_and_notices
    assert _turn(agent, shown, WAKE) == []
    assert _turn(agent, shown, WAKE) == []
    # The quota cooldown keeps the backup serving: the user's turn is told before it runs, once.
    notices = _turn(agent, shown)
    assert len(notices) == 1 and "backup-model" in notices[0] and "Insufficient credits" in notices[0]
    assert _turn(agent, shown) == []


def test_wake_turn_after_cooldown_posts_no_restore_or_switch_chatter(agent_and_notices):
    agent, shown = agent_and_notices
    assert _turn(agent, shown, WAKE) == []
    agent._rate_limited_until = 0  # cooldown over: the next turn retries the (still failing) primary
    assert _turn(agent, shown, WAKE) == []
    agent._rate_limited_until = 0
    # A user turn retrying the primary sees the switch it causes, and no restore line for a switch
    # it never saw.
    notices = _turn(agent, shown)
    assert len(notices) == 1 and notices[0].startswith("⚠️ Model fallback")
