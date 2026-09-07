import os, sys, json, asyncio, threading, tempfile, sqlite3, socket
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from types import SimpleNamespace

ROOT = Path(sys.argv[1]).resolve()
RECEIPT = Path(sys.argv[2]).resolve()
HOME = Path(tempfile.mkdtemp(prefix="hermes-104653-state-"))
# Discard inherited credentials/config, preserve only interpreter essentials.
keep = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "TZ")}
os.environ.clear()
os.environ.update(keep)
os.environ.update(
    HOME=str(HOME),
    HERMES_HOME=str(HOME),
    HERMES_DISABLE_PLUGINS="1",
    NO_PROXY="127.0.0.1,localhost",
)
sys.path.insert(0, str(ROOT))
os.chdir(HOME)
(HOME / "config.yaml").write_text(
    "model:\n  provider: openai-compat\n  default: fixture-model\n  context_length: 131072\nagent:\n  max_iterations: 2\ncompression:\n  enabled: false\ndatabase:\n  journal_mode: delete\n"
)
# Fence all network calls to loopback, including optional discovery/aux paths.
orig_connect = socket.socket.connect
blocked = []


def local_connect(self, address):
    if isinstance(address, tuple) and address[0] not in (
        "127.0.0.1",
        "::1",
        "localhost",
    ):
        blocked.append(str(address))
        raise OSError("fixture forbids external network")
    return orig_connect(self, address)


socket.socket.connect = local_connect
requests = []


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append({"path": self.path, "body": body})
        users = [
            m.get("content") for m in body.get("messages", []) if m["role"] == "user"
        ]
        fail = fault == "http400"
        if fail:
            data = {
                "error": {
                    "message": "fixture invalid request",
                    "type": "invalid_request_error",
                    "code": "fixture_rejected",
                }
            }
            status = 400
        else:
            data = {
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fixture reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                    "total_tokens": 104,
                },
            }
            status = 200
        if body.get("stream") and not fail:
            chunks = [
                {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "fixture reply"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": data["usage"],
                },
            ]
            payload = (
                "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n"
            ).encode()
            mime = "text/event-stream"
        else:
            payload = json.dumps(data).encode()
            mime = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
threading.Thread(target=server.serve_forever, daemon=True).start()
from run_agent import AIAgent
from hermes_state import SessionDB
from gateway.session import SessionStore, AsyncSessionStore, SessionSource
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner


import gateway.run as gateway_run

fault = None
faults = []
runtime = dict(
    api_key="fixture-only",
    base_url=f"http://127.0.0.1:{server.server_port}/v1",
    provider="openai-compat",
    api_mode="chat_completions",
)
gateway_run._resolve_runtime_agent_kwargs = lambda: runtime
import run_agent


class ControlledAgent(AIAgent):
    def __init__(self, *args, **kwargs):
        if fault == "pre-agent":
            faults.append("agent-construction")
            raise RuntimeError("controlled agent initialization failure")
        kwargs["enabled_toolsets"] = []
        super().__init__(*args, **kwargs)


run_agent.AIAgent = ControlledAgent
source = SessionSource(
    platform=Platform.TELEGRAM,
    chat_id="-100104653",
    chat_type="group",
    user_id="104653",
)
runner = GatewayRunner(GatewayConfig())
runner._recover_telegram_topic_thread_id = lambda source: None
runner._is_user_authorized = lambda *a: True


def voice_policy(*args, **kwargs):
    if fault == "post-agent":
        faults.append("voice-policy-after-persistence")
        raise RuntimeError("controlled voice policy failure")
    return False


runner._should_send_voice_reply = voice_policy
from gateway.session_transcript import TranscriptReadError

original_load = runner.session_store.load_transcript


def controlled_load(session_id, **kwargs):
    if fault == "raw-read" and kwargs.get("raw"):
        faults.append("raw-read")
        raise TranscriptReadError(session_id)
    return original_load(session_id, **kwargs)


runner.session_store.load_transcript = controlled_load
entry = runner.session_store.get_or_create_session(source)
sid = entry.session_id


def rows():
    with sqlite3.connect(HOME / "state.db") as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in conn.execute(
                "SELECT id,role,content,platform_message_id,timestamp FROM messages "
                "WHERE session_id=? ORDER BY id",
                (sid,),
            )
        ]


async def main():
    global fault
    from datetime import datetime

    observations = []
    expected = []
    cases = [
        ("warmup", "100", None, True),
        ("same happy input", "101", None, True),
        ("same happy input", "102", None, True),
        ("synthetic happy", None, None, True),
        ("synthetic happy", None, None, True),
        ("failed provider input", "103", "http400", True),
        ("after failure", "104", "post-agent", True),
        ("failed provider input", "105", "http400", True),
        ("synthetic post-agent", None, "post-agent", True),
        ("init failure", "106", "pre-agent", True),
        ("init failure", "106", "pre-agent", False),
        ("init failure", "107", "pre-agent", True),
        ("synthetic init failure", None, "pre-agent", True),
        ("synthetic init failure", None, "pre-agent", True),
        ("synthetic init failure", None, "raw-read", False),
        ("healthy follow-up", "108", None, True),
    ]
    # Deliberately identical timestamps: independently accepted keyless turns must survive too.
    timestamp = datetime.fromtimestamp(1700000000)
    for text, pid, fault, accepted in cases:
        event = MessageEvent(
            text=text,
            source=source,
            message_id=pid,
            internal=pid is None,
            timestamp=timestamp,
        )
        before_requests = len(requests)
        before_faults = len(faults)
        runner._evict_cached_agent(entry.session_key)
        response = await runner._handle_message(event)
        if accepted:
            expected.append((text, pid))
        actual = [
            (r["content"], r["platform_message_id"])
            for r in rows()
            if r["role"] == "user"
        ]
        hit = faults[before_faults:]
        if fault == "raw-read":
            reached = (
                hit == ["raw-read"]
                and len(requests) == before_requests
                and "history is temporarily unavailable" in response
            )
        elif fault == "pre-agent":
            reached = hit == ["agent-construction"] and len(requests) == before_requests
        elif fault == "post-agent":
            reached = (
                hit == ["voice-policy-after-persistence"]
                and len(requests) > before_requests
            )
        else:
            reached = len(requests) > before_requests
        observations.append(
            dict(
                text=text,
                platform_id=pid,
                fault=fault,
                accepted=accepted,
                expected_users=len(expected),
                actual_users=len(actual),
                pass_=actual == expected,
                reached=reached,
                faults=hit,
                requests=len(requests) - before_requests,
                response=response,
            )
        )
    receipt = dict(
        home=str(HOME),
        observations=observations,
        rows=rows(),
        requests=len(requests),
        blocked_external_attempts=blocked,
        passed=sum(o["pass_"] and o["reached"] for o in observations),
        total=len(observations),
    )
    RECEIPT.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return receipt["passed"] == receipt["total"]


try:
    passed = asyncio.run(main())
finally:
    server.shutdown()
sys.exit(0 if passed else 1)
