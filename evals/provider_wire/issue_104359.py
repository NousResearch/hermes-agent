"""Offline production-path wire probe; only loopback HTTP is permitted."""

import os, sys, tempfile, json, socket, threading, copy
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

root = Path(sys.argv[1]).resolve()
home = tempfile.mkdtemp(prefix="hermes-104359-")
os.environ.clear()
os.environ.update(
    HOME=home,
    HERMES_HOME=home + "/.hermes",
    PATH="/usr/bin:/bin",
    PYTHONDONTWRITEBYTECODE="1",
    NO_PROXY="*",
)
sys.dont_write_bytecode = True
sys.path.insert(0, str(root))
os.chdir(home)
Path(os.environ["HERMES_HOME"]).mkdir()
# Prevent accidental provider/model metadata egress, even during production imports.
connect = socket.socket.connect
blocked = []


def local_connect(self, address):
    if isinstance(address, tuple) and address[0] not in (
        "127.0.0.1",
        "::1",
        "localhost",
    ):
        blocked.append(str(address))
        raise RuntimeError("Non-loopback network blocked")
    return connect(self, address)


socket.socket.connect = local_connect
captures = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        leaks = [
            f"messages.{i}.{k}"
            for i, m in enumerate(body.get("messages", []))
            for k in m
            if k.startswith("_")
        ]
        status = 400 if leaks else 200
        captures.append({
            "path": self.path,
            "status": status,
            "leaks": leaks,
            "body": body,
        })
        response = (
            {
                "error": {
                    "message": leaks[0] + ": Extra inputs are not permitted",
                    "type": "invalid_request_error",
                }
            }
            if leaks
            else {
                "id": "local-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "probe-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "local fixture response",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
        data = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}/v1"
config = {
    "model": {"provider": "probe-local", "default": "probe-model"},
    "providers": {
        "probe-local": {
            "base_url": url,
            "api_key": "no-key-required",
            "api_mode": "chat_completions",
        }
    },
    "moa": {
        "default_preset": "probe",
        "presets": {
            "probe": {
                "enabled": True,
                "reference_models": [
                    {"provider": "probe-local", "model": "probe-model"}
                ],
                "aggregator": {"provider": "probe-local", "model": "probe-model"},
            }
        },
    },
    "prompt_caching": {"enabled": False},
}
import yaml

Path(os.environ["HERMES_HOME"], "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
from hermes_state import SessionDB
from agent.moa_loop import MoAClient
from agent.turn_request_assembly import _prepare_moa_request
from agent.transports.chat_completions import ChatCompletionsTransport
from types import SimpleNamespace
from openai import OpenAI

db = SessionDB(Path(os.environ["HERMES_HOME"]) / "state.db")
db.create_session("probe-session", source="cli")
db.append_message("probe-session", "user", "Remember this token")
db.append_message("probe-session", "assistant", "Acknowledged")
history = db.get_messages_as_conversation("probe-session")
assert all(m.get("_db_persisted") for m in history)
messages = [
    {"role": "system", "content": "You are a test assistant"},
    *history,
    {"role": "user", "content": "Continue"},
]
original = copy.deepcopy(messages)
client = MoAClient("probe")
prepared, api_messages, pending = _prepare_moa_request(
    SimpleNamespace(client=client), messages, None
)
transport = ChatCompletionsTransport()
kwargs = transport.build_kwargs("probe", api_messages)
assert not any("_db_persisted" in m for m in kwargs["messages"])
assert any("_db_persisted" in m for m in prepared["messages"])
kwargs["_moa_prepared_request"] = prepared
outcomes = {}
try:
    client.chat.completions.create(**kwargs)
    outcomes["moa"] = "success"
except Exception as exc:
    outcomes["moa"] = {"type": type(exc).__name__, "message": str(exc)}
# Identical request after the normal transport sanitation succeeds on same strict local endpoint.
normal = OpenAI(base_url=url, api_key="no-key-required", max_retries=0)
result = normal.chat.completions.create(
    **transport.build_kwargs("probe-model", messages)
)
outcomes["direct"] = result.choices[0].message.content
assert messages == original
assert (outcomes["moa"] == "success") == (sys.argv[2] == "fixed")
assert captures[-1]["status"] == 200
assert any(
    c["status"] == 200 and "advis" in json.dumps(c["body"]).lower()
    for c in captures[:-1]
)
print(
    json.dumps(
        {
            "repo": str(root),
            "isolated_home": home,
            "db_loaded_messages": history,
            "prepared_marker_present": True,
            "normal_transport_marker_absent": True,
            "history_unchanged": messages == original,
            "outcomes": outcomes,
            "captures": captures,
            "blocked_nonloopback_attempts": blocked,
            "production_imports": {
                name: sys.modules[name].__file__
                for name in [
                    "hermes_state",
                    "agent.moa_loop",
                    "agent.auxiliary_client",
                    "agent.turn_request_assembly",
                    "agent.transports.chat_completions",
                ]
            },
        },
        indent=2,
    )
)
server.shutdown()
db.close()
