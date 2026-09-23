import os, sys, tempfile, json, socket, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.clear()
home = tempfile.mkdtemp(prefix="hermes-104360-")
os.environ.update(
    HOME=home,
    HERMES_HOME=home + "/.hermes",
    PATH="/usr/bin:/bin",
    PYTHONDONTWRITEBYTECODE="1",
)
Path(os.environ["HERMES_HOME"]).mkdir()
sys.dont_write_bytecode = True
repo = sys.argv[1]
sys.path.insert(0, repo)
original = socket.socket.connect
blocked = []


def connect(s, address):
    if isinstance(address, tuple) and address[0] not in ["127.0.0.1", "::1"]:
        blocked.append(str(address))
        raise RuntimeError("nonloopback blocked")
    return original(s, address)


socket.socket.connect = connect
captures = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        auth = self.headers.get("Authorization")
        status = (
            200
            if auth in ("Bearer fixture-command-token", "Bearer rotated-fixture-token")
            else 401
        )
        captures.append(dict(path=self.path, auth=auth, status=status, body=body))
        data = (
            {
                "id": "fixture",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fixture-only"},
                        "finish_reason": "stop",
                    }
                ],
            }
            if status == 200
            else {
                "error": {
                    "message": "Credential was not sent or was of an unsupported type",
                    "type": "auth_error",
                }
            }
        )
        if self.path.endswith("/messages") and status == 200:
            data = {
                "id": "fixture",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "fixture-only"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}/serving-endpoints"
import yaml

config = {
    "model": {"provider": "fixture-provider", "default": "model-a"},
    "providers": {
        "fixture-provider": {
            "base_url": url,
            "key_cmd": "/usr/bin/printf fixture-command-token",
            "models": ["model-a", "model-b"],
        }
    },
    "fallback_providers": [{"provider": "fixture-provider", "model": "model-b"}],
}
Path(os.environ["HERMES_HOME"] + "/config.yaml").write_text(
    yaml.safe_dump(config), encoding="utf-8"
)
from hermes_cli.runtime_provider import resolve_runtime_provider
from agent.auxiliary_client import resolve_provider_client

try:
    from agent.client_lifecycle import _swap_fallback_clients
except ImportError:
    from agent.chat_completion_helpers import _swap_fallback_clients
from openai import OpenAI
from types import SimpleNamespace

runtime = resolve_runtime_provider(requested="fixture-provider", target_model="model-a")
out = {"runtime_key_callable": callable(runtime["api_key"]), "cases": []}


def request(label, client, model):
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "offline fixture probe"}]
        )
        result = "200"
    except Exception as e:
        result = type(e).__name__ + ": " + str(e)
    out["cases"].append({
        "label": label,
        "result": result,
        "sdk_key": client.api_key,
        "sdk_provider_callable": callable(getattr(client, "_api_key_provider", None)),
    })


direct = OpenAI(api_key=runtime["api_key"], base_url=runtime["base_url"], max_retries=0)
request("direct-runtime", direct, "model-a")
fb, _ = resolve_provider_client("fixture-provider", model="model-b", raw_codex=True)
out["fresh_fallback_sdk_key"] = fb.api_key
out["fresh_fallback_provider_callable"] = callable(
    getattr(fb, "_api_key_provider", None)
)
agent = SimpleNamespace()
_swap_fallback_clients(
    agent, fb, "fixture-provider", "model-b", str(fb.base_url), "chat_completions"
)
out["saved_rebuild_key_callable"] = callable(agent._client_kwargs["api_key"])
request("bare-fallback-first-request", agent.client, "model-b")
rebuilt = OpenAI(**agent._client_kwargs, max_retries=0)
request("fallback-rebuilt-from-production-kwargs", rebuilt, "model-b")
# Configured timeout forces the production swap to rebuild before its first request.
config["providers"]["fixture-provider"]["request_timeout_seconds"] = 15
Path(os.environ["HERMES_HOME"] + "/config.yaml").write_text(
    yaml.safe_dump(config), encoding="utf-8"
)
from hermes_cli.config import load_config_readonly
from hermes_cli.timeouts import get_provider_request_timeout

out["timeout_resolved"] = get_provider_request_timeout("fixture-provider", "model-b")
fb2, _ = resolve_provider_client("fixture-provider", model="model-b", raw_codex=True)
from agent.client_lifecycle import ClientLifecycleMixin


class ProbeAgent(ClientLifecycleMixin):
    pass


a2 = ProbeAgent()
a2.provider = "fixture-provider"
a2.model = "model-b"
a2.base_url = str(fb2.base_url)
_swap_fallback_clients(
    a2, fb2, "fixture-provider", "model-b", str(fb2.base_url), "chat_completions"
)
request("timeout-fallback-first-request", a2.client, "model-b")
# The rebuild must retain the live source, not just the most recent minted key.
token_path = Path(home) / "rotating-token"
token_path.write_text("fixture-command-token")
for dynamic in (False, True):
    source = (lambda: token_path.read_text()) if dynamic else "fixture-command-token"
    fresh = OpenAI(api_key=source, base_url=url, max_retries=0)
    holder = SimpleNamespace()
    _swap_fallback_clients(
        holder, fresh, "fixture-static", "model-b", url, "codex_responses"
    )
    derived = OpenAI(**holder._client_kwargs, max_retries=0)
    for token in ("fixture-command-token", "rotated-fixture-token"):
        token_path.write_text(token)
        request(f"rotation-{dynamic}-{token}", derived, "model-b")
        assert captures[-1]["auth"] == "Bearer " + (
            token if dynamic else "fixture-command-token"
        )
    fresh.close()
    derived.close()
token_path.write_text("fixture-command-token")
source = lambda: token_path.read_text()
fresh = OpenAI(api_key=source, base_url=url, max_retries=0)
holder = SimpleNamespace()
_swap_fallback_clients(holder, fresh, "anthropic", "model-b", url, "anthropic_messages")
for token in ("fixture-command-token", "rotated-fixture-token"):
    token_path.write_text(token)
    response = holder._anthropic_client.messages.create(
        model="model-b", max_tokens=8, messages=[{"role": "user", "content": "fixture"}]
    )
    assert response.content[0].text == "fixture-only"
    assert captures[-1]["auth"] == "Bearer " + token
out["anthropic_rotating_handoff"] = "200,200"
fresh.close()
holder._anthropic_client.close()
out["captures"] = captures
out["blocked"] = blocked
out["production_imports"] = {
    n: sys.modules[n].__file__
    for n in [
        "hermes_cli.runtime_provider",
        "agent.auxiliary_client",
        "agent.client_lifecycle",
    ]
}
print(json.dumps(out, indent=2))
server.shutdown()
assert captures[0]["status"] == 200
assert captures[1]["status"] == 200
assert captures[2]["status"] == 200
assert captures[3]["status"] == 200
