"""A real ``hermes gateway run``: each conversation is governed by its own agent's policy.

One gateway process serves every agent (multiplexed profiles, ``/p/<agent>/`` on the API
server). What this proves, against a scripted model asking to read a file:

- an agent's conversation runs under THAT agent's NOVA policy — outside its area, refused;
  inside its own profile, allowed (so the refusal is the rule, not a broken tool);
- the gateway's own default profile (``/``), which is not an agent, refuses every tool —
  before the fix it read /etc/hostname with no policy at all;
- the default profile's refuse-everything does not leak into the agents' conversations.

Skipped where the runtime (the ``hermes`` executable) is not installed.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

HERMES = shutil.which("hermes") or next(
    (p for p in ("/opt/hermes/bin/hermes", "/opt/hermes/.venv/bin/hermes") if os.path.exists(p)), None)
pytestmark = pytest.mark.skipif(HERMES is None, reason="needs the hermes runtime executable")
OUTSIDE = "/etc/hostname"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Model(BaseHTTPRequestHandler):
    """Reads the path named last in the user's message, then reports the tool result."""

    def log_message(self, *args):
        pass

    def _send(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({"data": [{"id": "acme-default", "object": "model", "context_length": 200000}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        messages = req.get("messages") or []
        results = [m for m in messages if m.get("role") == "tool"]
        if not req.get("tools"):
            msg, finish = {"role": "assistant", "content": "ok"}, "stop"
        elif not results:
            user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
            text = user.get("content") if isinstance(user.get("content"), str) else ""
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": text.split()[-1]})}}]}
            finish = "tool_calls"
        else:
            msg, finish = {"role": "assistant", "content": "RESULT " + str(results[-1].get("content"))}, "stop"
        body = {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": "acme-default",
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            delta = {"role": "assistant", **({"tool_calls": [dict(msg["tool_calls"][0], index=0)]}
                                             if msg.get("tool_calls") else {"content": msg["content"]})}
            for chunk in ({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": body["usage"]}):
                chunk.update({"id": "x", "object": "chat.completion.chunk", "created": body["created"], "model": "acme-default"})
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._send(body)


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    from nova.apply import apply_bundle
    from nova.audit import AuditLog
    from nova.runtime.hermes import HermesRuntime
    from nova.spec import load_bundle

    from .conftest import EXAMPLE_BUNDLE

    model = ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    model_url = f"http://127.0.0.1:{model.server_address[1]}/v1"

    work = tmp_path_factory.mktemp("gateway")
    home = work / "home"
    home.mkdir()
    bundle_dir = work / "bundle"
    shutil.copytree(EXAMPLE_BUNDLE, bundle_dir)
    (bundle_dir / "channels.yaml").write_text("channels: []\n")
    env = {**os.environ, "HERMES_HOME": str(home), "NOVA_HOME": str(home),
           "ACME_LLM_URL": model_url, "ACME_LLM_KEY": "test-key"}
    saved = {k: os.environ.get(k) for k in ("HERMES_HOME", "NOVA_HOME", "ACME_LLM_URL", "ACME_LLM_KEY")}
    os.environ.update({k: env[k] for k in saved})
    try:
        bundle = load_bundle(bundle_dir)
        runtime = HermesRuntime(home=home, tenant_id=bundle.tenant_id)
        apply_bundle(bundle, runtime, audit=AuditLog(home / "audit.jsonl", tenant_id=bundle.tenant_id))
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    key, port = secrets.token_hex(32), _free_port()
    config_path = home / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    config.setdefault("gateway", {})["multiplex_profiles"] = True
    config.setdefault("platforms", {})["api_server"] = {"enabled": True, "host": "127.0.0.1", "port": port, "key": key}
    config_path.write_text(yaml.safe_dump(config))
    # The gateway resolves ${ACME_*} and each profile's API key from .env files.
    for dotenv in [home / ".env", *(p / ".env" for p in (home / "profiles").iterdir() if p.is_dir())]:
        with dotenv.open("a") as fh:
            fh.write(f"\nACME_LLM_KEY=test-key\nACME_LLM_URL={model_url}\nAPI_SERVER_KEY={key}\n")

    log = (work / "gateway.log").open("w")
    process = subprocess.Popen([HERMES, "gateway", "run"], stdout=log, stderr=subprocess.STDOUT, env=env)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{base}/health", timeout=2)
            break
        except Exception:
            if process.poll() is not None:
                break
            time.sleep(1)
    else:
        pytest.fail("gateway did not come up")
    try:
        yield base, key, home
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        model.shutdown()


def ask(gateway, prefix: str, path: str) -> str:
    base, key, _ = gateway
    request = urllib.request.Request(
        f"{base}{prefix}/v1/chat/completions", method="POST",
        data=json.dumps({"model": "acme-default",
                         "messages": [{"role": "user", "content": f"please read {path}"}]}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["choices"][0]["message"]["content"]


@pytest.mark.parametrize("agent", ["operations", "customer-support"])
def test_an_agents_conversation_is_held_to_its_own_policy(gateway, agent):
    answer = ask(gateway, f"/p/{agent}", OUTSIDE)
    assert "BLOCKED by NOVA policy" in answer and "outside it" in answer


@pytest.mark.parametrize("agent", ["operations", "customer-support"])
def test_inside_its_own_profile_the_same_tool_works(gateway, agent):
    """Also the proof that the default profile's refuse-everything has not leaked in."""
    _, _, home = gateway
    answer = ask(gateway, f"/p/{agent}", str(home / "profiles" / agent / "SOUL.md"))
    assert answer.startswith("RESULT") and "BLOCKED" not in answer


def test_the_gateways_own_default_profile_refuses_everything(gateway):
    answer = ask(gateway, "", OUTSIDE)
    assert "BLOCKED by NOVA policy" in answer and "not a NOVA agent" in answer
