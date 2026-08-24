"""Credential-free named-route reasoning parity probe over localhost SDK HTTP."""

import json
import os
import sys
from pathlib import Path

import tempfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading

REPO = Path(sys.argv[1]).resolve()
arm = sys.argv[2]
home = tempfile.mkdtemp(prefix="named-wire-")
os.environ.clear()
os.environ.update(
    HOME=home, HERMES_HOME=home + "/.hermes", PATH="/usr/bin:/bin", NO_PROXY="*"
)
os.chdir(home)
sys.dont_write_bytecode = True
captures = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/chat/completions"):
            captures.append(body)
        data = json.dumps({
            "id": "fixture",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", body.get("name", "fixture")),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "fixture"},
                    "finish_reason": "stop",
                }
            ],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
sys.path.insert(0, str(REPO))
blocked = []


def audit(event, args):
    if event == "socket.getaddrinfo" and args[0] not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        blocked.append(event)
        raise RuntimeError("Offline probe: networking forbidden")


sys.addaudithook(audit)
assert Path(os.environ["HOME"]) != Path("/home/teknium")
assert Path(os.environ["HERMES_HOME"]).is_relative_to(Path(os.environ["HOME"]))
assert not any(k for k in os.environ if "API_KEY" in k or "TOKEN" in k or "SECRET" in k)

base_url = f"http://127.0.0.1:{server.server_port}/v1"
models = {
    "my-vllm-model": {"context_length": 204800, "reasoning_format": "vllm"},
    "glm-5.3": {"context_length": 180000, "reasoning_format": "zai"},
    "no-reasoning-model": {"context_length": 131072, "reasoning_format": "none"},
    "undeclared-model": {"context_length": 131072},
}
config = {
    "model": {
        "provider": "custom:my-gateway",
        "default": "my-vllm-model",
        "base_url": base_url,
    },
    "custom_providers": [
        {"name": "my-gateway", "base_url": base_url, "models": models}
    ],
    "memory": {"enabled": False},
    "skills": {"creation_nudge_interval": 0},
    "agent": {"reasoning_effort": "high"},
}
Path(os.environ["HERMES_HOME"]).mkdir(parents=True, exist_ok=True)
# JSON is valid YAML, allowing the real config loader to read the seeded file.
(Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(json.dumps(config))
from run_agent import AIAgent
from providers import get_provider_profile
from hermes_cli.config import get_compatible_custom_providers, load_config
from hermes_cli.config_providers import get_custom_provider_context_length
from agent.transports.chat_completions import ChatCompletionsTransport

loaded = get_compatible_custom_providers(load_config())
rows = []
for provider in ["custom:my-gateway", "custom"]:
    for model in models:
        agent = AIAgent(
            model=model,
            provider=provider,
            base_url=base_url,
            api_key="offline-placeholder",
            api_mode="chat_completions",
            enabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            skip_background_review=True,
            skip_context_files=True,
            save_trajectories=False,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        kwargs = agent._build_api_kwargs(
            [{"role": "user", "content": "offline payload only"}], []
        )
        profile = get_provider_profile(agent.provider)
        rows.append({
            "provider": agent.provider,
            "model": agent.model,
            "declared_format": models[model].get("reasoning_format"),
            "profile": None if profile is None else type(profile).__name__,
            "agent_has_custom_models": any(
                e.get("models") == models for e in agent._custom_providers
            ),
            "configured_context_length": agent._config_context_length,
            "supports_reasoning_extra_body": agent._supports_reasoning_extra_body(),
            "kwargs": kwargs,
        })
        agent.client.chat.completions.create(**kwargs)
        agent.client.close()

# Direct profile-path controls: even passing custom_providers to the transport
# cannot activate the proposed key; the shipped code has no such consumer.
transport = ChatCompletionsTransport()
profile_controls = []
for model in models:
    kwargs = transport.build_kwargs(
        model=model,
        messages=[{"role": "user", "content": "offline"}],
        provider_profile=get_provider_profile("custom"),
        base_url=base_url,
        reasoning_config={"enabled": True, "effort": "high"},
        custom_providers=loaded,
    )
    profile_controls.append({"model": model, "kwargs": kwargs})

named = [r for r in rows if r["provider"].startswith("custom:")]
generic = [r for r in rows if r["provider"] == "custom"]
assert all(r["agent_has_custom_models"] for r in rows)
assert all(
    r["configured_context_length"] == models[r["model"]]["context_length"] for r in rows
)
assert all(("reasoning_effort" in r["kwargs"]) == (arm == "fixed") for r in named)
assert all(r["kwargs"].get("reasoning_effort") == "high" for r in generic)
assert all(r["kwargs"].get("reasoning_effort") == "high" for r in profile_controls)
assert len(captures) == len(rows)
for request, row in zip(captures, rows):
    assert request["model"] == row["model"]
    assert request.get("reasoning_effort") == row["kwargs"].get("reasoning_effort")
print(
    json.dumps(
        {
            "status": "PASS: named-route parity restored" if arm == "fixed" else "PASS: named-route reasoning loss reproduced",
            "repo": str(REPO),
            "home": os.environ["HOME"],
            "hermes_home": os.environ["HERMES_HOME"],
            "captures": captures,
            "rows": rows,
            "profile_controls": profile_controls,
            "network_attempts_blocked": blocked,
            "scope": "Real AIAgent init and request construction plus localhost SDK HTTP; fixture responses, no vendor inference. Per-model reasoning_format is unsupported on both arms, including none.",
        },
        indent=2,
        default=str,
    )
)

server.shutdown()
