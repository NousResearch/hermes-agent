"""Zero-cost local SDK wire capture. Fixtures are not vendor inference evidence.

Run from a checkout with HERMES_HOME pointing to a disposable directory.
"""
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai import OpenAI
from anthropic import Anthropic

captures = []


class Capture(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "fixture", "context_length": 131072}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        captures.append({"path": self.path, "body": body})
        if "model" not in body:
            reply = {}
        elif self.path.endswith("messages"):
            reply = {"id": "fixture", "type": "message", "role": "assistant", "model": body["model"], "content": [{"type": "text", "text": "LOCAL_CAPTURE_ONLY"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            reply = {"id": "fixture", "object": "chat.completion", "created": 0, "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "LOCAL_CAPTURE_ONLY"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        self.send_response(200)
        if body.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0, "model": body["model"], "choices": [{"index": 0, "delta": {"content": "LOCAL_CAPTURE_ONLY"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
        else:
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

    def log_message(self, format, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), Capture)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = f"http://127.0.0.1:{server.server_port}"
home = Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True, exist_ok=True)
config = {"model": {"default": "fixture", "provider": "fixture-local", "max_tokens": 17}, "providers": {"fixture-local": {"api": url + "/v1", "api_key": "fixture", "max_output_tokens": 19}}}
(home / "config.yaml").write_text(json.dumps(config))
os.environ["HERMES_MAX_TOKENS"] = "13"
from gateway.run import _resolve_runtime_agent_kwargs
from gateway.platforms.api_server import _resolve_request_runtime_agent_kwargs
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_cli.moa_config import _normalize_preset
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.anthropic import AnthropicTransport
from agent.transports.bedrock import BedrockTransport

messages = [{"role": "user", "content": "fixture"}]
client = OpenAI(api_key="fixture", base_url=url + "/v1", max_retries=0)
results = {}
for label, runtime in [("gateway", _resolve_runtime_agent_kwargs()), ("api-server", _resolve_request_runtime_agent_kwargs("fixture-local", "fixture")), ("provider", resolve_runtime_provider(requested="fixture-local", target_model="fixture"))]:
    cap = runtime.get("max_tokens", runtime.get("max_output_tokens"))
    kwargs = ChatCompletionsTransport().build_kwargs("fixture", messages, max_tokens=cap, max_tokens_param_fn=lambda value: {"max_tokens": value})
    client.chat.completions.create(**kwargs)
    results[label] = captures[-1]
for label, params in [("compatible-claude", {"anthropic_max_output": 65536}), ("internal-budget", {"max_tokens": 43})]:
    kwargs = ChatCompletionsTransport().build_kwargs("claude-fixture", messages, max_tokens_param_fn=lambda value: {"max_tokens": value}, **params)
    client.chat.completions.create(**kwargs)
    results[label] = captures[-1]
from providers import get_provider_profile
from run_agent import AIAgent
agent = AIAgent(model="fixture", provider="custom", base_url=url + "/v1", api_key="fixture", quiet_mode=True, skip_memory=True, skip_context_files=True, enabled_toolsets=[])
actual_kwargs = agent._build_api_kwargs(messages, tools_for_api=[])
client.chat.completions.create(**actual_kwargs)
results["agent-main-custom"] = captures[-1]
kwargs = ChatCompletionsTransport().build_kwargs("fixture", messages, provider_profile=get_provider_profile("custom"), max_tokens_param_fn=lambda value: {"max_tokens": value})
client.chat.completions.create(**kwargs)
results["registered-custom-profile"] = captures[-1]
from agent.auxiliary_client import _compression_fast_lane_controls
from tools.delegate_tool_config import _resolve_delegation_credentials
route = {"provider": "custom", "model": "fixture", "reasoning_effort": "none", "max_output_tokens": 47}
compression_cap, _ = _compression_fast_lane_controls(
    "compression", actual_provider="custom", actual_model="fixture", requested_provider="custom",
    requested_model="fixture", route_config=route, leak_guard_config=route, max_tokens=None, extra_body={},
)
child_credentials = _resolve_delegation_credentials({"provider": "fixture-local", "model": "fixture"}, agent)
for label, cap in [("compression-config-helper", compression_cap), ("delegation-config-helper", child_credentials.get("max_output_tokens"))]:
    kwargs = ChatCompletionsTransport().build_kwargs("fixture", messages, max_tokens=cap, max_tokens_param_fn=lambda value: {"max_tokens": value})
    client.chat.completions.create(**kwargs)
    results[label] = captures[-1]
kwargs = AnthropicTransport().build_kwargs("claude-sonnet-4-5", messages)
kwargs.pop("__anthropic__", None)
Anthropic(api_key="fixture", base_url=url, max_retries=0).messages.create(**kwargs)
results["native-anthropic"] = captures[-1]
results["bedrock-converse-built-not-sent"] = BedrockTransport().build_kwargs("amazon.nova-pro-v1:0", messages)
results["moa-normalized"] = _normalize_preset({"max_tokens": 23, "reference_max_tokens": 29})
if os.environ.get("FULL_OUTPUT_CAP_SURFACES"):
    from output_caps_surfaces import exercise_surfaces
    results.update(exercise_surfaces(agent, captures, url, home, config))
print(json.dumps(results, indent=2, default=str))
server.shutdown()
server.server_close()
thread.join()
if os.environ.get("VERIFY_OUTPUT_CAP_REMOVAL"):
    for label in ("gateway", "api-server", "provider", "compatible-claude", "agent-main-custom", "registered-custom-profile", "compression-config-helper", "delegation-config-helper"):
        assert "max_tokens" not in results[label]["body"], label
    assert results["internal-budget"]["body"]["max_tokens"] == 43
    assert results["native-anthropic"]["body"]["max_tokens"] > 0
    assert "maxTokens" not in results["bedrock-converse-built-not-sent"].get("inferenceConfig", {})
