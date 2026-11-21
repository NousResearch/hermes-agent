"""Regression for #157305: a primary whose client construction raises at ``AIAgent.__init__``
(a process provider plugin that can no longer import its SDK) walks ``fallback_providers``
instead of failing every turn."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import providers as _providers
from providers.base import ProviderProfile


class _Fallback(BaseHTTPRequestHandler):
    models: list = []

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
        if "messages" in req:
            type(self).models.append(req.get("model"))
        if req.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish in (({"role": "assistant", "content": "FALLBACK-ANSWER"}, None), ({}, "stop")):
                chunk = {"id": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({
            "id": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "FALLBACK-ANSWER"},
                         "finish_reason": "stop"}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


class _MissingSdkProfile(ProviderProfile):
    def create_client(self, **kwargs):
        raise ModuleNotFoundError("No module named 'directsdk'")


@pytest.fixture()
def fallback_url():
    _Fallback.models = []
    srv = HTTPServer(("127.0.0.1", 0), _Fallback)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


@pytest.fixture()
def broken_process_provider():
    _providers._discover_providers()
    snapshot = (dict(_providers._REGISTRY), dict(_providers._ALIASES), _providers._PROVIDER_LIST_CACHE)
    _providers.register_provider(_MissingSdkProfile(
        name="directsdk-test", auth_type="external_process", base_url="process://directsdk-test"))
    yield "directsdk-test"
    _providers._REGISTRY.clear()
    _providers._REGISTRY.update(snapshot[0])
    _providers._ALIASES.clear()
    _providers._ALIASES.update(snapshot[1])
    _providers._PROVIDER_LIST_CACHE = snapshot[2]


def _agent(provider, fallback_model):
    from run_agent import AIAgent

    return AIAgent(
        provider=provider, model="primary-model", api_key="process", base_url="process://directsdk-test",
        acp_command="directsdk-cli", fallback_model=fallback_model, max_iterations=2, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False, platform="cli",
    )


def test_a_primary_whose_client_cannot_be_built_answers_from_the_fallback(broken_process_provider, fallback_url):
    agent = _agent(broken_process_provider, [
        {"provider": "custom", "model": "fallback-model", "base_url": fallback_url, "api_key": "fb-key"}])
    assert (agent.provider, agent.model, agent._fallback_activated) == ("custom", "fallback-model", True)

    result = agent.run_conversation("hello", conversation_history=[], task_id="t")

    assert result["final_response"] == "FALLBACK-ANSWER"
    assert _Fallback.models == ["fallback-model"]


def test_without_a_usable_fallback_the_primary_client_error_still_surfaces(broken_process_provider):
    with pytest.raises(RuntimeError, match="directsdk"):
        _agent(broken_process_provider, None)
