"""Regression for #138469: a credential-pool rotation carries the compressor's key with the main route.

The compressor sends its own copy of the main key on summary calls (the ``auto`` route's
``main_runtime``, and the latched fallback pin). ``_swap_credential`` moved the agent onto
the new pool entry but left that copy on the replaced key, so after a rotation compaction
kept sending a key the pool had refreshed or benched.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_pool import PooledCredential
from run_agent import AIAgent


@pytest.fixture
def summary_endpoint(monkeypatch):
    """Local chat-completions server recording the Authorization header of every completion request."""
    from agent.auxiliary_client import _reset_aux_unhealthy_cache, shutdown_cached_clients

    shutdown_cached_clients()
    _reset_aux_unhealthy_cache()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/chat/completions":  # local-server metadata probes
                self.send_error(404)
                return
            seen.append(self.headers.get("Authorization", ""))
            message = {"role": "assistant", "content": "## Active Task\nsummary"}
            response = {
                "id": "chatcmpl-summary", "object": "chat.completion", "created": 1, "model": payload["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            if payload.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                body, content_type = f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n".encode(), "text/event-stream"
            else:
                body, content_type = json.dumps(response).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        shutdown_cached_clients()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _agent(**kwargs) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        return AIAgent(quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                       enabled_toolsets=[], **kwargs)


def test_summary_request_after_pool_rotation_carries_the_new_key(summary_endpoint):
    base_url, seen = summary_endpoint
    agent = _agent(provider="custom", base_url=base_url, api_key="pool-key-old", model="local-model",
                   api_mode="chat_completions")
    try:
        assert agent._swap_credential(PooledCredential.from_dict(
            "custom", {"id": "second", "access_token": "pool-key-new", "base_url": base_url}))
        assert agent.api_key == "pool-key-new"

        summary = agent.context_compressor._call_summary_llm("Summarize this conversation.", time.monotonic())

        assert summary.startswith("## Active Task")
        assert seen == ["Bearer pool-key-new"]
    finally:
        agent.client.close()


_ANTHROPIC = "https://api.anthropic.com"
_CORP = "https://llm.corp.example/anthropic"


@pytest.mark.parametrize("cc_base_url,cc_key,follows", [
    (_ANTHROPIC, "sk-ant-oat01-old-entry", True),  # the compressor mirrors the main route
    (_CORP, "sk-ant-corp-key", False),  # its own route
    (_CORP, "sk-ant-oat01-old-entry", False),  # same key, another endpoint
    (_ANTHROPIC, "sk-ant-corp-key", False),  # same endpoint, its own key
])
def test_anthropic_pool_rotation_moves_only_a_mirroring_compressor(cc_base_url, cc_key, follows):
    old, new = "sk-ant-oat01-old-entry", "sk-ant-oat01-new-entry"
    with patch("agent.process_bootstrap.OpenAI"):
        agent = _agent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1")
    agent.api_mode, agent.provider, agent.model = "anthropic_messages", "anthropic", "claude-opus-4-6"
    agent.base_url = agent._anthropic_base_url = _ANTHROPIC
    agent.api_key = agent._anthropic_api_key = old
    agent._anthropic_client = MagicMock()
    cc = agent.context_compressor
    cc.update_model(model=agent.model, context_length=200_000, provider="anthropic", api_mode="anthropic_messages",
                    base_url=cc_base_url, api_key=cc_key)

    entry = PooledCredential.from_dict("anthropic", {"id": "claude_code", "access_token": new, "base_url": _ANTHROPIC})
    with patch.object(AIAgent, "_build_direct_anthropic_client", return_value=MagicMock()):
        assert agent._swap_credential(entry)
    assert agent.api_key == agent._anthropic_api_key == new

    calls = []
    response = MagicMock()
    response.choices[0].message.content = "## Active Task\nsummary"
    with patch("agent.context_compressor.call_llm", side_effect=lambda **kw: calls.append(kw) or response):
        cc._call_summary_llm("Summarize this conversation.", time.monotonic())

    assert calls[0]["main_runtime"]["api_key"] == (new if follows else cc_key)
