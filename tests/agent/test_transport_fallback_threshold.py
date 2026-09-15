"""Tests for agent.transport_fallback_threshold (config-driven eager fallback).

The threshold gates eager fallback on transport-layer failures (timeout /
overloaded).  Before this change the threshold was hardcoded to 2 in
``run_conversation``; it is now read from ``agent.transport_fallback_threshold``
in config.yaml (default 2).

Semantics: 1 = fall back on the first transport failure; N = require N
consecutive transport failures; 0 = disabled (transport-failure fallback never
fires — rate-limit/billing fallback still applies); negative values are
invalid and fall back to the default.

The gate itself is exercised as a *behaviour* test: a mock provider returns
HTTP 503 ("server busy" → classified ``overloaded``) for the first N requests,
then succeeds, and we observe whether the fallback chain actually activates
under each configured threshold.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _build_agent(monkeypatch, config_value):
    """Spin up an AIAgent with a patched agent-section config value.

    ``init_agent`` reads ``_load_agent_config()`` for the ``agent`` section;
    we patch just that section so every other default (api_max_retries etc.)
    stays at the shipped default.
    """
    from run_agent import AIAgent
    import hermes_cli.config as hc

    real_loader = hc.load_config_readonly

    def _patched():
        cfg = real_loader()
        agent_sec = dict(cfg.get("agent", {}))
        if config_value is not None:
            agent_sec["transport_fallback_threshold"] = config_value
        cfg["agent"] = agent_sec
        return cfg

    monkeypatch.setattr(hc, "load_config_readonly", _patched)
    return AIAgent(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_config_defaults_registers_transport_fallback_threshold():
    """DEFAULT_CONFIG registers the key with an int >= 0 (behaviour contract,
    not a snapshot of the current literal value; 0 is the valid "disabled"
    sentinel).
    """
    agent_cfg = DEFAULT_CONFIG.get("agent", {})
    assert "transport_fallback_threshold" in agent_cfg
    assert isinstance(agent_cfg["transport_fallback_threshold"], int)
    assert agent_cfg["transport_fallback_threshold"] >= 0


def test_init_agent_applies_configured_threshold(monkeypatch):
    agent = _build_agent(monkeypatch, 5)
    assert agent._transport_fallback_threshold == 5


def test_init_agent_zero_disables_transport_fallback(monkeypatch):
    """0 means 'never fall back on transport failure' — kept as 0, not clamped."""
    agent = _build_agent(monkeypatch, 0)
    assert agent._transport_fallback_threshold == 0


def test_init_agent_negative_falls_back_to_default(monkeypatch):
    """Negative values are invalid — degrade to the default, with a warning."""
    agent = _build_agent(monkeypatch, -3)
    assert agent._transport_fallback_threshold == 2


def test_init_agent_defaults_to_2_when_unset(monkeypatch):
    """The shipped default preserves the historical hardcoded behaviour."""
    agent = _build_agent(monkeypatch, None)
    assert agent._transport_fallback_threshold == 2


def test_init_agent_tolerates_non_int(monkeypatch):
    """A malformed config value falls back to the default, no crash."""
    agent = _build_agent(monkeypatch, "not-a-number")
    assert agent._transport_fallback_threshold == 2


# ── Behaviour tests: the gate under real transport failures ──────────────


class _MockHandler(BaseHTTPRequestHandler):
    # Set by the fixture before each request cycle.
    captured_requests: list = []
    status_queue: list = []  # ints: HTTP status per request; defaults to 200

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        status = type(self).status_queue.pop(0) if type(self).status_queue else 200
        if status != 200:
            body = json.dumps({"error": {"message": "server busy"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        is_stream = req.get("stream") is True
        resp = _text_resp("DONE")
        msg = resp["choices"][0]["message"]
        if is_stream:
            content = msg.get("content") or ""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            ]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):  # silence the default stderr logging
        pass


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture()
def transport_agent():
    """Mock provider (HTTP 503s then 200s) + an AIAgent with a fallback chain.

    Both the primary and the fallback entry point at the same mock server, so
    the status queue drives the whole retry/fallback cycle deterministically.
    """
    _MockHandler.captured_requests = []
    _MockHandler.status_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_tft_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    # Import fresh so the patched conversation_loop is exercised even when the
    # module was imported earlier in the same worker.
    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat", model="test-model",
        fallback_model=[{
            "provider": "openai-compat",
            "model": "fallback-model",
            "base_url": f"http://127.0.0.1:{port}/v1",
        }],
        max_iterations=10, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
    )
    agent.valid_tool_names = set()

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def test_threshold_1_falls_back_on_first_transport_failure(transport_agent):
    """1 = most aggressive: the first transport failure flips the gate."""
    agent, handler = transport_agent
    agent._transport_fallback_threshold = 1
    handler.status_queue.extend([503, 200, 200])

    result = agent.run_conversation("hello", conversation_history=[], task_id="t")

    assert agent._fallback_index == 1  # fallback chain activated
    assert result.get("final_response") == "DONE"


def test_threshold_2_requires_two_transport_failures(transport_agent):
    """2 = shipped default: one transient 503 is retried, the second flips."""
    agent, handler = transport_agent
    agent._transport_fallback_threshold = 2
    handler.status_queue.extend([503, 503, 200, 200])

    result = agent.run_conversation("hello", conversation_history=[], task_id="t")

    assert agent._fallback_index == 1
    assert result.get("final_response") == "DONE"


def test_threshold_0_disables_transport_fallback(transport_agent):
    """0 = disabled: repeated transport failures never flip the gate."""
    agent, handler = transport_agent
    agent._transport_fallback_threshold = 0
    handler.status_queue.extend([503] * 10)

    result = agent.run_conversation("hello", conversation_history=[], task_id="t")

    assert agent._fallback_index == 0  # never fell back despite 503s
    assert result.get("completed") is not True  # retries exhausted, turn failed


def test_large_threshold_tolerates_transport_hiccups(transport_agent):
    """A large threshold is extremely tolerant: hiccups don't flip the gate."""
    agent, handler = transport_agent
    agent._transport_fallback_threshold = 100
    handler.status_queue.extend([503, 503, 200, 200])

    result = agent.run_conversation("hello", conversation_history=[], task_id="t")

    assert agent._fallback_index == 0
    assert result.get("final_response") == "DONE"
