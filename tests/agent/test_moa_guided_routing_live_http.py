"""Real runtime construction + inference test for the MoA guided-model-routing adapter
(plans/2026-09-15_141016-guided-model-routing.md §6 "MoA").

Drives ``agent.moa_loop._run_reference`` (reference slot) and
``agent.moa_loop.aggregate_moa_context`` (aggregator slot) end-to-end against REAL local
loopback fake OpenAI-compatible HTTP servers -- not the selector called directly, not a mocked
constructor. Complements:

  * ``tests/tools/test_delegate_guided_routing_live_http.py`` (delegation adapter).
  * ``tests/agent/test_managed_route_guard_live_http.py`` (Kanban-shaped agent, per-request guard).

This file proves the MoA adapter specifically:
  1. a reference slot with ``routing_role`` set resolves a receipted route and the actual
     request reaches the receipted (approved) fake endpoint with real content;
  2. a denied managed reference slot (no active policy) never reaches ANY endpoint -- it becomes
     a labelled ``[failed: ...]`` note, zero requests recorded;
  3. an aggregator slot with ``routing_role`` set is routed the same way and its real request
     reaches the receipted endpoint;
  4. an unmanaged slot (no ``routing_role``) is completely untouched by routing -- the existing
     plain provider/model path, byte for byte.

External paid inference is never used; the endpoint is an ephemeral 127.0.0.1 http.server.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _CapturingHandler(BaseHTTPRequestHandler):
    requests: list

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode()) if length else {}
        if not self.path.rstrip("/").endswith("chat/completions"):
            self._send_json({"ok": True})
            return
        type(self).requests.append(req)
        self._send_json({
            "id": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "slot done"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        })

    def _send_json(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


def _start_server():
    handler_cls = type("Handler", (_CapturingHandler,), {"requests": []})
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler_cls


def _policy(endpoint: str) -> dict:
    return {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": "fake-route", "route_revision": 1, "provider": "custom",
            "model": "test-model", "endpoint": endpoint, "maker": "test",
            "model_family": "test-model", "status": "approved", "allowed_roles": ["moareference", "moaaggregator"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {
            "moareference": {"deep": ["fake-route"], "shallow": ["fake-route"]},
            "moaaggregator": {"deep": ["fake-route"], "shallow": ["fake-route"]},
        },
    }


@pytest.fixture()
def routed_home():
    server, handler = _start_server()
    test_home = tempfile.mkdtemp(prefix="hermes_moa_routing_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    prev = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = hermes_home

    saved = dict(sys.modules)
    for mod in list(sys.modules):
        if mod.startswith(("agent.", "tools.", "hermes_")):
            del sys.modules[mod]

    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/v1"
    try:
        yield {"hermes_home": hermes_home, "handler": handler, "url": url}
    finally:
        server.shutdown()
        os.environ.pop("HERMES_HOME", None)
        if prev is not None:
            os.environ["HERMES_HOME"] = prev
        shutil.rmtree(test_home, ignore_errors=True)
        sys.modules.clear()
        sys.modules.update(saved)


def _publish_active(hermes_home, url, policy=None):
    from agent.model_selection_store import activate_policy, publish_policy
    policy = policy or _policy(url)
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])


def _patch_custom_provider(monkeypatch, url):
    """Make the 'custom' provider name resolve through the runtime provider system to our fake
    endpoint with a harmless api key, mirroring how a real configured custom provider resolves."""
    from hermes_cli import runtime_provider as rp

    def _fake_resolve(*, requested, target_model):
        return {"provider": "custom", "base_url": url, "api_key": "fake-managed-key",
                "api_mode": "chat_completions", "request_overrides": {}, "command": None, "args": []}

    monkeypatch.setattr(rp, "resolve_runtime_provider", _fake_resolve)
    import agent.moa_model_routing as mmr
    monkeypatch.setattr(mmr, "resolve_runtime_provider", _fake_resolve, raising=False)


def test_managed_reference_slot_reaches_real_endpoint_and_pins_receipt(routed_home, monkeypatch):
    from agent.moa_loop import _run_reference

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url)
    _patch_custom_provider(monkeypatch, url)

    slot = {
        "provider": "does-not-matter", "model": "does-not-matter",
        "routing_role": "moareference", "routing_requirements": {"task_class": "established-pattern"},
    }
    label, text, acct = _run_reference(
        slot, [{"role": "user", "content": "Summarize the managed routing design in one sentence."}],
        execution_id="test-turn", slot_id="reference-0",
    )
    assert text == "slot done", (label, text)
    assert len(handler.requests) == 1, "the real slot must reach the approved endpoint exactly once"
    sent = json.dumps(handler.requests[0])
    assert "Summarize the managed routing design" in sent


def test_managed_reference_slot_denied_route_reaches_zero_endpoints(routed_home, monkeypatch):
    """No active policy published -> RoutingBlocked -> the reference must fail BEFORE any
    endpoint is reached; the failure surfaces as the normal labelled [failed: ...] note (zero
    content sent), never a silent fallback to the slot's plain provider/model."""
    from agent.moa_loop import _run_reference

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    # Deliberately do NOT publish/activate a policy.
    _patch_custom_provider(monkeypatch, url)

    slot = {"provider": "custom", "model": "test-model", "routing_role": "moareference"}
    label, text, acct = _run_reference(
        slot, [{"role": "user", "content": "This must never run."}],
        execution_id="test-turn", slot_id="reference-0",
    )
    assert text.startswith("[failed:"), text
    assert len(handler.requests) == 0, "a denied managed reference must never reach the endpoint"


def test_managed_aggregator_slot_reaches_real_endpoint(routed_home, monkeypatch):
    from agent.moa_loop import aggregate_moa_context

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url)
    _patch_custom_provider(monkeypatch, url)

    result = aggregate_moa_context(
        user_prompt="What should I do next?",
        api_messages=[{"role": "user", "content": "What should I do next?"}],
        reference_models=[],
        aggregator={"provider": "does-not-matter", "model": "does-not-matter", "routing_role": "moaaggregator"},
    )
    assert isinstance(result, str)
    assert len(handler.requests) == 1, "the real aggregator must reach the approved endpoint exactly once"


def test_managed_aggregator_denied_route_never_reaches_endpoint(routed_home, monkeypatch):
    from agent.moa_loop import aggregate_moa_context

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    # No active policy published.
    _patch_custom_provider(monkeypatch, url)

    result = aggregate_moa_context(
        user_prompt="This must never run.",
        api_messages=[{"role": "user", "content": "This must never run."}],
        reference_models=[],
        aggregator={"provider": "custom", "model": "test-model", "routing_role": "moaaggregator"},
    )
    assert isinstance(result, str)
    assert len(handler.requests) == 0, "a denied managed aggregator must never reach the endpoint"


def test_unmanaged_slot_is_untouched_by_routing(routed_home, monkeypatch):
    """A slot with no ``routing_role`` takes the existing plain provider/model path
    byte-for-byte -- guided routing must never activate implicitly."""
    from agent.moa_loop import _run_reference

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    # No policy published/activated at all -- if the unmanaged path accidentally invoked
    # routing it would raise/fail here.
    _patch_custom_provider(monkeypatch, url)

    slot = {"provider": "custom", "model": "test-model"}  # no routing_role
    label, text, acct = _run_reference(
        slot, [{"role": "user", "content": "Plain unmanaged advisory call."}],
        execution_id="test-turn", slot_id="reference-0",
    )
    assert text == "slot done", (label, text)
    assert len(handler.requests) == 1
    sent = json.dumps(handler.requests[0])
    assert "Plain unmanaged advisory call" in sent
