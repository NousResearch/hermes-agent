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
                "api_mode": "chat_completions", "request_overrides": {"extra_body": {"reasoning": {"effort": "medium"}}},
                "command": None, "args": []}

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
        "routing_role": "moareference", "routing_requirements": {
            "task_class": "established-pattern", "input_tokens": 1000, "reserve_tokens": 8192},
    }
    label, text, acct = _run_reference(
        slot, [{"role": "user", "content": "Summarize the managed routing design in one sentence."}],
        execution_id="test-turn", slot_id="reference-0",
    )
    assert text == "slot done", (label, text)
    assert len(handler.requests) == 1, "the real slot must reach the approved endpoint exactly once"
    sent = json.dumps(handler.requests[0])
    assert "Summarize the managed routing design" in sent


@pytest.mark.parametrize("mutation", ["model", "reasoning", "messages"])
def test_managed_reference_checks_final_auxiliary_send(routed_home, monkeypatch, mutation):
    import importlib
    from agent.moa_loop import _run_reference
    from agent.moa_model_routing import MoARequiredSlotDenied
    auxiliary_client = importlib.import_module("agent.auxiliary_client")


    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    original = auxiliary_client._create_with_progress
    reached = []

    def altered(client, kwargs, *args, **options):
        reached.append(True)
        kwargs = dict(kwargs)
        if mutation == "model":
            kwargs["model"] = "unapproved-model"
        elif mutation == "reasoning":
            kwargs["extra_body"] = {"reasoning": {"effort": "low"}}
            kwargs["reasoning_effort"] = "low"
        else:
            kwargs["messages"] = [{"role": "user", "content": "x" * 300000}]
        return original(client, kwargs, *args, **options)

    monkeypatch.setattr(auxiliary_client, "_create_with_progress", altered)
    with pytest.raises(MoARequiredSlotDenied):
        _run_reference({"provider": "custom", "model": "test-model", "routing_role": "moareference",
                        "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}},
                       [{"role": "user", "content": "bounded reference"}],
                       execution_id="final-aux", slot_id="reference-0")
    assert reached, "the production auxiliary sender must be exercised"
    assert not handler.requests


@pytest.mark.parametrize("slot_kind", ["reference", "aggregator"])
def test_moa_assembled_budget_cannot_be_understated(routed_home, monkeypatch, slot_kind):
    from agent.moa_loop import _run_reference, aggregate_moa_context
    from agent.model_selection_types import RoutingBlocked

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    slot = {"provider": "custom", "model": "test-model", "routing_role": "moa" + slot_kind,
            "routing_requirements": {"input_tokens": 1, "reserve_tokens": 1}}
    messages = [{"role": "user", "content": "oversize " * 30000}]
    with pytest.raises(RoutingBlocked, match="input_too_large"):
        if slot_kind == "reference":
            _run_reference(slot, messages, execution_id="oversize", slot_id="reference-0")
        else:
            aggregate_moa_context(user_prompt=messages[0]["content"], api_messages=messages,
                                  reference_models=[], aggregator=slot)
    assert not handler.requests


def test_managed_reference_slot_denied_route_hard_fails_by_default(routed_home, monkeypatch):
    """BINDING default (plan §6 "MoA"): a managed reference slot (``routing_role`` set) with no
    explicit ``routing_requirements.required`` opt-out is REQUIRED. No active policy published ->
    RoutingBlocked -> the reference must fail BEFORE any endpoint is reached, and that failure
    MUST propagate as ``MoARequiredSlotDenied`` -- never degrade to a labelled ``[failed: ...]``
    note that lets the aggregator silently synthesize over it as if the gate had passed."""
    from agent.moa_loop import _run_reference
    from agent.moa_model_routing import MoARequiredSlotDenied

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    # Deliberately do NOT publish/activate a policy.
    _patch_custom_provider(monkeypatch, url)

    slot = {"provider": "custom", "model": "test-model", "routing_role": "moareference"}
    with pytest.raises(MoARequiredSlotDenied):
        _run_reference(
            slot, [{"role": "user", "content": "This must never run."}],
            execution_id="test-turn", slot_id="reference-0",
        )
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
        aggregator={"provider": "does-not-matter", "model": "does-not-matter", "routing_role": "moaaggregator",
                    "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}},
    )
    assert isinstance(result, str)
    assert len(handler.requests) == 1, "the real aggregator must reach the approved endpoint exactly once"


def test_managed_aggregator_denied_route_hard_fails_by_default(routed_home, monkeypatch):
    """BINDING default: an aggregator slot with ``routing_role`` set and no explicit
    ``required: false`` opt-out is REQUIRED -- a denied route must hard-fail the whole MoA
    attempt (``MoARequiredSlotDenied``), never return a friendly "proceeding without aggregated
    guidance" string that looks like a successful completion."""
    from agent.moa_loop import aggregate_moa_context
    from agent.moa_model_routing import MoARequiredSlotDenied

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    # No active policy published.
    _patch_custom_provider(monkeypatch, url)

    with pytest.raises(MoARequiredSlotDenied):
        aggregate_moa_context(
            user_prompt="This must never run.",
            api_messages=[{"role": "user", "content": "This must never run."}],
            reference_models=[],
            aggregator={"provider": "custom", "model": "test-model", "routing_role": "moaaggregator"},
        )
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


def test_shadow_reference_records_recommendation_but_uses_legacy_slot(routed_home, monkeypatch):
    """Shadow mode must not replace the fixed preset's provider/model even
    though it resolves and persists a recommendation under the active policy."""
    from agent.moa_loop import _run_reference
    from agent.model_selection_store import _connect

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url)
    _patch_custom_provider(monkeypatch, url)
    slot = {
        "provider": "custom", "model": "legacy-reference-model",
        "routing_role": "moareference", "routing_mode": "shadow",
        "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192},
    }

    _run_reference(
        slot, [{"role": "user", "content": "Observe this advisory request."}],
        execution_id="shadow-reference", slot_id="reference-0",
    )

    assert handler.requests[-1]["model"] == "legacy-reference-model"
    conn = _connect(hermes_home)
    try:
        shadow_events = conn.execute(
            "SELECT COUNT(*) AS n FROM routing_outcomes WHERE kind='routing_shadow'"
        ).fetchone()["n"]
    finally:
        conn.close()
    assert shadow_events == 1


def test_required_reference_slot_denial_hard_fails_not_degraded_note(routed_home, monkeypatch):
    """Explicit ``routing_requirements: {"required": true}`` is equivalent to the default (any
    managed slot is required unless it opts OUT): a denied route must raise
    ``MoARequiredSlotDenied`` (never degrade to a labelled ``[failed: ...]`` note that the
    aggregator would silently synthesize over as if the gate passed), and zero endpoints are
    reached."""
    from agent.moa_loop import _run_reference
    from agent.moa_model_routing import MoARequiredSlotDenied

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    # Deliberately do NOT publish/activate a policy -> denial.
    _patch_custom_provider(monkeypatch, url)

    slot = {
        "provider": "custom", "model": "test-model", "routing_role": "moareference",
        "routing_requirements": {"required": True},
    }
    with pytest.raises(MoARequiredSlotDenied):
        _run_reference(
            slot, [{"role": "user", "content": "This must never run."}],
            execution_id="test-turn", slot_id="reference-0",
        )
    assert len(handler.requests) == 0, "a denied REQUIRED reference must never reach any endpoint"


def test_required_aggregator_denial_hard_fails_no_synthesis(routed_home, monkeypatch):
    """A required aggregator's denial must propagate as a hard failure of the whole MoA
    attempt -- never a successful empty/degraded synthesis string."""
    from agent.moa_loop import aggregate_moa_context
    from agent.moa_model_routing import MoARequiredSlotDenied

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    # No active policy published.
    _patch_custom_provider(monkeypatch, url)

    with pytest.raises(MoARequiredSlotDenied):
        aggregate_moa_context(
            user_prompt="This must never run.",
            api_messages=[{"role": "user", "content": "This must never run."}],
            reference_models=[],
            aggregator={
                "provider": "custom", "model": "test-model", "routing_role": "moaaggregator",
                "routing_requirements": {"required": True},
            },
        )
    assert len(handler.requests) == 0, "a denied REQUIRED aggregator must never reach any endpoint"


@pytest.mark.parametrize("denied_slot", ["reference", "aggregator"])
def test_managed_cohort_rejects_optional_escape_before_legacy_contact(routed_home, monkeypatch, denied_slot):
    from agent.moa_loop import aggregate_moa_context
    from agent.model_selection_types import RoutingBlocked

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _patch_custom_provider(monkeypatch, url)

    slot = {
        "provider": "custom", "model": "test-model", "routing_role": "moareference",
        "routing_requirements": {"required": False},
    }
    with pytest.raises((RoutingBlocked, ValueError)):
        aggregate_moa_context(
            user_prompt="This must never run.",
            api_messages=[{"role": "user", "content": "This must never run."}],
            reference_models=[slot] if denied_slot == "reference" else [],
            aggregator=slot if denied_slot == "aggregator" else {"provider": "custom", "model": "legacy"},
        )
    assert len(handler.requests) == 0


@pytest.mark.parametrize("fault", ["provenance", "store", "outcome"])
def test_shadow_cohort_observation_failure_does_not_cancel_legacy(routed_home, monkeypatch, fault):
    from pathlib import Path
    import sqlite3
    from agent.moa_loop import aggregate_moa_context
    import agent.model_selection_store as store

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    if fault == "store":
        database = Path(home) / "model_routing.db"
        database.rename(database.with_suffix(".saved"))
        database.mkdir()
    elif fault == "outcome":
        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("private storage failure")
        monkeypatch.setattr(store, "append_outcome", fail)
    slot = {"provider": "custom", "model": "legacy-reference", "routing_mode": "shadow",
            "routing_role": "reviewquality" if fault == "provenance" else "moareference",
            "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}}
    aggregate_moa_context(user_prompt="legacy", api_messages=[{"role": "user", "content": "legacy"}],
                          reference_models=[slot], aggregator={"provider": "custom", "model": "legacy-aggregator"})
    assert [request["model"] for request in handler.requests] == ["legacy-reference", "legacy-aggregator"]
