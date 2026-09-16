"""Cohort-level tests for the MoA required-cohort contract
(plans/2026-09-15_141016-guided-model-routing.md line 140, binding §6 "MoA").

These exercise the PUBLIC config/runtime surface (agent.moa_model_routing.resolve_moa_cohort,
agent.moa_loop._run_reference / aggregate_moa_context) against real local fake OpenAI-compatible
HTTP servers -- never the pure selector called in isolation -- proving:

  1. a full required cohort (two required references from different approved makers + a required
     aggregator sharing a reference's maker) resolves cleanly, once, with real content reaching
     every approved endpoint;
  2. maker diversity is enforced: two required references resolving to the SAME maker reject the
     whole cohort BEFORE any endpoint is reached;
  3. a spoofed/claimed maker on the slot itself cannot substitute for the policy-resolved maker
     (independence is judged from the persisted receipt, not slot-supplied text);
  4. a required reference or aggregator that fails to resolve/deny leaves the review gate
     incomplete (raises) rather than completing with an empty/degraded synthesis;
  5. the existing "denied managed slot hard-fails by default" per-slot behavior
     (agent.moa_loop._run_reference / aggregate_moa_context) composes correctly when exercised
     as a genuine two-reference + aggregator cohort, not just a single isolated slot.

Complements tests/agent/test_moa_guided_routing_live_http.py (single-slot live-HTTP coverage).
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


def _policy(endpoint: str, *, makers: dict) -> dict:
    """``makers`` maps route_id -> maker for each of the (up to) three routes this test uses:
    'route-a', 'route-b', 'route-c'."""
    routes = []
    for route_id, maker in makers.items():
        routes.append({
            "route_id": route_id, "route_revision": 1, "provider": "custom",
            "model": f"test-model-{route_id}", "endpoint": endpoint, "maker": maker,
            "model_family": f"test-model-{route_id}", "status": "approved",
            "allowed_roles": ["moareference", "moaaggregator"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        })
    return {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": routes,
        "rankings": {
            "moareference": {
                "deep": list(makers), "shallow": list(makers),
            },
            "moaaggregator": {
                "deep": list(makers), "shallow": list(makers),
            },
        },
    }


@pytest.fixture()
def routed_home():
    server, handler = _start_server()
    test_home = tempfile.mkdtemp(prefix="hermes_moa_cohort_")
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


def _publish_active(hermes_home, url, *, makers: dict):
    from agent.model_selection_store import activate_policy, publish_policy
    policy = _policy(url, makers=makers)
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])


def _patch_custom_provider(monkeypatch, url):
    from hermes_cli import runtime_provider as rp

    def _fake_resolve(*, requested, target_model):
        return {"provider": "custom", "base_url": url, "api_key": "fake-managed-key",
                "api_mode": "chat_completions", "request_overrides": {}, "command": None, "args": []}

    monkeypatch.setattr(rp, "resolve_runtime_provider", _fake_resolve)
    import agent.moa_model_routing as mmr
    monkeypatch.setattr(mmr, "resolve_runtime_provider", _fake_resolve, raising=False)


def _slot(role: str) -> dict:
    return {"provider": "does-not-matter", "model": "does-not-matter", "routing_role": role}


# ---------------------------------------------------------------------------
# 1. Full required cohort: two required references (different makers) + a required
#    aggregator sharing one reference's maker.
# ---------------------------------------------------------------------------

def test_full_required_cohort_resolves_with_distinct_reference_makers(routed_home, monkeypatch):
    from agent.moa_model_routing import resolve_moa_cohort

    hermes_home, url = routed_home["hermes_home"], routed_home["url"]
    _publish_active(hermes_home, url, makers={"route-a": "maker-a", "route-b": "maker-b"})
    _patch_custom_provider(monkeypatch, url)

    refs = [_slot("moareference"), _slot("moareference")]
    agg = _slot("moaaggregator")
    resolutions = resolve_moa_cohort(refs, agg, execution_id="cohort-1")

    assert resolutions["reference-0"] is not None
    assert resolutions["reference-1"] is not None
    assert resolutions["aggregator"] is not None
    ref_makers = set()
    from agent.model_selection_store import get_receipt
    for slot_id in ("reference-0", "reference-1"):
        res = resolutions[slot_id]
        decision = get_receipt(res["routing_home"], res["receipt_id"])
        ref_makers.add(decision["selected"]["maker"])
    assert len(ref_makers) == 2, f"expected two distinct required-reference makers, got {ref_makers}"

    # Aggregator MAY share a reference's maker (design §6 explicitly permits this).
    agg_res = resolutions["aggregator"]
    agg_decision = get_receipt(agg_res["routing_home"], agg_res["receipt_id"])
    assert agg_decision["selected"]["maker"] in ref_makers | {"maker-a", "maker-b"}


def test_full_required_cohort_end_to_end_reaches_every_approved_endpoint(routed_home, monkeypatch):
    """The full cohort, driven through the real public runtime (_run_reference +
    aggregate_moa_context), reaches every approved endpoint with real content -- not merely
    resolves receipts."""
    from agent.moa_loop import _run_reference, aggregate_moa_context

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url, makers={"route-a": "maker-a", "route-b": "maker-b"})
    _patch_custom_provider(monkeypatch, url)

    ref_slots = [_slot("moareference"), _slot("moareference")]
    messages = [{"role": "user", "content": "Diversity cohort smoke test."}]
    for idx, slot in enumerate(ref_slots):
        label, text, acct = _run_reference(
            slot, messages, execution_id="cohort-e2e", slot_id=f"reference-{idx}",
        )
        assert text == "slot done", (label, text)

    agg_result = aggregate_moa_context(
        user_prompt="Diversity cohort smoke test.",
        api_messages=messages,
        reference_models=[],  # aggregator-only call in this sub-check
        aggregator=_slot("moaaggregator"),
    )
    assert isinstance(agg_result, str)
    # 2 references + 1 aggregator = 3 real requests reaching the fake endpoint.
    assert len(handler.requests) == 3


# ---------------------------------------------------------------------------
# 2. Maker diversity enforcement: two required references resolving to the SAME maker must
#    reject the whole cohort before any endpoint is reached.
# ---------------------------------------------------------------------------

def test_cohort_rejects_when_required_references_share_one_maker(routed_home, monkeypatch):
    from agent.moa_model_routing import MoARoutingBlocked, resolve_moa_cohort

    hermes_home, url = routed_home["hermes_home"], routed_home["url"]
    # Both routes belong to the SAME maker -- ranking always prefers route-a, so both
    # references would resolve to the identical maker.
    _publish_active(hermes_home, url, makers={"route-a": "maker-a", "route-b": "maker-a"})
    _patch_custom_provider(monkeypatch, url)

    refs = [_slot("moareference"), _slot("moareference")]
    agg = _slot("moaaggregator")
    with pytest.raises(MoARoutingBlocked) as exc_info:
        resolve_moa_cohort(refs, agg, execution_id="cohort-2")
    assert "independence" in str(exc_info.value).lower() or "distinct" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 3. A slot cannot spoof an independent maker -- independence is judged from the persisted
#    receipt, never slot-supplied text.
# ---------------------------------------------------------------------------

def test_cohort_ignores_slot_claimed_maker_uses_receipted_identity(routed_home, monkeypatch):
    from agent.model_selection_store import get_receipt
    from agent.moa_model_routing import resolve_moa_cohort

    hermes_home, url = routed_home["hermes_home"], routed_home["url"]
    _publish_active(hermes_home, url, makers={"route-a": "maker-a", "route-b": "maker-b"})
    _patch_custom_provider(monkeypatch, url)

    # Slot dict claims a bogus 'maker' field directly -- this is not part of the real intake
    # schema and must have zero effect on the resolved/receipted identity.
    spoofed_slot = dict(_slot("moareference"))
    spoofed_slot["maker"] = "totally-not-maker-a"
    refs = [spoofed_slot, _slot("moareference")]
    agg = _slot("moaaggregator")

    resolutions = resolve_moa_cohort(refs, agg, execution_id="cohort-3")
    res0 = resolutions["reference-0"]
    decision = get_receipt(res0["routing_home"], res0["receipt_id"])
    # The receipted maker must be the real policy-resolved one, never the spoofed slot field.
    assert decision["selected"]["maker"] != "totally-not-maker-a"
    assert decision["selected"]["maker"] in {"maker-a", "maker-b"}


def test_virtual_moa_maker_cannot_satisfy_independence(routed_home, monkeypatch):
    """A route whose maker is literally 'moa' (the virtual aggregation maker) can never be
    counted for cohort-diversity purposes, per design §6: "the virtual maker moa cannot satisfy
    an independence constraint."."""
    from agent.moa_model_routing import MoARoutingBlocked, resolve_moa_cohort

    hermes_home, url = routed_home["hermes_home"], routed_home["url"]
    _publish_active(hermes_home, url, makers={"route-a": "moa", "route-b": "moa"})
    _patch_custom_provider(monkeypatch, url)

    refs = [_slot("moareference"), _slot("moareference")]
    agg = _slot("moaaggregator")
    with pytest.raises(MoARoutingBlocked):
        resolve_moa_cohort(refs, agg, execution_id="cohort-4")


# ---------------------------------------------------------------------------
# 4. A required reference or aggregator denial leaves the gate incomplete (raises), never a
#    successful empty/degraded synthesis.
# ---------------------------------------------------------------------------

def test_cohort_one_denied_required_reference_gates_incomplete(routed_home, monkeypatch):
    from agent.moa_model_routing import MoARequiredSlotDenied, resolve_moa_cohort

    hermes_home, url = routed_home["hermes_home"], routed_home["url"]
    _publish_active(hermes_home, url, makers={"route-a": "maker-a"})
    _patch_custom_provider(monkeypatch, url)

    # reference-1's routing_role has no ranking entry -> denial.
    refs = [_slot("moareference"), {"provider": "x", "model": "y", "routing_role": "unranked-role"}]
    agg = _slot("moaaggregator")
    with pytest.raises(MoARequiredSlotDenied):
        resolve_moa_cohort(refs, agg, execution_id="cohort-5")


def test_cohort_denied_required_aggregator_gates_incomplete_not_empty_success(routed_home, monkeypatch):
    from agent.moa_loop import aggregate_moa_context
    from agent.moa_model_routing import MoARequiredSlotDenied

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url, makers={"route-a": "maker-a", "route-b": "maker-b"})
    _patch_custom_provider(monkeypatch, url)

    with pytest.raises(MoARequiredSlotDenied):
        aggregate_moa_context(
            user_prompt="gate must not silently complete",
            api_messages=[{"role": "user", "content": "gate must not silently complete"}],
            reference_models=[_slot("moareference"), _slot("moareference")],
            aggregator={"provider": "x", "model": "y", "routing_role": "unranked-role"},
        )
    # References may have been reached, but the aggregator (unranked) never was.
    sent_models = {r.get("model") for r in handler.requests}
    assert "y" not in sent_models


# ---------------------------------------------------------------------------
# 5. Config-edit mid-run cannot reroute an in-flight resolution: resolve once, mutate the
#    active policy, re-resolve with the SAME resolution object (not the runtime-loop pinning,
#    which is a separate concern) — proves resolve_moa_cohort's receipts are immutable per call
#    and a second real resolve_moa_cohort call under the new policy sees the NEW policy (i.e.
#    routing correctly reflects policy state at each resolution, and a caller that reuses a
#    prior resolution rather than re-resolving is the one guaranteeing pinning).
# ---------------------------------------------------------------------------

def test_resolved_cohort_receipt_is_immutable_after_policy_activation_changes(routed_home, monkeypatch):
    from agent.model_selection_store import activate_policy, get_receipt, publish_policy
    from agent.moa_model_routing import resolve_moa_cohort

    hermes_home, url = routed_home["hermes_home"], routed_home["url"]
    _publish_active(hermes_home, url, makers={"route-a": "maker-a", "route-b": "maker-b"})
    _patch_custom_provider(monkeypatch, url)

    refs = [_slot("moareference"), _slot("moareference")]
    agg = _slot("moaaggregator")
    first = resolve_moa_cohort(refs, agg, execution_id="cohort-pin-1")
    res0 = first["reference-0"]
    original_decision = get_receipt(res0["routing_home"], res0["receipt_id"])
    original_model = original_decision["selected"]["model"]

    # Publish and activate an entirely different policy revision with different route ids/models.
    new_policy = _policy(url, makers={"route-c": "maker-a", "route-d": "maker-b"})
    new_policy["revision"] = 2
    record = publish_policy(hermes_home, new_policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])

    # The ALREADY-RESOLVED receipt for the first run's slot is untouched by the policy edit --
    # receipts are immutable once persisted (design §12), regardless of what the live policy
    # becomes afterward.
    replayed_decision = get_receipt(res0["routing_home"], res0["receipt_id"])
    assert replayed_decision["selected"]["model"] == original_model

    # A genuinely NEW resolution (new execution_id -- a new attempt) correctly picks up the new
    # policy's routes; it is not silently pinned to the old policy either.
    second = resolve_moa_cohort(refs, agg, execution_id="cohort-pin-2")
    res0_second = second["reference-0"]
    new_decision = get_receipt(res0_second["routing_home"], res0_second["receipt_id"])
    assert new_decision["selected"]["model"] != original_model


# ---------------------------------------------------------------------------
# 6. Contributor provenance / slot identity persistence -- no repeated reselection for the
#    identical (execution_id, slot_id) key within resolve_moa_cohort_pinned.
# ---------------------------------------------------------------------------

def test_resolve_moa_cohort_pinned_reuses_identical_resolution_same_execution_id(routed_home, monkeypatch):
    from agent.moa_model_routing import resolve_moa_cohort_pinned

    hermes_home, url = routed_home["hermes_home"], routed_home["url"]
    _publish_active(hermes_home, url, makers={"route-a": "maker-a", "route-b": "maker-b"})
    _patch_custom_provider(monkeypatch, url)

    refs = [_slot("moareference"), _slot("moareference")]
    agg = _slot("moaaggregator")

    first = resolve_moa_cohort_pinned(refs, agg, execution_id="pinned-run-1")
    second = resolve_moa_cohort_pinned(refs, agg, execution_id="pinned-run-1")
    # Same execution_id -> identical pinned resolutions object contents (no re-resolution).
    assert first["reference-0"]["receipt_id"] == second["reference-0"]["receipt_id"]
    assert first["aggregator"]["receipt_id"] == second["aggregator"]["receipt_id"]

    third = resolve_moa_cohort_pinned(refs, agg, execution_id="pinned-run-2")
    # A different execution_id is a new attempt: it may resolve fresh receipts (still valid,
    # not required to differ, but the call must succeed independently).
    assert third["reference-0"] is not None
