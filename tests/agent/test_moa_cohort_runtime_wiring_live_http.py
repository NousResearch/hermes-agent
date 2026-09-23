"""Runtime-wiring test for MoA cohort selection/pinning (plans/2026-09-15_141016-guided-model-
routing.md §6 "MoA").

Unlike ``tests/agent/test_moa_guided_routing_live_http.py`` (which drives ``_run_reference`` /
``aggregate_moa_context`` directly with a hand-supplied constant ``execution_id``), this file
drives the REAL class-facade consumer entry point (``agent.moa_loop.MoAChatCompletions.create``,
as built by ``build_moa_facade`` and called through ``client.chat.completions.create(...)`` in
the actual conversation loop) with a genuine per-attempt identity supplied only via
``agent._current_turn_id`` -- never by calling ``resolve_moa_cohort``/``resolve_moa_cohort_pinned``
directly as the whole evidence.

Covers the required scenarios:
  1. two SEPARATE runs (distinct ``_current_turn_id``s) each resolve their own cohort and reach
     the real endpoints independently -- no cross-run cache bleed.
  2. multiple iterations of the SAME run (same ``_current_turn_id``, e.g. a repeated ``create()``
     call as the acting loop would make on cache-MISS re-entry) reuse the identical pinned cohort
     receipt -- no reselection mid-run.
  3. a config/policy edit performed WHILE a run's cohort is pinned does not reroute that
     in-flight run; only a NEW run (new turn id) observes the edit.
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
        if req.get("stream"):
            chunk = {"id": "stream-fixture", "choices": [{"index": 0,
                     "delta": {"content": "slot done"}, "finish_reason": "stop"}]}
            body = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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


def _policy(endpoint: str, route_id: str = "fake-route", revision: int = 1) -> dict:
    return {
        "schema_version": 1, "policy_id": "kanban-default", "revision": revision,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": route_id, "route_revision": 1, "provider": "custom",
            "model": "test-model", "endpoint": endpoint, "maker": "test-maker-a",
            "model_family": "test-model", "status": "approved", "allowed_roles": ["moareference", "moaaggregator"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {
            "moareference": {"deep": [route_id], "shallow": [route_id]},
            "moaaggregator": {"deep": [route_id], "shallow": [route_id]},
        },
    }


@pytest.fixture()
def routed_home():
    server, handler = _start_server()
    test_home = tempfile.mkdtemp(prefix="hermes_moa_cohort_runtime_")
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


def _publish_active(hermes_home, url, route_id="fake-route", revision=1):
    from agent.model_selection_store import activate_policy, publish_policy
    policy = _policy(url, route_id=route_id, revision=revision)
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


class _FakeAgent:
    """Minimal stand-in for the real AIAgent: only the attributes the MoA facade/fan-out
    actually reads. ``_current_turn_id`` is the SAME attribute the real turn lifecycle
    (agent.turn_context._bind_turn_identity) stamps -- the genuine per-attempt identity."""

    def __init__(self, turn_id: str):
        self._current_turn_id = turn_id
        self._interrupt_requested = False
        self.tool_progress_callback = None


def _make_client(turn_id: str, preset_name: str = "default"):
    from agent.moa_loop import build_moa_facade

    agent = _FakeAgent(turn_id)
    client = build_moa_facade(agent, preset_name)
    return client, agent


def _managed_preset() -> dict:
    return {
        "enabled": True,
        "reference_models": [
            {"provider": "does-not-matter", "model": "does-not-matter", "routing_role": "moareference",
             "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}},
        ],
        "aggregator": {"provider": "does-not-matter", "model": "does-not-matter", "routing_role": "moaaggregator",
                       "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}},
    }


def _write_moa_config(hermes_home, preset: dict, preset_name: str = "default"):
    from hermes_cli.config import load_config, save_config
    cfg = load_config() or {}
    cfg["moa"] = {"default_preset": preset_name, "presets": {preset_name: preset}}
    save_config(cfg)


def test_two_separate_runs_each_resolve_and_pin_their_own_cohort(routed_home, monkeypatch):
    """Two distinct live runs (distinct turn ids / execution ids) each independently resolve and
    reach the real endpoint -- no cross-run cache bleed, no shared/constant execution_id."""
    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url)
    _patch_custom_provider(monkeypatch, url)
    _write_moa_config(hermes_home, _managed_preset())

    client_a, agent_a = _make_client("turn-AAA")
    result_a = client_a.chat.completions.create(
        messages=[{"role": "user", "content": "Run A: what should I do next?"}],
    )
    assert result_a is not None

    client_b, agent_b = _make_client("turn-BBB")
    result_b = client_b.chat.completions.create(
        messages=[{"role": "user", "content": "Run B: what should I do next?"}],
    )
    assert result_b is not None

    # Each run fans out to 1 reference + 1 aggregator call = 2 real requests; two runs = 4 total.
    assert len(handler.requests) == 4, "each independent run must reach the real endpoint on its own"

    from agent.moa_model_routing import _cohort_cache, _origin_home_key
    home_key = _origin_home_key(hermes_home)
    assert ("turn-AAA", home_key) in _cohort_cache and ("turn-BBB", home_key) in _cohort_cache
    assert _cohort_cache[("turn-AAA", home_key)] is not _cohort_cache[("turn-BBB", home_key)], (
        "two separate runs must never share the identical pinned cohort object"
    )


def test_multi_iteration_same_run_reuses_pinned_cohort_receipt(routed_home, monkeypatch):
    """Repeated iterations within the SAME run (same turn id) must reuse the identical pinned
    cohort resolution -- never re-resolve/reselect mid-run."""
    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url)
    _patch_custom_provider(monkeypatch, url)
    _write_moa_config(hermes_home, _managed_preset())

    client, agent = _make_client("turn-multi-iter")

    from agent.moa_model_routing import resolve_moa_slot_route_pinned

    seen_resolutions = []
    orig = resolve_moa_slot_route_pinned

    def _spy(slot, *, execution_id, attempt_id="0", slot_id):
        resolved = orig(slot, execution_id=execution_id, attempt_id=attempt_id, slot_id=slot_id)
        seen_resolutions.append((execution_id, slot_id, id(resolved)))
        return resolved

    import agent.moa_model_routing as mmr
    monkeypatch.setattr(mmr, "resolve_moa_slot_route_pinned", _spy)

    for i in range(3):
        client.chat.completions.create(
            messages=[{"role": "user", "content": f"Iteration {i}: same run, next step?"}],
        )

    ref_ids = {sr[2] for sr in seen_resolutions if sr[1] == "reference-0"}
    agg_ids = {sr[2] for sr in seen_resolutions if sr[1] == "aggregator"}
    assert len(ref_ids) == 1, f"reference-0 must resolve to the SAME pinned object across iterations, got {ref_ids}"
    assert len(agg_ids) == 1, f"aggregator must resolve to the SAME pinned object across iterations, got {agg_ids}"


def test_config_edit_mid_run_never_silently_reroutes(routed_home, monkeypatch):
    """A ROUTINE live policy edit (publish + activate a new revision, same route,
    no explicit revocation) performed AFTER a run's cohort is pinned must never affect
    that in-flight run at all (design §12: "Routine policy edits affect new attempts,
    not active conversations"). The pinned receipt keeps serving the run unchanged --
    only an EXPLICIT emergency revocation (see
    ``test_emergency_revocation_mid_run_blocks_next_request`` below) blocks it. A
    subsequent NEW run (new turn id) observes the routine edit normally."""
    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url, route_id="route-v1", revision=1)
    _patch_custom_provider(monkeypatch, url)
    _write_moa_config(hermes_home, _managed_preset())

    client, agent = _make_client("turn-config-edit")
    client.chat.completions.create(
        messages=[{"role": "user", "content": "Before the edit: what next?"}],
    )
    first_round_requests = len(handler.requests)
    assert first_round_requests == 2  # 1 reference + 1 aggregator

    from agent.moa_model_routing import _cohort_cache, _origin_home_key
    home_key = _origin_home_key(hermes_home)
    pinned_before = dict(_cohort_cache.get(("turn-config-edit", home_key)) or {})
    assert pinned_before, "the cohort must be pinned after the first call in this run"

    # Routine mid-run policy edit: republish + activate a new revision of the SAME route,
    # with NO explicit revocation. This must be a complete no-op for the already-pinned,
    # in-flight run -- the next call in the SAME run succeeds, reusing the identical pinned
    # receipt, never re-resolving and never failing closed.
    _publish_active(hermes_home, url, route_id="route-v1", revision=2)

    client.chat.completions.create(
        messages=[{"role": "user", "content": "After the routine edit, same run: what next?"}],
    )
    assert len(handler.requests) == first_round_requests + 2, (
        "a ROUTINE mid-run policy edit (no explicit revocation) must never block the "
        "in-flight run's next request"
    )

    pinned_after = dict(_cohort_cache.get(("turn-config-edit", home_key)) or {})
    assert pinned_after.keys() == pinned_before.keys()
    for slot_id in pinned_before:
        before_val, after_val = pinned_before[slot_id], pinned_after[slot_id]
        if before_val is None:
            assert after_val is None
        else:
            assert before_val["receipt_id"] == after_val["receipt_id"], (
                f"slot {slot_id} was re-resolved after a routine mid-run config edit -- the "
                "pinned cohort must remain immutable for the life of this run"
            )

    # A brand-new run (new turn id) picks up the edited policy fresh and succeeds normally.
    client_new, _ = _make_client("turn-after-edit")
    client_new.chat.completions.create(
        messages=[{"role": "user", "content": "Brand new run after the edit: what next?"}],
    )


def test_emergency_revocation_mid_run_blocks_next_request(routed_home, monkeypatch):
    """An EXPLICIT emergency revocation (``model_selection_store.revoke_route``), issued
    AFTER a run's cohort is pinned, must block that in-flight run's NEXT request -- fail
    closed, never silently reroute to a different route (design §12: "it never substitutes
    another model"). This is distinct from an ordinary republish/activate, which must never
    have this effect (see the routine-edit test above)."""
    from agent.moa_model_routing import MoARequiredSlotDenied

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url, route_id="route-v1", revision=1)
    _patch_custom_provider(monkeypatch, url)
    _write_moa_config(hermes_home, _managed_preset())

    client, agent = _make_client("turn-revoke")
    client.chat.completions.create(
        messages=[{"role": "user", "content": "Before the revocation: what next?"}],
    )
    first_round_requests = len(handler.requests)
    assert first_round_requests == 2

    from agent.model_selection_store import revoke_route
    revoke_route(
        hermes_home, "kanban-default", route_id="route-v1",
        reason="emergency: credential compromised", approval_ref="operator:test-emergency",
    )

    with pytest.raises(MoARequiredSlotDenied):
        client.chat.completions.create(
            messages=[{"role": "user", "content": "After the revocation, same run: what next?"}],
        )
    assert len(handler.requests) == first_round_requests, (
        "an explicit emergency revocation must never let the in-flight run reach the "
        "endpoint again under a silently substituted route"
    )


@pytest.mark.parametrize("mutation", [None, "model", "reasoning", "reserve", "messages"])
def test_streaming_aggregator_validates_after_auxiliary_transform(routed_home, monkeypatch, mutation):
    import importlib
    from agent.model_selection_types import RoutingBlocked

    auxiliary = importlib.import_module("agent.auxiliary_client")
    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    _write_moa_config(home, _managed_preset())
    client, agent = _make_client("stream-wire")
    original = auxiliary._relay_sync_stream
    reached = []

    def altered(sdk_client, kwargs, **options):
        reached.append(True)
        kwargs = dict(kwargs)
        if mutation == "model":
            kwargs["model"] = "unapproved-model"
        elif mutation == "reasoning":
            kwargs["extra_body"] = {"reasoning": {"effort": "low"}}
            kwargs["reasoning_effort"] = "low"
        elif mutation == "reserve":
            kwargs["max_tokens"] = 200001
        elif mutation == "messages":
            kwargs["messages"] = [{"role": "user", "content": "x" * 300000}]
        return original(sdk_client, kwargs, **options)

    monkeypatch.setattr(auxiliary, "_relay_sync_stream", altered)

    def execute():
        return list(client.chat.completions.create(
            messages=[{"role": "user", "content": "Stream a bounded answer"}], stream=True,
        ))

    if mutation is None:
        assert execute()
        assert len(handler.requests) == 2
        assert handler.requests[-1]["stream"] is True
    else:
        with pytest.raises(RoutingBlocked):
            execute()
        assert len(handler.requests) == 1, "only the unmodified reference may reach the endpoint"
    assert reached, "the real streaming auxiliary sender must be exercised"
