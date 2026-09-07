"""Top-level dispatch identity and awaited-mode integration tests."""

import json

import pytest

import run_agent
from tools import async_delegation as ad


def _agent():
    agent = object.__new__(run_agent.AIAgent)
    agent._delegate_depth = 0
    agent._current_turn_id = "turn-1"
    agent._current_work_id = ""
    agent._current_work_generation = 0
    agent._current_work_delivery_id = ""
    agent._current_work_claim_id = ""
    return agent


def _capture(monkeypatch, *, enabled, supported, args):
    captured = {}
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: enabled)
    monkeypatch.setattr(
        "gateway.session_context.closeout_delivery_supported", lambda: supported
    )

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return json.dumps({"status": "dispatched"})

    monkeypatch.setattr("tools.delegate_tool.delegate_task", fake_delegate)
    result = run_agent.AIAgent._dispatch_delegate_task(_agent(), args)
    return captured, json.loads(result)


def test_default_off_preserves_ignored_explicit_false(monkeypatch):
    captured, _ = _capture(
        monkeypatch, enabled=False, supported=True,
        args={"goal": "review", "background": False},
    )
    assert captured["background"] is True
    assert captured["origin_work_id"] == ""


def test_enabled_explicit_false_is_synchronous_inline(monkeypatch):
    captured, _ = _capture(
        monkeypatch, enabled=True, supported=True,
        args={"goal": "review", "background": False},
    )
    assert captured["background"] is False
    assert captured["origin_work_id"] == ""


@pytest.mark.parametrize("background", [None, True])
def test_enabled_omitted_or_true_allocates_tracked_work(monkeypatch, background):
    args = {"goal": "research"}
    if background is not None:
        args["background"] = background
    captured, _ = _capture(monkeypatch, enabled=True, supported=True, args=args)
    assert captured["background"] is True
    assert captured["origin_work_id"]
    assert captured["work_generation"] == 0


def test_unsupported_surface_creates_no_group_identity(monkeypatch):
    captured, _ = _capture(
        monkeypatch, enabled=True, supported=False, args={"goal": "research"}
    )
    assert captured["background"] is False
    assert captured["origin_work_id"] == ""


def test_replacement_passes_next_generation_and_claim(monkeypatch):
    agent = _agent()
    agent._current_work_id = "work-1"
    agent._current_work_generation = 3
    agent._current_work_delivery_id = "delivery-3"
    agent._current_work_claim_id = "claim-3"
    captured = {}
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: True)
    monkeypatch.setattr("gateway.session_context.closeout_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_tool.delegate_task",
        lambda **kwargs: captured.update(kwargs) or json.dumps({"status": "dispatched"}),
    )
    run_agent.AIAgent._dispatch_delegate_task(agent, {"goal": "replacement"})
    assert captured["origin_work_id"] == "work-1"
    assert captured["work_generation"] == 4
    assert captured["closeout_delivery_id"] == "delivery-3"
    assert captured["closeout_claim_id"] == "claim-3"
    assert agent._current_work_generation == 4
    assert agent._current_work_delivery_id == ""
    assert agent._current_work_claim_id == ""


def test_existing_group_keeps_replacement_semantics_after_disable(monkeypatch):
    agent = _agent()
    agent._current_work_id = "work-1"
    agent._current_work_generation = 3
    agent._current_work_delivery_id = "delivery-3"
    agent._current_work_claim_id = "claim-3"
    captured = {}
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: False)
    monkeypatch.setattr("gateway.session_context.closeout_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_tool.delegate_task",
        lambda **kwargs: captured.update(kwargs) or json.dumps({"status": "dispatched"}),
    )

    run_agent.AIAgent._dispatch_delegate_task(agent, {"goal": "replacement"})

    assert captured["background"] is True
    assert captured["origin_work_id"] == "work-1"
    assert captured["work_generation"] == 4
    assert captured["closeout_delivery_id"] == "delivery-3"


@pytest.mark.parametrize("replacement", [False, True])
def test_multiple_spawns_share_generation_and_rejection_preserves_identity(monkeypatch, replacement):
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: True)
    monkeypatch.setattr("gateway.session_context.closeout_delivery_supported", lambda: True)
    agent = _agent()
    if replacement:
        assert ad.register_work_group_member(
            work_id="work", owner_turn_id="owner", delegation_id="original",
            feature_config={"delegation": {"task_scoped_closeout": True}},
        )
        ad.persist_group_member_completion("original", {"status": "completed"}, {})
        assert ad.seal_work_group("work", "owner")
        claim = ad.claim_ready_work_group("work", "test")
        delivery = claim["envelope"]["delivery_id"]
        assert ad.bind_work_group_closeout_turn("work", delivery, claim["claim_id"], "turn-1")
        agent._current_work_id = "work"
        agent._current_work_delivery_id = delivery
        agent._current_work_claim_id = claim["claim_id"]

    def identity():
        return (agent._current_work_id, agent._current_work_generation,
                agent._current_work_delivery_id, agent._current_work_claim_id)

    captures = []
    def ledger_delegate(**kwargs):
        captures.append(kwargs)
        if kwargs["goal"] == "reject":
            return json.dumps({"status": "error"})
        accepted = ad._register_grouped_dispatch(
            dict(delegation_id=f"child-{len(captures)}", origin_work_id=kwargs["origin_work_id"],
                 work_generation=kwargs["work_generation"], dispatched_at=1.0),
            owner_turn_id=kwargs["owner_turn_id"],
            closeout_delivery_id=kwargs["closeout_delivery_id"],
            closeout_claim_id=kwargs["closeout_claim_id"],
        )
        return json.dumps({"status": "dispatched" if accepted else "error"})
    monkeypatch.setattr("tools.delegate_tool.delegate_task", ledger_delegate)
    for goal in ("reject", "first", "reject", "second"):
        before = identity()
        result = json.loads(agent._dispatch_delegate_task({"goal": goal}))
        if goal == "reject":
            assert result["status"] == "error"
            assert identity() == before
        else:
            assert result["status"] == "dispatched"
            assert agent._current_work_generation == int(replacement)
    assert captures[-1]["work_generation"] == captures[1]["work_generation"]


@pytest.mark.parametrize("action", ["list", " steer ", "STOP", "invalid"])
@pytest.mark.parametrize("bound", [False, True])
def test_non_spawn_actions_bypass_closeout_identity(monkeypatch, action, bound):
    agent = _agent()
    if bound:
        agent._current_work_id = "work"
        agent._current_work_generation = 3
        agent._current_work_delivery_id = "delivery"
        agent._current_work_claim_id = "claim"
    before = dict(agent.__dict__)
    captured = {}
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: True)
    monkeypatch.setattr("gateway.session_context.closeout_delivery_supported", lambda: True)
    def control(**kwargs):
        captured.update(kwargs)
        assert agent.__dict__ == before
        return json.dumps({"status": "ok"})
    monkeypatch.setattr("tools.delegate_tool.delegate_task", control)
    agent._dispatch_delegate_task({"action": action})
    assert agent.__dict__ == before
    assert captured["origin_work_id"] == captured["closeout_delivery_id"] == captured["closeout_claim_id"] == ""


def test_dynamic_description_changes_only_when_enabled(monkeypatch):
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: False)
    off = __import__("tools.delegate_tool", fromlist=["x"])._build_dynamic_schema_overrides()
    assert "background" not in off["parameters"]["properties"]
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: True)
    on = __import__("tools.delegate_tool", fromlist=["x"])._build_dynamic_schema_overrides()
    assert "required final read-only review" in on["description"]
    assert "Set false" in on["parameters"]["properties"]["background"]["description"]
    assert "origin_work_id" not in on["parameters"]["properties"]


def test_registry_fallback_forces_sync_on_unsupported_closeout_surface(monkeypatch):
    delegate = __import__("tools.delegate_tool", fromlist=["x"])
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: True)
    monkeypatch.setattr(
        "gateway.session_context.closeout_delivery_supported", lambda: False
    )
    assert delegate._model_background_value({}, _agent()) is False
