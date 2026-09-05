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
