from __future__ import annotations

import json
import hashlib
import pytest
from dataclasses import replace
from types import SimpleNamespace

from agent.session_capabilities import (
    CAPABILITY_PLAN_VERSION,
    CapabilityPlan,
    _build_session_plan,
    _plan_hash,
    apply_capability_plan,
    ensure_session_capability_plan,
)
from hermes_state import SessionDB


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _plan() -> CapabilityPlan:
    wire = (_tool("clarify"), _tool("tool_search"), _tool("tool_describe"), _tool("tool_call"))
    fallback = (_tool("terminal"),)
    canonical = json.dumps(list(wire), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    plan = CapabilityPlan(
        version=CAPABILITY_PLAN_VERSION,
        direct_tools=("clarify",),
        deferred_tools=("terminal",),
        matched_toolsets=(),
        selected_skills=(),
        direct_schema_tokens=1,
        intent_hash="intent",
        manifest_hash="manifest",
        tool_schema_hash=hashlib.sha256(canonical.encode()).hexdigest()[:16],
        wire_tool_defs=wire,
        fallback_tool_defs=fallback,
    )
    return replace(plan, plan_hash=_plan_hash(plan))


def test_resume_reads_capability_plan_from_real_session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="routed", source="cli", model="test")
        plan = _plan()
        db.set_session_model_config_value_once("routed", "capability_plan", plan.to_dict())
        agent = SimpleNamespace(
            _session_db=db,
            _session_init_model_config={},
            session_id="routed",
            tools=[_tool("clarify"), _tool("terminal")],
            valid_tool_names={"clarify", "terminal"},
        )

        restored = ensure_session_capability_plan(agent, "different intent", [{"role": "user"}])

        assert restored == plan
        assert agent._capability_plan == plan
        assert agent._capability_fallback_names == ("terminal",)
        assert [item["function"]["name"] for item in agent.tools] == [
            "clarify", "tool_search", "tool_describe", "tool_call",
        ]
    finally:
        db.close()


def test_first_turn_persists_once_through_real_session_db(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            session_id="new", source="cli", model="test",
            model_config={"reasoning_effort": "low"},
        )
        plan = _plan()
        monkeypatch.setattr("agent.session_capabilities.intent_routing_enabled", lambda: True)
        monkeypatch.setattr("agent.session_capabilities._build_session_plan", lambda *_args: plan)
        agent = SimpleNamespace(
            _session_db=db,
            _session_init_model_config={"reasoning_effort": "low"},
            session_id="new",
            tools=[_tool("clarify"), _tool("terminal")],
            valid_tool_names={"clarify", "terminal"},
        )

        winner = ensure_session_capability_plan(agent, "run a command", [])
        stored = json.loads(db.get_session("new")["model_config"])

        assert winner == plan
        assert stored["reasoning_effort"] == "low"
        assert CapabilityPlan.from_dict(stored["capability_plan"]) == plan
    finally:
        db.close()


def test_dynamic_kernel_tool_is_not_duplicated_in_wire_schema(monkeypatch):
    monkeypatch.setattr("agent.session_capabilities._routing_settings", lambda: {})
    monkeypatch.setattr("agent.session_capabilities._rank_skills", lambda *_args: ())
    extra = _tool("dynamic_context_tool")
    agent = SimpleNamespace(
        _authorized_tool_defs_snapshot=(_tool("clarify"),),
        tools=[_tool("clarify"), extra],
        enabled_toolsets=None,
        disabled_toolsets=None,
    )

    plan = _build_session_plan(agent, "use dynamic context")
    names = [item["function"]["name"] for item in plan.wire_tool_defs]

    assert names.count("dynamic_context_tool") == 1


def test_direct_connection_manager_keeps_connector_bridge_authorized():
    import model_tools
    from tools.connectors import CONNECTOR_BATCH_SENTINEL

    base = _plan()
    wire = tuple(base.wire_tool_defs) + (_tool("manage_connections"),)
    canonical = json.dumps(list(wire), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    plan = replace(
        base,
        direct_tools=("clarify", "manage_connections"),
        wire_tool_defs=wire,
        tool_schema_hash=hashlib.sha256(canonical.encode()).hexdigest()[:16],
    )
    plan = replace(plan, plan_hash=_plan_hash(plan))
    agent = SimpleNamespace(tools=[], valid_tool_names=set())
    apply_capability_plan(agent, plan)

    result, underlying = model_tools._dispatch_bridge_tool(
        "tool_call",
        {"calls": [{"name": "connectors__gmail__SEND_EMAIL", "arguments": {}}]},
        None,
        None,
        bridge_tool_defs=list(agent._capability_bridge_tool_defs),
        bridge_allowed_names=list(agent._capability_fallback_names),
    )

    assert result is None
    assert isinstance(underlying, tuple)
    assert underlying[0] == CONNECTOR_BATCH_SENTINEL


def test_persisted_plan_integrity_covers_fallback_and_skills():
    payload = _plan().to_dict()
    payload["selected_skills"] = ["injected-skill"]
    with pytest.raises(ValueError, match="integrity hash"):
        CapabilityPlan.from_dict(payload)


def test_resume_fails_closed_when_authorized_tool_disappears():
    agent = SimpleNamespace(
        tools=[_tool("clarify")],
        valid_tool_names={"clarify"},
        _authorized_tool_defs_snapshot=(_tool("clarify"),),
    )
    with pytest.raises(ValueError, match="no longer authorized or available"):
        apply_capability_plan(agent, _plan())


def test_resume_fails_closed_when_authorization_scope_is_empty():
    agent = SimpleNamespace(
        tools=[],
        valid_tool_names=set(),
        _authorized_tool_defs_snapshot=(),
    )
    with pytest.raises(ValueError, match="no longer authorized or available"):
        apply_capability_plan(agent, _plan())


def test_skill_visibility_uses_authorized_not_only_direct_tools(monkeypatch):
    from agent import system_prompt

    captured = {}
    monkeypatch.setattr(
        system_prompt._pb,
        "build_skills_system_prompt",
        lambda **kwargs: captured.update(kwargs) or "skills",
    )
    monkeypatch.setattr(system_prompt, "resolve_context_cwd", lambda: None)
    agent = SimpleNamespace(
        valid_tool_names={"skills_list", "skills_view"},
        _authorized_tool_names=("skills_list", "skills_view", "read_file"),
        _capability_plan=None,
        platform="cli",
    )

    assert system_prompt._skills_prompt(agent) == "skills"
    assert "read_file" in captured["available_tools"]
    assert "file" in captured["available_toolsets"]
