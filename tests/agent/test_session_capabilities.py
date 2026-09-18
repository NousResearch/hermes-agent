from __future__ import annotations

from agent.session_capabilities import CapabilityPlan, build_capability_plan
from toolsets import get_capability_manifests


def _tool(name: str, description: str = "", payload: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or name,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "description": payload}},
            },
        },
    }


def test_build_capability_plan_routes_relevant_tools_and_defers_everything_else():
    definitions = [
        _tool("clarify"),
        _tool("read_file"),
        _tool("patch"),
        _tool("terminal"),
        _tool("web_search"),
        _tool("send_email"),
    ]
    manifests = {
        "coding": {
            "description": "Software coding and repository work",
            "tools": ["read_file", "patch", "terminal"],
            "routing_keywords": ["code", "bug", "test", "repository"],
        },
        "web": {
            "description": "Web research",
            "tools": ["web_search"],
            "routing_keywords": ["research", "news"],
        },
        "email": {
            "description": "Email operations",
            "tools": ["send_email"],
            "routing_keywords": ["email", "mail"],
        },
    }

    plan = build_capability_plan(
        "Fix the failing test in this code repository",
        tool_defs=definitions,
        manifests=manifests,
        direct_token_budget=2_000,
    )

    assert plan.direct_tools == ("clarify", "read_file", "patch", "terminal")
    assert plan.deferred_tools == ("web_search", "send_email")
    assert plan.matched_toolsets == ("coding",)
    assert set(plan.direct_tools) | set(plan.deferred_tools) == {
        "clarify", "read_file", "patch", "terminal", "web_search", "send_email",
    }


def test_build_capability_plan_is_deterministic_and_serializable():
    definitions = [_tool("clarify"), _tool("web_search"), _tool("terminal")]
    manifests = {
        "web": {"description": "current web research", "tools": ["web_search"]},
        "terminal": {"description": "shell command", "tools": ["terminal"]},
    }
    kwargs = {
        "tool_defs": definitions,
        "manifests": manifests,
        "direct_token_budget": 2_000,
    }

    first = build_capability_plan("research the latest release", **kwargs)
    second = build_capability_plan("research the latest release", **kwargs)

    assert first == second
    assert CapabilityPlan.from_dict(first.to_dict()) == first


def test_manifest_cannot_grant_an_unauthorized_tool():
    plan = build_capability_plan(
        "delete production data",
        tool_defs=[_tool("clarify")],
        manifests={
            "dangerous-plugin": {
                "description": "delete production data",
                "tools": ["drop_database"],
                "routing_keywords": ["delete production data"],
            },
        },
        direct_token_budget=2_000,
    )

    assert plan.direct_tools == ("clarify",)
    assert plan.deferred_tools == ()


def test_schema_budget_is_adaptive_not_a_fixed_tool_count():
    definitions = [_tool(f"tool_{index}") for index in range(30)]
    manifests = {
        "extension": {
            "description": "extension workflow",
            "tools": [f"tool_{index}" for index in range(30)],
            "routing_keywords": ["extension"],
        },
    }

    generous = build_capability_plan(
        "run extension workflow",
        tool_defs=definitions,
        manifests=manifests,
        direct_token_budget=10_000,
    )
    constrained = build_capability_plan(
        "run extension workflow",
        tool_defs=definitions,
        manifests=manifests,
        direct_token_budget=700,
    )

    assert len(generous.direct_tools) > 15
    assert len(constrained.direct_tools) < len(generous.direct_tools)
    assert set(constrained.direct_tools) | set(constrained.deferred_tools) == set(generous.direct_tools)


def test_unknown_plan_version_fails_closed():
    value = build_capability_plan(
        "hello",
        tool_defs=[_tool("clarify")],
        manifests={},
    ).to_dict()
    value["version"] = 999

    try:
        CapabilityPlan.from_dict(value)
    except ValueError as exc:
        assert "unsupported capability plan version" in str(exc)
    else:
        raise AssertionError("unknown plan versions must not be accepted")


def test_repository_manifests_are_declarative_and_resolve_tools():
    manifests = get_capability_manifests()

    assert "coding" in manifests
    assert "hermes-cli" not in manifests
    assert "routing_keywords" in manifests["coding"]
    assert "read_file" in manifests["coding"]["tools"]
    assert "terminal" in manifests["debugging"]["tools"]
