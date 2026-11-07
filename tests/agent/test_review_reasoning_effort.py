import agent.review_engine as re_mod
from tools import delegate_tool
from tools.delegate_tool_routes import route_for_config


def test_review_cfg_carries_reasoning_effort(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"auxiliary": {"review": {
        "provider": "openrouter", "model": "some/model", "reasoning_effort": "high"}}})
    assert re_mod._load_review_credentials_cfg()["reasoning_effort"] == "high"


def test_route_for_config_forwards_reasoning_effort(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda cfg, parent: {"model": "m"})
    route = route_for_config({"provider": "openrouter", "reasoning_effort": "low"}, parent_agent=object())
    assert route.reasoning_effort == "low"
