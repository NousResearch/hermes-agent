"""Lifecycle ``model_profile`` routing: SubagentLaunchRequest resolves delegation profiles
through the SAME seam delegate_task uses (agent.delegation_model_routing.resolve_profile_route),
so a profile name means the same provider/model on both APIs."""

import time
from types import SimpleNamespace

import pytest

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
)

PROFILES_CFG = {"profiles": {"small": {"provider": "openrouter", "model": "prof/small-model"}}}


class FakeChild:
    def __init__(self, ident="sa-prof"):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "test"
        self.model = "test-model"
        self.interrupted = False


def _run(_index, _goal, _child, _parent):
    return {"status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0.01}


@pytest.fixture
def captured(monkeypatch):
    """Lifecycle service wired to a fake parent; captures _build_child_agent kwargs."""
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return FakeChild(f"sa-{len(calls)}")

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", _run)
    parent = SimpleNamespace(session_id="parent-prof", enabled_toolsets=["file"])
    return SubagentLifecycleService(lambda: parent), calls


@pytest.fixture
def fake_routing(monkeypatch):
    """Stub the resolver's lazy collaborators at their source modules and pin the delegation cfg."""

    def _resolve(*, requested=None, explicit_api_key=None, explicit_base_url=None, target_model=None):
        return {
            "provider": requested or "resolved-default",
            "model": target_model,
            "base_url": "https://api.example.test/v1",
            "api_key": "sk-test",
            "api_mode": "chat_completions",
        }

    import hermes_cli.runtime_provider as rp
    import agent.models_dev as md
    monkeypatch.setattr(rp, "resolve_runtime_provider", _resolve)
    monkeypatch.setattr(
        md, "get_model_capabilities", lambda *a, **k: SimpleNamespace(supports_tools=True)
    )
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: dict(PROFILES_CFG))


def test_parity_lifecycle_and_delegate_task_resolve_same_route(captured, fake_routing):
    """The lifecycle path and the delegate_task oracle (resolve_profile_route, the shared seam
    delegate_tool_config.py calls at :365-366) must land on the same (model, provider)."""
    from agent.delegation_model_routing import resolve_profile_route

    oracle = resolve_profile_route("small", PROFILES_CFG)

    service, calls = captured
    service.launch(SubagentLaunchRequest(goal="parity", model_profile="small"))
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["model"] == oracle.model == "prof/small-model"
    assert kwargs["override_provider"] == oracle.provider == "openrouter"


def test_both_model_and_model_profile_rejected(captured, fake_routing):
    service, calls = captured
    with pytest.raises(SubagentLifecycleError, match="model_profile"):
        service.launch(SubagentLaunchRequest(goal="x", model="gpt-x", model_profile="small"))
    assert calls == []


def test_unknown_profile_errors_naming_configured_profiles(captured, fake_routing):
    service, calls = captured
    with pytest.raises(SubagentLifecycleError) as exc:
        service.launch(SubagentLaunchRequest(goal="x", model_profile="huge"))
    msg = str(exc.value)
    assert "huge" in msg and "small" in msg
    assert calls == []


def test_no_profile_keeps_construction_kwargs_unchanged(captured, fake_routing):
    """model_profile=None must be byte-identical to today's behavior: no override_* kwargs
    appear, and model passes through verbatim."""
    service, calls = captured
    service.launch(SubagentLaunchRequest(goal="plain", model="raw-model"))
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["model"] == "raw-model"
    assert not any(k.startswith("override_") for k in kwargs)
    assert "requested_profile" not in kwargs


@pytest.mark.parametrize("profile_iter,budget,expected", [
    (100, 50, 50),  # profile can never WIDEN the configured budget
    (7, 50, 7),     # profile may tighten it
])
def test_profile_max_iterations_clamps_against_configured_budget(
        captured, monkeypatch, profile_iter, budget, expected):
    """The lifecycle path clamps min(profile, delegation.max_iterations) — the CONFIGURED
    budget, same as delegate_task, not the DEFAULT_MAX_ITERATIONS constant. Both directions
    are asserted so a min→max mutation cannot survive."""
    from types import SimpleNamespace as _NS
    cfg = {
        "max_iterations": budget,
        "profiles": {"small": {
            "provider": "openrouter", "model": "prof/small-model", "max_iterations": profile_iter,
        }},
    }
    import hermes_cli.runtime_provider as rp
    import agent.models_dev as md
    monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **kw: {
        "provider": kw.get("requested"), "model": kw.get("target_model"),
        "base_url": "https://api.example.test/v1", "api_key": "sk-test",
        "api_mode": "chat_completions",
    })
    monkeypatch.setattr(md, "get_model_capabilities", lambda *a, **k: _NS(supports_tools=True))
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)

    service, calls = captured
    service.launch(SubagentLaunchRequest(goal="budget", model_profile="small"))
    assert len(calls) == 1
    assert calls[0]["max_iterations"] == expected
