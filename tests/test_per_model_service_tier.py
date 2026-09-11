"""Request-time per-model service_tier_overrides and /fast pin provenance.

Resolution lives in ``agent.fast_mode.effective_request_overrides``:

session ``/fast`` pin > ``agent.service_tier_overrides`` > global ``agent.service_tier``.

Routing overlays belong in ``tests/agent/test_per_model_provider_routing.py``.
"""

from types import SimpleNamespace

import pytest

from agent import fast_mode
from hermes_cli.models import resolve_fast_mode_overrides, resolve_service_tier_overrides
from hermes_constants import parse_service_tier, resolve_service_tier_for_model, service_tier_status_label


_OPENROUTER = "https://openrouter.ai/api/v1"
_OPENAI = "https://api.openai.com/v1"
_NOUS = "https://inference-api.nousresearch.com/v1"
_MATCH = "openai/gpt-5"


def _agent(**kw):
    base = dict(
        service_tier=None,
        model=_MATCH,
        provider="openrouter",
        base_url=_OPENROUTER,
        api_mode="chat_completions",
        request_overrides={"extra_body": {"keep": 1}},
        fast_auto_seconds=60,
        _service_tier_session_pinned=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def tier_cfg(monkeypatch):
    cfg = {
        "agent": {
            "service_tier": "priority",
            "service_tier_overrides": {_MATCH: "flex"},
        }
    }
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
    return cfg


def _empty_tier_cfg(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"agent": {"service_tier": "", "service_tier_overrides": {}}},
    )


def test_default_config_effective_overrides_match_baseline(monkeypatch):
    """Default-off: no new keys, no empty objects, extra_body unchanged."""
    _empty_tier_cfg(monkeypatch)
    agent = _agent(provider="openai", base_url=_OPENAI, request_overrides={"extra_body": {"keep": 1}})
    orig = dict(agent.request_overrides)
    assert fast_mode.effective_request_overrides(agent) == {"extra_body": {"keep": 1}}
    assert agent.request_overrides == orig

    empty = _agent(provider="openai", base_url=_OPENAI, request_overrides={})
    assert fast_mode.effective_request_overrides(empty) == {}


def test_raw_request_overrides_passthrough_default_config(monkeypatch):
    """User-supplied service_tier reaches the wire when no framework source applies."""
    _empty_tier_cfg(monkeypatch)
    agent = _agent(
        request_overrides={"service_tier": "priority", "extra_body": {"keep": 1}},
    )
    orig = dict(agent.request_overrides)
    wire = fast_mode.effective_request_overrides(agent)
    assert wire["service_tier"] == "priority"
    assert wire["extra_body"] == {"keep": 1}
    assert agent.request_overrides == orig


def test_delegated_child_request_overrides_service_tier(monkeypatch):
    """delegation.request_overrides service_tier is a raw user key, not a bake."""
    from tools.delegate_tool import _resolve_delegation_credentials

    _empty_tier_cfg(monkeypatch)
    creds = _resolve_delegation_credentials(
        {
            "model": "openai/gpt-5",
            "base_url": _OPENROUTER,
            "api_key": "test-key-1234567890",
            "request_overrides": {"service_tier": "priority"},
        },
        parent_agent=None,
    )
    child = _agent(request_overrides=dict(creds["request_overrides"] or {}))
    assert fast_mode.effective_request_overrides(child)["service_tier"] == "priority"


def test_framework_pin_overrides_raw_request_keys(monkeypatch):
    """Session pin (including flex) wins over raw service_tier/speed on the dict."""
    _empty_tier_cfg(monkeypatch)
    agent = _agent(
        service_tier="flex",
        _service_tier_session_pinned=True,
        request_overrides={"service_tier": "priority", "speed": "fast", "extra_body": {"keep": 1}},
    )
    wire = fast_mode.effective_request_overrides(agent)
    assert wire["service_tier"] == "flex"
    assert "speed" not in wire
    assert wire["extra_body"] == {"keep": 1}
    assert agent.request_overrides["service_tier"] == "priority"


def test_framework_baked_tier_keys_stripped_when_source_removed(monkeypatch):
    """Marked loader keys do not leak after the framework source is gone; raw keys do."""
    _empty_tier_cfg(monkeypatch)
    baked = _agent(
        service_tier=None,
        _service_tier_session_pinned=False,
        request_overrides={"service_tier": "priority", "extra_body": {"keep": 1}},
    )
    fast_mode.set_framework_baked_tier_keys(baked, {"service_tier": "priority"})
    baked_wire = fast_mode.effective_request_overrides(baked)
    assert "service_tier" not in baked_wire
    assert "speed" not in baked_wire
    assert baked_wire["extra_body"] == {"keep": 1}
    assert baked.request_overrides["service_tier"] == "priority"

    raw = _agent(
        service_tier=None,
        _service_tier_session_pinned=False,
        request_overrides={"service_tier": "priority", "extra_body": {"keep": 1}},
    )
    fast_mode.set_framework_baked_tier_keys(raw, None)
    raw_wire = fast_mode.effective_request_overrides(raw)
    assert raw_wire["service_tier"] == "priority"
    assert raw_wire["extra_body"] == {"keep": 1}


def test_pin_beats_per_model_beats_global(tier_cfg):
    unpinned = _agent()
    assert fast_mode.logical_service_tier(unpinned) == "flex"
    assert fast_mode.effective_request_overrides(unpinned)["service_tier"] == "flex"

    other = _agent(model="moonshotai/kimi-k2.6")
    assert fast_mode.logical_service_tier(other) == "priority"
    assert fast_mode.effective_request_overrides(other)["service_tier"] == "priority"

    pinned_normal = _agent(_service_tier_session_pinned=True, service_tier=None)
    assert fast_mode.logical_service_tier(pinned_normal) is None
    assert "service_tier" not in fast_mode.effective_request_overrides(pinned_normal)

    pinned_priority = _agent(_service_tier_session_pinned=True, service_tier="priority")
    assert fast_mode.logical_service_tier(pinned_priority) == "priority"
    assert fast_mode.effective_request_overrides(pinned_priority)["service_tier"] == "priority"


def test_fast_normal_pin_blocks_per_model_flex(tier_cfg):
    agent = _agent(_service_tier_session_pinned=True, service_tier=None)
    assert service_tier_status_label(fast_mode.logical_service_tier(agent)) == "normal"
    wire = fast_mode.effective_request_overrides(agent)
    assert "service_tier" not in wire
    assert wire["extra_body"] == {"keep": 1}


def test_unpinned_model_switch_picks_new_overlay(tier_cfg):
    agent = _agent()
    assert fast_mode.effective_request_overrides(agent)["service_tier"] == "flex"
    agent.model = "moonshotai/kimi-k2.6"
    assert fast_mode.effective_request_overrides(agent)["service_tier"] == "priority"


def test_spelling_tolerant_override_match():
    cfg = {"service_tier_overrides": {"gpt-5": "flex"}, "service_tier": ""}
    assert resolve_service_tier_for_model(cfg, "openai/gpt-5") == "flex"
    assert resolve_service_tier_for_model(cfg, "openrouter/openai/gpt-5") == "flex"
    assert resolve_service_tier_for_model(cfg, "gpt-5") == "flex"
    assert resolve_service_tier_for_model(cfg, "other-model", fallback="priority") == "priority"


def test_fallback_model_uses_that_models_overlay(tier_cfg):
    agent = _agent(model=_MATCH)
    assert fast_mode.effective_request_overrides(agent)["service_tier"] == "flex"
    agent.model = "openai/gpt-4.1"
    assert fast_mode.effective_request_overrides(agent)["service_tier"] == "priority"


def test_delegated_child_does_not_inherit_parent_pin(tier_cfg):
    parent = _agent(_service_tier_session_pinned=True, service_tier=None)
    child = _agent(_service_tier_session_pinned=False, model=_MATCH)
    assert fast_mode.logical_service_tier(parent) is None
    assert fast_mode.logical_service_tier(child) == "flex"
    assert fast_mode.effective_request_overrides(child)["service_tier"] == "flex"


def test_nous_and_first_party_ignore_flex():
    nous = _agent(
        provider="nous", base_url=_NOUS, service_tier="flex",
        _service_tier_session_pinned=True,
    )
    wire = fast_mode.effective_request_overrides(nous)
    assert "service_tier" not in wire
    assert "speed" not in wire
    assert wire["extra_body"] == {"keep": 1}

    openai = _agent(
        provider="openai", base_url=_OPENAI, model="gpt-5.4", service_tier="flex",
        _service_tier_session_pinned=True,
    )
    wire = fast_mode.effective_request_overrides(openai)
    assert "service_tier" not in wire
    assert "speed" not in wire

    portal = resolve_service_tier_overrides(
        "gpt-5.4", "flex", provider="nous", base_url=_NOUS,
    )
    assert portal is None


def test_openrouter_any_catalog_model_gets_flex_and_priority():
    llama = "meta-llama/llama-3.1-8b-instruct"
    assert resolve_fast_mode_overrides(
        llama, provider="openrouter", base_url=_OPENROUTER,
    ) == {"service_tier": "priority"}
    assert resolve_service_tier_overrides(
        llama, "flex", provider="openrouter", base_url=_OPENROUTER,
    ) == {"service_tier": "flex"}
    agent = _agent(model=llama, service_tier="flex", _service_tier_session_pinned=True)
    assert fast_mode.effective_request_overrides(agent)["service_tier"] == "flex"


def test_extra_body_not_rewritten_when_tier_applied(tier_cfg):
    extra = {"keep": 1, "provider": {"order": ["anthropic"]}}
    agent = _agent(request_overrides={"extra_body": extra, "temperature": 0.2})
    wire = fast_mode.effective_request_overrides(agent)
    assert wire["service_tier"] == "flex"
    assert wire["extra_body"] is extra
    assert wire["temperature"] == 0.2
    assert agent.request_overrides["extra_body"] is extra
    assert "service_tier" not in agent.request_overrides


def test_stale_parent_tier_keys_are_replaced(tier_cfg):
    agent = _agent(request_overrides={"service_tier": "priority", "speed": "fast", "extra_body": {"keep": 1}})
    wire = fast_mode.effective_request_overrides(agent)
    assert wire["service_tier"] == "flex"
    assert "speed" not in wire
    assert agent.request_overrides["service_tier"] == "priority"


def test_parse_and_status_labels():
    assert parse_service_tier("flex") == "flex"
    assert parse_service_tier("FAST") == "priority"
    assert parse_service_tier("normal") is None
    assert parse_service_tier("") is None
    assert service_tier_status_label("priority") == "fast"
    assert service_tier_status_label("flex") == "flex"
    assert service_tier_status_label(None) == "normal"


def test_explicit_per_model_normal_strips_raw_priority(monkeypatch):
    """Explicit per-model ``normal`` is a framework source; raw tier keys are stripped."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {
            "agent": {
                "service_tier": "",
                "service_tier_overrides": {_MATCH: "normal"},
            }
        },
    )
    agent = _agent(request_overrides={"service_tier": "priority", "extra_body": {"keep": 1}})
    wire = fast_mode.effective_request_overrides(agent)
    assert "service_tier" not in wire
    assert "speed" not in wire
    assert wire["extra_body"] == {"keep": 1}
    assert agent.request_overrides["service_tier"] == "priority"


def test_explicit_per_model_normal_omits_tier_on_wire(monkeypatch):
    """Wire kwargs have no service_tier when per-model normal beats a raw priority."""
    from run_agent import AIAgent

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {
            "agent": {
                "service_tier": "",
                "service_tier_overrides": {_MATCH: "normal"},
            }
        },
    )
    agent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "priority"},
    )
    try:
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in kwargs
        assert "speed" not in kwargs
    finally:
        agent.close()


def test_explicit_global_normal_strips_raw_priority(monkeypatch):
    """Explicit global ``agent.service_tier: normal`` is a framework source; raw keys are stripped."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {
            "agent": {
                "service_tier": "normal",
                "service_tier_overrides": {},
            }
        },
    )
    agent = _agent(request_overrides={"service_tier": "priority", "extra_body": {"keep": 1}})
    wire = fast_mode.effective_request_overrides(agent)
    assert "service_tier" not in wire
    assert "speed" not in wire
    assert wire["extra_body"] == {"keep": 1}
    assert agent.request_overrides["service_tier"] == "priority"


def test_explicit_global_normal_omits_tier_on_wire(monkeypatch):
    """Wire kwargs have no service_tier when global normal beats a raw priority."""
    from run_agent import AIAgent

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {
            "agent": {
                "service_tier": "normal",
                "service_tier_overrides": {},
            }
        },
    )
    agent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "priority"},
    )
    try:
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in kwargs
        assert "speed" not in kwargs
    finally:
        agent.close()


def test_empty_config_raw_priority_passthrough_on_wire(monkeypatch):
    """Default-off empty config still passes a raw request_overrides tier to the wire."""
    from run_agent import AIAgent

    _empty_tier_cfg(monkeypatch)
    agent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "priority"},
    )
    try:
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["service_tier"] == "priority"
    finally:
        agent.close()


def test_delegated_child_strips_parent_baked_tier_on_real_path(monkeypatch):
    """Plain delegated child does not inherit the parent's /fast bake unmarked."""
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc

    _empty_tier_cfg(monkeypatch)
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    kw = dict(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "priority", "extra_body": {"keep": 1}},
        service_tier="priority",
    )
    parent = AIAgent(session_id="p-bake", **kw)
    parent._service_tier_session_pinned = True
    # * /fast marks baked after construction; the constructor does not.
    fast_mode.set_framework_baked_tier_keys(parent, {"service_tier": "priority"})
    child = None
    try:
        creds = dt._resolve_delegation_credentials({"model": "", "provider": ""}, parent)
        child = dt._build_child_agent(
            task_index=0,
            goal="goal",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=4,
            task_count=1,
            parent_agent=parent,
            override_request_overrides=creds.get("request_overrides"),
        )
        assert child._service_tier_session_pinned is False
        assert "service_tier" not in (child.request_overrides or {})
        assert (child.request_overrides or {}).get("extra_body", {}).get("keep") == 1
        kwargs = child._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in kwargs
        assert "speed" not in kwargs
    finally:
        if child is not None:
            child.close()
        parent.close()


def test_constructor_does_not_bake_preexisting_request_override_keys(monkeypatch):
    """Ctor service_tier does not mark caller-supplied request_overrides keys as baked."""
    from run_agent import AIAgent

    _empty_tier_cfg(monkeypatch)
    agent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "flex"},
        service_tier="priority",
    )
    try:
        assert not (getattr(agent, "_framework_baked_tier_keys", None) or ())
        assert agent.request_overrides["service_tier"] == "flex"
        assert agent._build_api_kwargs([{"role": "user", "content": "hi"}])["service_tier"] == "priority"
    finally:
        agent.close()


def test_delegated_child_keeps_parent_raw_when_logical_and_raw_conflict(monkeypatch):
    """Parent ctor priority + raw flex: child inherits unmarked flex, not a baked strip."""
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc

    _empty_tier_cfg(monkeypatch)
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    parent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "flex"},
        service_tier="priority",
        session_id="p-conflict",
    )
    child = None
    try:
        assert parent._build_api_kwargs([{"role": "user", "content": "hi"}])["service_tier"] == "priority"
        creds = dt._resolve_delegation_credentials({"model": "", "provider": ""}, parent)
        child = dt._build_child_agent(
            task_index=0,
            goal="goal",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=4,
            task_count=1,
            parent_agent=parent,
            override_request_overrides=creds.get("request_overrides"),
        )
        assert child._service_tier_session_pinned is False
        assert (child.request_overrides or {}).get("service_tier") == "flex"
        kwargs = child._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["service_tier"] == "flex"
    finally:
        if child is not None:
            child.close()
        parent.close()


def test_delegated_child_keeps_explicit_delegation_tier(monkeypatch):
    """Explicit delegation.request_overrides still reach the child wire unmarked."""
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc

    _empty_tier_cfg(monkeypatch)
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    parent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "priority"},
        service_tier="priority",
        session_id="p-explicit",
    )
    parent._service_tier_session_pinned = True
    child = None
    try:
        creds = dt._resolve_delegation_credentials(
            {"model": "", "provider": "", "request_overrides": {"service_tier": "flex"}},
            parent,
        )
        child = dt._build_child_agent(
            task_index=0,
            goal="goal",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=4,
            task_count=1,
            parent_agent=parent,
            override_request_overrides=creds.get("request_overrides"),
        )
        assert child._service_tier_session_pinned is False
        kwargs = child._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["service_tier"] == "flex"
    finally:
        if child is not None:
            child.close()
        parent.close()


def _cli_openrouter_stub(service_tier="priority"):
    return SimpleNamespace(
        model=_MATCH,
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        service_tier=service_tier,
        _service_tier_session_pinned=service_tier is not None,
    )


def test_cli_fast_parent_bake_not_inherited_by_delegated_child(monkeypatch):
    """Pinned CLI /fast resolve+bake: parent wire keeps the pin; child does not inherit it."""
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    from hermes_cli.cli_agent_setup_mixin import (
        CLIAgentSetupMixin,
        _apply_framework_tier_bake,
    )

    _empty_tier_cfg(monkeypatch)
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    stub = _cli_openrouter_stub("priority")
    route = CLIAgentSetupMixin._resolve_turn_agent_config(stub, "hi")
    parent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides=route["request_overrides"],
        service_tier="priority",
        session_id="p-cli-fast",
    )
    parent._service_tier_session_pinned = True
    _apply_framework_tier_bake(parent, route.get("framework_baked_tier_keys"))
    child = None
    try:
        assert "service_tier" in (getattr(parent, "_framework_baked_tier_keys", None) or ())
        assert parent._build_api_kwargs([{"role": "user", "content": "hi"}])["service_tier"] == "priority"
        creds = dt._resolve_delegation_credentials({"model": "", "provider": ""}, parent)
        child = dt._build_child_agent(
            task_index=0,
            goal="goal",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=4,
            task_count=1,
            parent_agent=parent,
            override_request_overrides=creds.get("request_overrides"),
        )
        assert child._service_tier_session_pinned is False
        assert "service_tier" not in (child.request_overrides or {})
        assert "speed" not in (child.request_overrides or {})
        kwargs = child._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in kwargs
        assert "speed" not in kwargs
    finally:
        if child is not None:
            child.close()
        parent.close()


def test_cli_raw_overrides_stay_unmarked_without_fast(monkeypatch):
    """CLI agent with only user raw overrides (no /fast) keeps them unmarked."""
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    from hermes_cli.cli_agent_setup_mixin import _apply_framework_tier_bake

    _empty_tier_cfg(monkeypatch)
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    parent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "flex"},
        service_tier=None,
        session_id="p-cli-raw",
    )
    _apply_framework_tier_bake(parent, None)
    child = None
    try:
        assert not (getattr(parent, "_framework_baked_tier_keys", None) or ())
        assert parent.request_overrides["service_tier"] == "flex"
        creds = dt._resolve_delegation_credentials({"model": "", "provider": ""}, parent)
        child = dt._build_child_agent(
            task_index=0,
            goal="goal",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=4,
            task_count=1,
            parent_agent=parent,
            override_request_overrides=creds.get("request_overrides"),
        )
        assert (child.request_overrides or {}).get("service_tier") == "flex"
        assert child._build_api_kwargs([{"role": "user", "content": "hi"}])["service_tier"] == "flex"
    finally:
        if child is not None:
            child.close()
        parent.close()


def test_cli_new_release_strips_snapshot_so_restore_cannot_revive_bake(monkeypatch):
    """CLI /new must clear baked keys from ``_primary_runtime`` before fallback restore."""
    from unittest.mock import MagicMock, patch

    from run_agent import AIAgent
    from hermes_cli.cli_agent_setup_mixin import (
        CLIAgentSetupMixin,
        _apply_framework_tier_bake,
        _release_framework_tier_bake,
    )

    _empty_tier_cfg(monkeypatch)
    stub = _cli_openrouter_stub("priority")
    route = CLIAgentSetupMixin._resolve_turn_agent_config(stub, "hi")
    overrides = dict(route["request_overrides"] or {})
    overrides["extra_body"] = {"keep": 1}
    agent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides=overrides,
        service_tier="priority",
        session_id="p-cli-new-snap",
        fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
    )
    agent._service_tier_session_pinned = True
    _apply_framework_tier_bake(agent, route.get("framework_baked_tier_keys"))
    try:
        assert agent._primary_runtime["request_overrides"].get("service_tier") == "priority"
        agent.service_tier = None
        agent._service_tier_session_pinned = False
        _release_framework_tier_bake(agent)
        assert "service_tier" not in (agent.request_overrides or {})
        assert "service_tier" not in (agent._primary_runtime.get("request_overrides") or {})
        assert agent._primary_runtime["request_overrides"].get("extra_body") == {"keep": 1}

        mock = MagicMock()
        mock.base_url = _OPENROUTER
        mock.api_key = "fb-key"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock, "anthropic/claude-sonnet-4"),
        ):
            assert agent._try_activate_fallback() is True
        with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True
        assert "service_tier" not in (agent.request_overrides or {})
        assert agent.request_overrides.get("extra_body") == {"keep": 1}
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in kwargs
        assert "speed" not in kwargs
    finally:
        agent.close()
