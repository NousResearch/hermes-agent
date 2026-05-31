from types import ModuleType, SimpleNamespace

from hermes_cli.fallback_config import (
    get_fallback_chain,
    is_opus_48_model,
    role_fallback_chain,
    role_has_route,
    role_primary_route,
    sanitize_fallback_chain,
)
from agent.chat_completion_helpers import try_activate_fallback
from skills.productivity.linear.scripts import linear_api


def test_sanitize_fallback_chain_filters_opus_48_and_dedupes():
    raw = [
        {"provider": "anthropic", "model": "claude-opus-4.8"},
        {"provider": "minimax", "model": "MiniMax-M2.7"},
        {"provider": "minimax", "model": "MiniMax-M2.7"},
        {"provider": "openai-codex", "model": "gpt-5.5", "base_url": "https://example.com/"},
    ]

    chain = sanitize_fallback_chain(raw)

    assert len(chain) == 2
    assert all(not is_opus_48_model(entry["model"]) for entry in chain)
    assert chain[0]["provider"] == "minimax"
    assert chain[1]["base_url"] == "https://example.com"


def test_get_fallback_chain_merges_legacy_and_filters_opus_48():
    config = {
        "fallback_providers": [
            {"provider": "anthropic", "model": "claude-opus-4-8"},
            {"provider": "minimax", "model": "MiniMax-M2.7"},
        ],
        "fallback_model": {"provider": "openai-codex", "model": "gpt-5.5"},
    }

    assert get_fallback_chain(config) == [
        {"provider": "minimax", "model": "MiniMax-M2.7"},
        {"provider": "openai-codex", "model": "gpt-5.5"},
    ]


def test_role_routes_match_approved_matrix():
    assert role_primary_route("default", {}) == {
        "provider": "openai-codex",
        "model": "gpt-5.5",
        "reasoning_effort": "medium",
    }
    assert role_primary_route("architect", {}) == {
        "provider": "anthropic",
        "model": "claude-opus-4.8",
        "reasoning_effort": "xhigh",
    }
    assert role_fallback_chain("architect", {}) == [
        {"provider": "openai-codex", "model": "gpt-5.5", "reasoning_effort": "xhigh"},
    ]
    assert role_primary_route("builder", {}) == {
        "provider": "minimax",
        "model": "MiniMax-M2.7",
        "reasoning_effort": "medium",
    }
    assert role_fallback_chain("builder", {}) == [
        {"provider": "openai-codex", "model": "gpt-5.5", "reasoning_effort": "medium"},
    ]
    assert role_primary_route("optimization_pass", {}) == {
        "provider": "openai-codex",
        "model": "gpt-5.5",
        "reasoning_effort": "medium",
    }
    assert role_fallback_chain("hardening_pass", {}) == [
        {"provider": "minimax", "model": "MiniMax-M2.7", "reasoning_effort": "medium"},
    ]
    assert role_primary_route("refactoring_pass", {}) == {
        "provider": "openai-codex",
        "model": "gpt-5.5",
        "reasoning_effort": "medium",
    }
    assert role_fallback_chain("refactor", {}) == [
        {"provider": "minimax", "model": "MiniMax-M2.7", "reasoning_effort": "medium"},
    ]
    assert role_primary_route("final_synthesis", {}) == {
        "provider": "anthropic",
        "model": "claude-opus-4.8",
        "reasoning_effort": "xhigh",
    }
    assert role_fallback_chain("final_synthesis", {}) == [
        {"provider": "openai-codex", "model": "gpt-5.5", "reasoning_effort": "xhigh"},
    ]
    assert role_primary_route("adversarial_review", {}) == {
        "provider": "openai-codex",
        "model": "gpt-5.5",
        "reasoning_effort": "xhigh",
    }
    assert role_fallback_chain("adversarial_review", {}) == []
    assert role_has_route("adversarial_review", {}) is True


def test_role_fallback_chain_honors_legacy_raw_alias_keys():
    cfg = {
        "role_fallbacks": {
            "implementer": [
                {"provider": "custom", "model": "custom-builder", "reasoning_effort": "low"}
            ]
        }
    }

    assert role_fallback_chain("implementer", cfg) == [
        {"provider": "custom", "model": "custom-builder", "reasoning_effort": "low"}
    ]
    assert role_has_route("implementer", cfg) is True
    assert role_has_route("unknown_future_role", {}) is True
    assert role_primary_route("unknown_future_role", {}) == {
        "provider": "openai-codex",
        "model": "gpt-5.5",
        "reasoning_effort": "medium",
    }


def test_role_fallback_chain_builder_default_semantics():
    chain = role_fallback_chain("builder", {})

    assert [(entry["provider"], entry["model"], entry["reasoning_effort"]) for entry in chain] == [
        ("openai-codex", "gpt-5.5", "medium"),
    ]


def test_runtime_fallback_skips_leaked_opus_48(monkeypatch):
    calls = []

    def fake_resolve_provider_client(provider, model, raw_codex=False, explicit_base_url=None, explicit_api_key=None):
        calls.append((provider, model))
        return SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex", api_key="fallback-key"), model

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", fake_resolve_provider_client)

    agent = SimpleNamespace(
        _fallback_chain=[
            {"provider": "anthropic", "model": "claude-opus-4.8"},
            {"provider": "openai-codex", "model": "gpt-5.5", "reasoning_effort": "xhigh"},
        ],
        _fallback_index=0,
        _fallback_activated=False,
        _primary_runtime={"provider": "anthropic"},
        _rate_limited_until=0,
        provider="anthropic",
        model="claude-sonnet-4.6",
        base_url="https://api.anthropic.com",
        api_key="primary",
        api_mode="anthropic_messages",
        request_overrides={},
        _is_azure_openai_url=lambda _url: False,
        _is_direct_openai_url=lambda _url: False,
        _provider_model_requires_responses_api=lambda _model, provider=None: False,
        _anthropic_prompt_cache_policy=lambda **_kwargs: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
        _buffer_status=lambda _message: None,
        _replace_primary_openai_client=lambda **_kwargs: None,
        context_compressor=None,
        _try_activate_fallback=lambda: try_activate_fallback(agent),
        _abort_request_openai_client=lambda *_args, **_kwargs: None,
        _close_request_openai_client=lambda *_args, **_kwargs: None,
    )

    assert try_activate_fallback(agent) is True
    assert calls == [("openai-codex", "gpt-5.5")]
    assert agent.provider == "openai-codex"
    assert agent.model == "gpt-5.5"
    assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}


def test_runtime_fallback_without_effort_clears_stale_reasoning(monkeypatch):
    def fake_resolve_provider_client(provider, model, raw_codex=False, explicit_base_url=None, explicit_api_key=None):
        return SimpleNamespace(base_url="https://api.minimax.io/anthropic", api_key="fallback-key"), model

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", fake_resolve_provider_client)

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "minimax", "model": "MiniMax-M2.7"}],
        _fallback_index=0,
        _fallback_activated=False,
        _primary_runtime={"provider": "openai-codex"},
        _rate_limited_until=0,
        provider="openai-codex",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="primary",
        api_mode="codex_responses",
        reasoning_config={"enabled": True, "effort": "xhigh"},
        request_overrides={},
        _is_azure_openai_url=lambda _url: False,
        _is_direct_openai_url=lambda _url: False,
        _provider_model_requires_responses_api=lambda _model, provider=None: False,
        _anthropic_prompt_cache_policy=lambda **_kwargs: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
        _buffer_status=lambda _message: None,
        _replace_primary_openai_client=lambda **_kwargs: None,
        context_compressor=None,
        _try_activate_fallback=lambda: try_activate_fallback(agent),
        _abort_request_openai_client=lambda *_args, **_kwargs: None,
        _close_request_openai_client=lambda *_args, **_kwargs: None,
    )

    assert try_activate_fallback(agent) is True
    assert agent.provider == "minimax"
    assert agent.model == "MiniMax-M2.7"
    assert agent.reasoning_config is None


def test_runtime_fallback_invalid_effort_clears_stale_reasoning(monkeypatch):
    def fake_resolve_provider_client(provider, model, raw_codex=False, explicit_base_url=None, explicit_api_key=None):
        return SimpleNamespace(base_url="https://api.minimax.io/anthropic", api_key="fallback-key"), model

    def raising_parse(_effort):
        raise ValueError("bad effort")

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", fake_resolve_provider_client)
    monkeypatch.setattr("hermes_constants.parse_reasoning_effort", raising_parse)

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "minimax", "model": "MiniMax-M2.7", "reasoning_effort": "bogus"}],
        _fallback_index=0,
        _fallback_activated=False,
        _primary_runtime={"provider": "openai-codex"},
        _rate_limited_until=0,
        provider="openai-codex",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="primary",
        api_mode="codex_responses",
        reasoning_config={"enabled": True, "effort": "xhigh"},
        request_overrides={},
        _is_azure_openai_url=lambda _url: False,
        _is_direct_openai_url=lambda _url: False,
        _provider_model_requires_responses_api=lambda _model, provider=None: False,
        _anthropic_prompt_cache_policy=lambda **_kwargs: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
        _buffer_status=lambda _message: None,
        _replace_primary_openai_client=lambda **_kwargs: None,
        context_compressor=None,
        _try_activate_fallback=lambda: try_activate_fallback(agent),
        _abort_request_openai_client=lambda *_args, **_kwargs: None,
        _close_request_openai_client=lambda *_args, **_kwargs: None,
    )

    assert try_activate_fallback(agent) is True
    assert agent.reasoning_config is None


def test_linear_attribution_replaces_stale_footer(monkeypatch):
    args = SimpleNamespace(
        model="gpt-5.5",
        provider="openai-codex",
        reasoning_effort="xhigh",
        session_id="test-session",
    )
    monkeypatch.setattr(linear_api, "datetime", SimpleNamespace(
        now=lambda _tz: SimpleNamespace(
            replace=lambda microsecond=0: SimpleNamespace(isoformat=lambda: "2026-05-29T00:00:00+00:00")
        )
    ))

    stamped = linear_api._with_attribution("Body\nattr: a=old; m=claude-opus-4.8; src=manual-unverified", args, role="reviewer")

    assert stamped.count("attr:") == 1
    assert "m=gpt-5.5" in stamped
    assert "p=openai-codex" in stamped
    assert "r=reviewer" in stamped
    assert "claude-opus-4.8" not in stamped


def test_delegate_child_uses_builder_role_fallback_chain(monkeypatch):
    import tools.delegate_tool as delegate_tool

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._active_children = {}
            self._active_subagents = {}
            self._delegate_role = None

    fake_run_agent = ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", FakeAgent)
    monkeypatch.setitem(__import__("sys").modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 1)
    monkeypatch.setattr(delegate_tool, "_get_orchestrator_enabled", lambda: True)
    monkeypatch.setattr(delegate_tool, "_build_child_progress_callback", lambda *args, **kwargs: None)

    def fake_resolve_runtime_provider(requested, target_model=None, **_kwargs):
        return {
            "provider": requested,
            "model": target_model,
            "base_url": f"https://{requested}.example.test",
            "api_key": f"{requested}-key",
            "api_mode": "anthropic_messages" if requested in {"anthropic", "minimax"} else "codex_responses",
        }

    fake_runtime_provider = ModuleType("hermes_cli.runtime_provider")
    setattr(fake_runtime_provider, "resolve_runtime_provider", fake_resolve_runtime_provider)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.runtime_provider", fake_runtime_provider)

    parent = SimpleNamespace(
        _delegate_depth=0,
        enabled_toolsets=["terminal"],
        model="claude-sonnet-4.6",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_key="primary",
        api_mode="anthropic_messages",
        platform="cli",
        _fallback_chain=[{"provider": "anthropic", "model": "claude-opus-4.8"}],
    )

    child = delegate_tool._build_child_agent(
        task_index=0,
        goal="build",
        context=None,
        toolsets=None,
        model=None,
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
    )

    assert child is not None
    assert captured["model"] == "MiniMax-M2.7"
    assert captured["provider"] == "minimax"
    assert captured["base_url"] == "https://minimax.example.test"
    assert captured["fallback_model"] == role_fallback_chain("builder", {})
    assert captured["reasoning_config"] == {"enabled": True, "effort": "medium"}


def test_delegate_role_route_intentionally_overrides_delegation_credentials(monkeypatch):
    from tools import delegate_tool

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._active_children = {}
            self._active_subagents = {}
            self._delegate_role = None

    fake_run_agent = ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", FakeAgent)
    monkeypatch.setitem(__import__("sys").modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 1)
    monkeypatch.setattr(delegate_tool, "_get_orchestrator_enabled", lambda: True)
    monkeypatch.setattr(delegate_tool, "_build_child_progress_callback", lambda *args, **kwargs: None)

    def fake_resolve_runtime_provider(requested, target_model=None, **_kwargs):
        return {
            "provider": requested,
            "model": target_model,
            "base_url": f"https://{requested}.example.test",
            "api_key": f"{requested}-key",
            "api_mode": "anthropic_messages" if requested in {"anthropic", "minimax"} else "codex_responses",
        }

    fake_runtime_provider = ModuleType("hermes_cli.runtime_provider")
    setattr(fake_runtime_provider, "resolve_runtime_provider", fake_resolve_runtime_provider)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.runtime_provider", fake_runtime_provider)

    parent = SimpleNamespace(
        _delegate_depth=0,
        enabled_toolsets=["terminal"],
        model="claude-sonnet-4.6",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_key="primary",
        api_mode="anthropic_messages",
        platform="cli",
        _fallback_chain=[],
    )

    delegate_tool._build_child_agent(
        task_index=0,
        goal="build",
        context=None,
        toolsets=None,
        model="gpt-5.5",
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
        override_provider="openai-codex",
        override_base_url="https://chatgpt.com/backend-api/codex",
        override_api_key="codex-key",
        override_api_mode="codex_responses",
    )

    assert captured["provider"] == "minimax"
    assert captured["model"] == "MiniMax-M2.7"
    assert captured["base_url"] == "https://minimax.example.test"


def test_delegate_child_uses_approved_fallback_when_primary_role_runtime_incomplete(monkeypatch):
    from tools import delegate_tool

    captured = {}
    callback = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._active_children = {}
            self._active_subagents = {}
            self._delegate_role = None

    fake_run_agent = ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", FakeAgent)
    monkeypatch.setitem(__import__("sys").modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 1)
    monkeypatch.setattr(delegate_tool, "_get_orchestrator_enabled", lambda: True)
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_progress_callback",
        lambda *args, **kwargs: callback.update(kwargs) or None,
    )

    def resolve_runtime_provider(requested, target_model=None, **_kwargs):
        if requested == "minimax":
            return {
                "provider": requested,
                "model": target_model,
                "base_url": "https://minimax.example.test",
                "api_mode": "anthropic_messages",
            }
        return {
            "provider": requested,
            "model": target_model,
            "base_url": "https://codex.example.test",
            "api_key": "fallback-codex-key",
            "api_mode": "codex_responses",
        }

    fake_runtime_provider = ModuleType("hermes_cli.runtime_provider")
    setattr(fake_runtime_provider, "resolve_runtime_provider", resolve_runtime_provider)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.runtime_provider", fake_runtime_provider)

    parent = SimpleNamespace(
        _delegate_depth=0,
        enabled_toolsets=["terminal"],
        model="claude-sonnet-4.6",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_key="primary",
        api_mode="anthropic_messages",
        platform="cli",
        _fallback_chain=[],
    )

    child = delegate_tool._build_child_agent(
        task_index=0,
        goal="build",
        context=None,
        toolsets=None,
        model="gpt-5.5",
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
        override_provider="unapproved-provider",
        override_base_url="https://unapproved.example.test",
        override_api_key="unapproved-key",
        override_api_mode="chat_completions",
    )

    assert child is not None
    assert captured["model"] == "gpt-5.5"
    assert captured["provider"] == "openai-codex"
    assert captured["base_url"] == "https://codex.example.test"
    assert captured["api_key"] == "fallback-codex-key"
    assert captured["api_mode"] == "codex_responses"
    assert captured["fallback_model"] is None
    assert callback["model"] == "gpt-5.5"


def test_delegate_child_preserves_explicit_empty_role_fallback(monkeypatch):
    from tools import delegate_tool

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._active_children = {}
            self._active_subagents = {}
            self._delegate_role = None

    fake_run_agent = ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", FakeAgent)
    monkeypatch.setitem(__import__("sys").modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"routing_role": "adversarial_review"})
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 1)
    monkeypatch.setattr(delegate_tool, "_get_orchestrator_enabled", lambda: True)
    monkeypatch.setattr(delegate_tool, "_build_child_progress_callback", lambda *args, **kwargs: None)

    def fake_resolve_runtime_provider(requested, target_model=None, **_kwargs):
        return {
            "provider": requested,
            "model": target_model,
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-key",
            "api_mode": "codex_responses",
        }

    fake_runtime_provider = ModuleType("hermes_cli.runtime_provider")
    setattr(fake_runtime_provider, "resolve_runtime_provider", fake_resolve_runtime_provider)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.runtime_provider", fake_runtime_provider)

    parent = SimpleNamespace(
        _delegate_depth=0,
        enabled_toolsets=["terminal"],
        model="claude-sonnet-4.6",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_key="primary",
        api_mode="anthropic_messages",
        platform="cli",
        _fallback_chain=[{"provider": "minimax", "model": "MiniMax-M2.7"}],
    )

    child = delegate_tool._build_child_agent(
        task_index=0,
        goal="review",
        context=None,
        toolsets=None,
        model=None,
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
    )

    assert child is not None
    assert captured["provider"] == "openai-codex"
    assert captured["model"] == "gpt-5.5"
    assert captured["fallback_model"] == []


def test_linear_offline_receipt_queue_and_replay(tmp_path, monkeypatch):
    calls = []

    def failing_once_then_success(query, variables=None):
        calls.append((query, variables))
        if len(calls) == 1:
            raise SystemExit(1)
        return {"commentCreate": {"success": True}}

    monkeypatch.setenv("LINEAR_OFFLINE_RECEIPT_DIR", str(tmp_path))
    monkeypatch.setattr(linear_api, "gql", failing_once_then_success)

    queued = linear_api._mutation_with_receipt(
        "add-comment",
        "mutation X",
        {"input": {"issueId": "ALF-1", "body": "Body"}},
        queue_on_failure=True,
    )
    receipts = list(tmp_path.glob("*.json"))

    assert queued["offlineReceiptQueued"]["operation"] == "add-comment"
    assert len(receipts) == 1

    replay = linear_api.replay_offline_receipts(tmp_path)

    assert replay["replayed"] == 1
    assert replay["failed"] == 0
    assert replay["remaining"] == 0
    assert not list(tmp_path.glob("*.json"))
