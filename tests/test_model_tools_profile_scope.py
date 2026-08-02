"""Profile-isolation regressions for model tool schema and fallback caches."""

from __future__ import annotations

import contextlib
import contextvars
import sys
import types

import pytest

import model_tools


@pytest.fixture
def profile_scope(monkeypatch):
    current = contextvars.ContextVar("test_model_profile", default="launch")
    module = types.ModuleType("agent.plugin_profile_scope")

    def canonical(key=None):
        return str(current.get() if key is None else key)

    def current_profile_key():
        return canonical()

    def freeze_profile_key(profile_key=None):
        return canonical(profile_key)

    @contextlib.contextmanager
    def bind_profile_key(profile_key):
        key = canonical(profile_key)
        token = current.set(key)
        try:
            yield key
        finally:
            current.reset(token)

    for name, value in {
        "current_profile_key": current_profile_key,
        "freeze_profile_key": freeze_profile_key,
        "bind_profile_key": bind_profile_key,
    }.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "agent.plugin_profile_scope", module)
    model_tools._clear_tool_defs_cache(all_profiles=True)
    yield module
    model_tools._clear_tool_defs_cache(all_profiles=True)


class _ProfileRegistry:
    def __init__(self, scope):
        self.scope = scope
        self.generations = {"alpha": 3, "beta": 3}
        self.calls = {"alpha": 0, "beta": 0}
        self.bump_during_first_call = False
        self.bump_during_every_call = False
        self.dispatched = []

    @property
    def _generation(self):
        return self.generations.get(self.scope.current_profile_key(), 0)

    def generation(self, profile_key=None):
        key = self.scope.freeze_profile_key(profile_key)
        return self.generations.get(key, 0)

    def get_definitions(self, _names, quiet=False):
        key = self.scope.current_profile_key()
        self.calls[key] += 1
        marker = f"{key}-{self.generations[key]}"
        if self.bump_during_every_call or (
            self.bump_during_first_call and self.calls[key] == 1
        ):
            self.generations[key] += 1
        return [{
            "type": "function",
            "function": {
                "name": f"tool_{marker}",
                "description": marker,
                "parameters": {"type": "object", "properties": {}},
            },
        }]

    def get_schema(self, _name):
        return None

    def dispatch(self, name, args, **kwargs):
        self.dispatched.append((self.scope.current_profile_key(), name, args, kwargs))
        return '{"ok": true}'


def _patch_minimal_schema_path(monkeypatch, registry):
    monkeypatch.setattr(model_tools, "registry", registry)
    monkeypatch.setattr(model_tools, "validate_toolset", lambda _name: True)
    monkeypatch.setattr(model_tools, "resolve_toolset", lambda _name: ["profile_tool"])
    monkeypatch.setattr(model_tools, "_is_delegated_child_context", lambda: False)
    monkeypatch.setattr("tools.tool_search.load_config", lambda: types.SimpleNamespace(enabled="off"))


def test_schema_cache_key_includes_profile_identity_and_profile_generation(profile_scope, monkeypatch):
    reg = _ProfileRegistry(profile_scope)
    _patch_minimal_schema_path(monkeypatch, reg)

    with profile_scope.bind_profile_key("alpha"):
        first = model_tools.get_tool_definitions(["profile"], quiet_mode=True)
        again = model_tools.get_tool_definitions(["profile"], quiet_mode=True)
    with profile_scope.bind_profile_key("beta"):
        other = model_tools.get_tool_definitions(["profile"], quiet_mode=True)

    assert first == again
    assert first[0]["function"]["description"] == "alpha-3"
    assert other[0]["function"]["description"] == "beta-3"
    assert reg.calls == {"alpha": 1, "beta": 1}


def test_schema_cache_does_not_publish_result_computed_against_stale_generation(profile_scope, monkeypatch):
    reg = _ProfileRegistry(profile_scope)
    reg.bump_during_first_call = True
    _patch_minimal_schema_path(monkeypatch, reg)

    with profile_scope.bind_profile_key("alpha"):
        result = model_tools.get_tool_definitions(["profile"], quiet_mode=True)
        cached = model_tools.get_tool_definitions(["profile"], quiet_mode=True)

    assert result == cached
    assert result[0]["function"]["description"] == "alpha-4"
    assert reg.calls["alpha"] == 2


def test_schema_assembly_fails_closed_during_continuous_generation_churn(profile_scope, monkeypatch):
    reg = _ProfileRegistry(profile_scope)
    reg.bump_during_every_call = True
    _patch_minimal_schema_path(monkeypatch, reg)
    monkeypatch.setattr(
        model_tools,
        "_last_resolved_tool_names_by_profile",
        {"beta": ["beta_stable"]},
    )

    with profile_scope.bind_profile_key("alpha"):
        with pytest.raises(RuntimeError, match="registry changed during schema assembly"):
            model_tools.get_tool_definitions(["profile"], quiet_mode=True)
        model_tools.handle_function_call(
            "execute_code",
            {"code": "pass"},
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    assert reg.calls["alpha"] == 3
    assert not any(key[0] == "alpha" for key in model_tools._tool_defs_cache)
    assert model_tools._last_resolved_tool_names_by_profile == {"beta": ["beta_stable"]}
    assert reg.dispatched[-1][3]["enabled_tools"] == []


def test_continuous_churn_preserves_only_each_profiles_last_stable_fallback(
    profile_scope, monkeypatch
):
    reg = _ProfileRegistry(profile_scope)
    _patch_minimal_schema_path(monkeypatch, reg)

    with profile_scope.bind_profile_key("alpha"):
        model_tools.get_tool_definitions(["profile"], quiet_mode=True)
    with profile_scope.bind_profile_key("beta"):
        model_tools.get_tool_definitions(["profile"], quiet_mode=True)

    reg.generations["alpha"] += 1
    reg.bump_during_every_call = True
    with profile_scope.bind_profile_key("alpha"):
        with pytest.raises(RuntimeError, match="registry changed during schema assembly"):
            model_tools.get_tool_definitions(["profile"], quiet_mode=True)
        model_tools.handle_function_call(
            "execute_code",
            {"code": "pass"},
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    assert model_tools._last_resolved_tool_names_by_profile == {
        "alpha": ["tool_alpha-3"],
        "beta": ["tool_beta-3"],
    }
    assert reg.dispatched[-1][3]["enabled_tools"] == ["tool_alpha-3"]


def test_execute_code_compatibility_fallback_is_profile_scoped(profile_scope, monkeypatch):
    reg = _ProfileRegistry(profile_scope)
    monkeypatch.setattr(model_tools, "registry", reg)
    monkeypatch.setattr(model_tools, "_last_resolved_tool_names", ["beta_tool"])
    monkeypatch.setattr(
        model_tools,
        "_last_resolved_tool_names_by_profile",
        {"alpha": ["alpha_tool"], "beta": ["beta_tool"]},
        raising=False,
    )

    with profile_scope.bind_profile_key("alpha"):
        model_tools.handle_function_call(
            "execute_code",
            {"code": "pass"},
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    assert reg.dispatched[-1][0] == "alpha"
    assert reg.dispatched[-1][3]["enabled_tools"] == ["alpha_tool"]


def test_explicit_enabled_tools_remain_frozen_after_other_profile_reload(profile_scope, monkeypatch):
    reg = _ProfileRegistry(profile_scope)
    monkeypatch.setattr(model_tools, "registry", reg)
    explicit = ["alpha_tool"]

    with profile_scope.bind_profile_key("beta"):
        reg.generations["beta"] += 1
    with profile_scope.bind_profile_key("alpha"):
        model_tools.handle_function_call(
            "execute_code",
            {"code": "pass"},
            enabled_tools=explicit,
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    assert explicit == ["alpha_tool"]
    assert reg.dispatched[-1][3]["enabled_tools"] is explicit
