"""Content-keyed turn reuse and rebuild diagnostics; regression for #144442."""

import importlib
import logging
import re
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import Platform
from gateway.run_agent_cache import GatewayAgentCacheMixin
from gateway.session import SessionSource
from gateway.session_execution import GatewaySessionAgentMixin
from gateway.turn_context import TurnContext
from tools.registry import ToolRegistry, session_tool_scope


@pytest.fixture
def registry(monkeypatch):
    reg = ToolRegistry()
    monkeypatch.setattr(importlib.import_module("tools.registry"), "registry", reg)
    monkeypatch.setattr(GatewayAgentCacheMixin, "_TOOL_SURFACE_MEMO", {}, raising=False)
    monkeypatch.setattr(GatewayAgentCacheMixin, "_MEMORY_IDENTITY_PROVIDER_MEMO", {})
    return reg


def _register(registry, **overrides):
    kwargs = dict(name="recall", toolset="memory-test", schema={"parameters": {"type": "object"}},
                  handler=lambda args: "ok")
    registry.register(**(kwargs | overrides))


def _surface():
    from gateway.run import GatewayRunner

    keys = GatewayRunner._extract_cache_busting_config({})
    assert "tools.registry_surface" in keys
    return keys["tools.registry_surface"]


def test_identical_registration_and_schema_order_keep_surface(registry):
    _register(registry, schema={"description": "Recall", "parameters": {"type": "object"}})
    original, generation = _surface(), registry._generation
    _register(registry, schema={"parameters": {"type": "object"}, "description": "Recall"})
    assert registry._generation > generation
    assert _surface() == original
    assert re.fullmatch(r"[0-9a-f]{16}", original)


def test_schema_change_removal_and_restore(registry):
    _register(registry)
    original = _surface()
    _register(registry, schema={"parameters": {"type": "object", "required": ["query"]}})
    assert _surface() != original
    _register(registry)
    assert _surface() == original
    registry.deregister("recall")
    assert _surface() != original
    _register(registry)
    assert _surface() == original


@pytest.mark.parametrize("change", [
    {"description": "Updated"}, {"requires_env": ["TEST_SECRET_REF"]}, {"is_async": True},
    {"emoji": "*"}, {"max_result_size_chars": 500}, {"execution_timeout": 5},
    {"dynamic_schema_overrides": dict}, {"toolset": "another", "override": True},
])
def test_metadata_changes_surface(registry, change):
    _register(registry)
    original = _surface()
    _register(registry, **change)
    assert _surface() != original
    registry.deregister("recall")
    _register(registry)
    assert _surface() == original


def test_swapping_one_schema_override_for_another_changes_surface(registry):
    def first():
        return {"description": "First"}

    def second():
        return {"description": "Second"}

    class Equal:
        def __init__(self, text):
            self.text = text

        def __eq__(self, other):
            return isinstance(other, Equal)

        def __hash__(self):
            return 0

        def __call__(self):
            return {"description": self.text}

    class Unweakable:
        __slots__ = ()

        def __call__(self):
            return {}

    _register(registry, dynamic_schema_overrides=first)
    original = _surface()
    _register(registry, dynamic_schema_overrides=second)
    assert _surface() != original
    _register(registry, dynamic_schema_overrides=first)
    assert _surface() == original
    # Equal is not the same: two live callables that compare equal can still return different schemas.
    a, b = Equal("A"), Equal("B")
    _register(registry, dynamic_schema_overrides=a)
    equal_first = _surface()
    _register(registry, dynamic_schema_overrides=b)
    assert _surface() != equal_first
    _register(registry, dynamic_schema_overrides=a)
    assert _surface() == equal_first
    # A callable that cannot be weakly keyed falls back to the registry generation: every change rebuilds.
    _register(registry, dynamic_schema_overrides=Unweakable())
    fallback = _surface()
    _register(registry, dynamic_schema_overrides=Unweakable())
    assert _surface() not in (original, fallback)

def test_handler_and_availability_callables_do_not_change_surface(registry):
    _register(registry)
    original = _surface()
    _register(registry, handler=lambda args: "new result", check_fn=lambda: False)
    assert _surface() == original


def test_aliases_follow_content(registry):
    _register(registry)
    original = _surface()
    registry.register_toolset_alias("remember", "memory-test")
    aliased = _surface()
    assert aliased != original
    registry.register_toolset_alias("remember", "memory-test")
    assert _surface() == aliased
    registry.register_toolset_alias("remember", "other")
    assert _surface() != aliased


def test_session_overlays_do_not_share_memo(registry):
    for slot in ("editor-session:a", "editor-session:b"):
        _register(registry, description=slot, scope=slot)
    with session_tool_scope("editor-session:a"):
        first = _surface()
    with session_tool_scope("editor-session:b"):
        second = _surface()
    assert first != second
    for slot, expected in (("editor-session:a", first), ("editor-session:b", second)):
        with session_tool_scope(slot):
            assert _surface() == expected


def test_memo_recomputes_only_for_generation_scope_or_slot(registry, monkeypatch):
    _register(registry)
    entries = Mock(wraps=registry.get_all_entries)
    monkeypatch.setattr(registry, "get_all_entries", entries)
    _surface()
    _surface()
    assert entries.call_count == 1
    _register(registry)
    _surface()
    assert entries.call_count == 2
    monkeypatch.setattr(registry, "current_scope_key", lambda: "another-scope")
    _surface()
    assert entries.call_count == 3
    with session_tool_scope("editor-session:a"):
        _surface()
        _surface()
    assert entries.call_count == 4
    _surface()
    assert entries.call_count == 4


def test_memo_stays_bounded(registry):
    for _ in range(258):
        _register(registry)
        _surface()
    assert 0 < len(GatewayAgentCacheMixin._TOOL_SURFACE_MEMO) <= 256


def test_surface_exception_falls_back_to_none(registry, monkeypatch):
    monkeypatch.setattr(registry, "get_all_entries", Mock(side_effect=RuntimeError("unavailable")))
    assert _surface() is None


@pytest.fixture
def turn():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_sig_parts = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_db = None
    runner._prefill_messages = []
    runner._service_tier = None
    runner._refresh_fallback_model = lambda: None
    runner._apply_fallback_chain_to_agent = lambda *_args: None
    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            created.append(self)

    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="test-chat", user_id="test-user"),
        session_key="test-session", session_id="conversation", user_config={}, enabled_toolsets=["memory-test"],
        AIAgent=FakeAgent,
    )
    obj = GatewaySessionAgentMixin()
    obj._runner, obj._ctx = runner, ctx
    return obj, created


def _resolve(turn, model="test-model", prompt="private prompt text"):
    obj, _ = turn
    route = {"model": model, "runtime": {"api_key": "private-credential", "base_url": "https://private.invalid"}}
    return obj._resolve_turn_agent(route, "telegram", prompt, 5, None, {})


def test_memory_provider_registration_reuses_second_turn(registry, turn, monkeypatch):
    from agent.agent_init_memory_provider import resolve_memory_provider
    from plugins import memory

    loads = []

    def load(name, **_kwargs):
        loads.append(name)
        _register(registry)
        return SimpleNamespace(identity_signature=dict, is_available=lambda: True)

    monkeypatch.setattr(memory, "load_memory_provider", load)
    obj, created = turn
    obj._ctx.user_config = {"memory": {"provider": "test-memory"}}
    base = obj._ctx.AIAgent

    class ProviderAgent(base):
        def __init__(self, **kwargs):
            resolve_memory_provider("test-memory")
            super().__init__(**kwargs)

    obj._ctx.AIAgent = ProviderAgent
    first, reused = _resolve(turn)
    assert not reused
    second, reused = _resolve(turn)
    assert reused
    assert second is first
    assert created == [first]
    assert loads == ["test-memory", "test-memory"]


@pytest.mark.parametrize("component", ["model", "toolsets", "tools.registry_surface"])
def test_signature_change_logs_names_only(registry, turn, caplog, component):
    _register(registry)
    first, _ = _resolve(turn)
    obj, created = turn
    if component == "toolsets":
        obj._ctx.enabled_toolsets = ["another"]
    elif component == "tools.registry_surface":
        _register(registry, description="changed")
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        second, reused = _resolve(turn, model="changed-model" if component == "model" else "test-model")
    assert not reused and second is not first and len(created) == 2
    records = [r for r in caplog.records if r.getMessage().startswith("Agent cache rebuild")]
    assert len(records) == 1
    assert records[0].getMessage() == f"Agent cache rebuild for session test-session: signature changed ({component})"
    assert all(raw not in records[0].getMessage() for raw in (
        "api_key", "base_url", "private-credential", "https://private.invalid", "private prompt text"))


def test_missing_previous_parts_logs_unknown_and_absent_dict_skips(registry, turn, caplog):
    _resolve(turn)
    obj, _ = turn
    obj._runner._agent_cache_sig_parts.clear()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _resolve(turn, model="second-model")
        assert "signature changed (unknown)" in caplog.text
        caplog.clear()
        del obj._runner._agent_cache_sig_parts
        _resolve(turn, model="third-model")
        assert "Agent cache rebuild" not in caplog.text


def test_message_count_mismatch_keeps_invalidation_log(registry, turn, caplog, monkeypatch):
    obj, created = turn
    monkeypatch.setattr(obj, "_current_message_count", lambda: 1)
    _resolve(turn)
    monkeypatch.setattr(obj, "_current_message_count", lambda: 2)
    monkeypatch.setattr(obj, "_release_evicted_agent", lambda _agent: None)
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _, reused = _resolve(turn)
    assert not reused and len(created) == 2
    assert "Agent cache invalidated for session test-session: message_count changed (1 -> 2)" in caplog.text
    assert "Agent cache rebuild" not in caplog.text


def test_parts_pruned_after_direct_cache_drop(registry, turn):
    obj, _ = turn
    _resolve(turn)
    obj._runner._agent_cache.clear()
    obj._ctx.session_key = "second-session"
    _resolve(turn)
    assert set(obj._runner._agent_cache_sig_parts) == {"second-session"}


def test_signature_parts_are_hashes_and_accept_non_dict_keys():
    from gateway.run_agent_cache import _agent_config_signature_parts

    args = ("test-model", {"api_key": "private-credential"}, ["memory-test"], "private prompt")
    parts = _agent_config_signature_parts(*args, cache_keys={"compression.threshold": 0.5})
    assert set(parts) == {"model", "credential", "route", "capabilities", "toolsets", "ephemeral_prompt",
                          "user_id", "user_id_alt", "skip_context_files", "compression.threshold"}
    assert all(re.fullmatch(r"[0-9a-f]{16}", value) for value in parts.values())
    assert _agent_config_signature_parts(*args, cache_keys=()) == _agent_config_signature_parts(*args)
