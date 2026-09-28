"""Admission and commit gates preserve the original context on bad decisions."""

from types import SimpleNamespace
from unittest.mock import Mock
import threading
import time

import pytest

from agent.context_governance import compression_commit_allowed, memory_context_allowed
from agent.memory_manager import MemoryManager
from agent import conversation_compression as compression


def test_memory_prefetch_gate_blocks_injection_on_missing_or_invalid_decision(monkeypatch):
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    import hermes_cli.config as config
    monkeypatch.setattr(plugins, "has_hook", lambda event: event == "pre_memory_context")
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"plugins": {"required_policy_hooks": []}})
    provider = SimpleNamespace(name="builtin", prefetch=lambda query, session_id="": "private memory")
    manager = MemoryManager()
    for result in ([], [None], [{"action": "block"}], [{"action": "unexpected"}]):
        monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: result)
        assert manager._prefetch_provider(provider, "private query", session_id="s") == ""
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: [{"action": "allow"}])
    assert manager._prefetch_provider(provider, "private query", session_id="s") == "private memory"
    manager.shutdown_all()


def test_no_policy_hook_keeps_existing_context_behavior(monkeypatch):
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    import hermes_cli.config as config
    monkeypatch.setattr(plugins, "has_hook", lambda event: False)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"plugins": {"required_policy_hooks": []}})
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected call")))
    assert memory_context_allowed(provider="builtin", query="private", context="private", session_id="s")
    assert compression_commit_allowed(original=[{"content": "private"}], candidate=[{"content": "summary"}], session_id="s")


def test_gate_errors_and_profile_scope_are_closed(monkeypatch):
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    import hermes_cli.config as config
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    monkeypatch.setattr(plugins, "has_hook", lambda event: True)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"plugins": {"required_policy_hooks": []}})
    seen = []
    def decision(event, **payload):
        from hermes_constants import get_hermes_home
        seen.append(str(get_hermes_home()))
        assert "query" not in payload and "context" not in payload
        assert "original" not in payload and "candidate" not in payload
        assert set(payload["state"]).isdisjoint({"private", "raw"})
        return [{"action": "allow"}]
    monkeypatch.setattr(lifecycle, "invoke_hook", decision)
    for home in ("/tmp/jev-profile-a", "/tmp/jev-profile-b", "/tmp/jev-profile-a"):
        token = set_hermes_home_override(home)
        try:
            assert memory_context_allowed(provider="builtin", query="q", context="c", session_id="s")
        finally:
            reset_hermes_home_override(token)
    assert seen == ["/tmp/jev-profile-a", "/tmp/jev-profile-b", "/tmp/jev-profile-a"]
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()))
    assert not compression_commit_allowed(original=[], candidate=[], session_id="s")


def test_compression_gate_restores_original_before_commit(monkeypatch):
    from agent import context_governance
    monkeypatch.setattr(context_governance, "compression_commit_allowed", lambda **kw: False)
    monkeypatch.setattr(compression, "_emit_aborted_attempt_telemetry", lambda *a, **kw: None)
    original = [{"role": "user", "content": "original private context"}]
    messages = [{"role": "user", "content": "mutated summary"}]
    compressor = object()
    agent = SimpleNamespace(session_id="s", context_compressor=compressor,
                            _last_compaction_in_place=True, _emit_warning=Mock())
    attempt = SimpleNamespace(restore_compressor=Mock(), started_at=1.0)
    assert compression._govern_compression_candidate(agent, messages, original,
                                                     [{"role": "user", "content": "candidate"}], attempt)
    assert messages == original
    assert agent._last_compaction_in_place is False
    attempt.restore_compressor.assert_called_once_with(compressor)
    agent._emit_warning.assert_called_once()


@pytest.mark.parametrize("event", ["pre_memory_context", "pre_compression_commit"])
def test_hung_policy_hook_is_bounded_and_blocks(monkeypatch, event):
    from hermes_cli.plugins import PluginManager
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    import hermes_cli.config as config
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05)
    hold = threading.Event()
    manager = PluginManager()
    manager._hooks[event] = [lambda **kw: hold.wait(timeout=10)]
    monkeypatch.setattr(plugins, "has_hook", lambda name: name == event)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"plugins": {"required_policy_hooks": []}})
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda name, **kw: manager.invoke_hook(name, **kw))
    try:
        started = time.monotonic()
        if event == "pre_memory_context":
            allowed = memory_context_allowed(provider="builtin", query="q", context="private", session_id="s")
        else:
            allowed = compression_commit_allowed(original=[{"content": "private"}], candidate=[], session_id="s")
        assert not allowed
        assert time.monotonic() - started < 1
    finally:
        hold.set()


def test_required_policy_hooks_are_profile_scoped_a_b_a(tmp_path, monkeypatch):
    import hermes_cli.plugins as plugins
    import hermes_cli.lifecycle as lifecycle
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.context_governance import policy_hook_required

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "config.yaml").write_text(
        "plugins:\n  required_policy_hooks:\n    - pre_tool_call\n    - pre_memory_context\n    - pre_compression_commit\n")
    (b / "config.yaml").write_text("plugins:\n  required_policy_hooks: []\n")
    monkeypatch.setattr(plugins, "has_hook", lambda name: False)
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *a, **kw: [])
    decisions = []
    for home in (a, b, a):
        token = set_hermes_home_override(str(home))
        try:
            decisions.append((
                policy_hook_required("pre_tool_call"),
                memory_context_allowed(provider="builtin", query="query", context="memory", session_id="s"),
                compression_commit_allowed(original=[{"content": "old"}], candidate=[{"content": "new"}], session_id="s"),
                plugins._get_pre_tool_call_directive_details("read_file", {"path": "x"}).action,
            ))
        finally:
            reset_hermes_home_override(token)
    assert decisions == [(True, False, False, "block"), (False, True, True, None), (True, False, False, "block")]


def test_required_tool_hook_needs_explicit_valid_directive(monkeypatch):
    import hermes_cli.plugins as plugins
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.config as config
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"plugins": {"required_policy_hooks": ["pre_tool_call"]}})
    monkeypatch.setattr(plugins, "has_hook", lambda event: True)
    for result in ([], [None], [{"action": "modify", "args": {"x": 1}}], [{"action": "unknown"}]):
        monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: result)
        assert plugins._get_pre_tool_call_directive_details("read_file", {}).action == "block"
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: [{"action": "allow"}])
    assert plugins._get_pre_tool_call_directive_details("read_file", {}).action is None
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: [{"action": "approve"}])
    assert plugins._get_pre_tool_call_directive_details("read_file", {}).action == "approve"
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: (_ for _ in ()).throw(RuntimeError("private detail")))
    assert plugins._get_pre_tool_call_directive_details("read_file", {}).action == "block"
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: None)
    assert plugins._get_pre_tool_call_directive_details("read_file", {}).action == "block"
    monkeypatch.setattr(plugins, "has_hook", lambda event: (_ for _ in ()).throw(RuntimeError("discovery failed")))
    assert plugins._get_pre_tool_call_directive_details("read_file", {}).action == "block"


def test_required_tool_hook_timeout_blocks(monkeypatch):
    import hermes_cli.plugins as plugins
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.config as config
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"plugins": {"required_policy_hooks": ["pre_tool_call"]}})
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05)
    hold = threading.Event()
    manager = plugins.PluginManager()
    manager._hooks["pre_tool_call"] = [lambda **kw: hold.wait(timeout=10)]
    monkeypatch.setattr(plugins, "has_hook", lambda event: True)
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: manager.invoke_hook(event, **kw))
    try:
        started = time.monotonic()
        assert plugins._get_pre_tool_call_directive_details("read_file", {}).action == "block"
        assert time.monotonic() - started < 1
    finally:
        hold.set()
