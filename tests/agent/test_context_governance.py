"""Memory admission gate preserves the original context on bad decisions."""

from types import SimpleNamespace

from agent.context_governance import memory_context_allowed
from agent.memory_manager import MemoryManager


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
        assert manager._prefetch_provider(provider, "private query", session_id="s") == ""  # type: ignore[arg-type]
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: [{"action": "allow"}])
    assert manager._prefetch_provider(provider, "private query", session_id="s") == "private memory"  # type: ignore[arg-type]
    manager.shutdown_all()


def test_no_policy_hook_keeps_existing_context_behavior(monkeypatch):
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    import hermes_cli.config as config
    monkeypatch.setattr(plugins, "has_hook", lambda event: False)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"plugins": {"required_policy_hooks": []}})
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected call")))
    assert memory_context_allowed(provider="builtin", query="private", context="private", session_id="s")


def test_gate_errors_and_profile_scope_are_closed(monkeypatch):
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    import hermes_cli.config as config
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(plugins, "has_hook", lambda event: True)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"plugins": {"required_policy_hooks": []}})
    seen = []
    def decision(event, **payload):
        seen.append(str(get_hermes_home()))
        assert "query" not in payload and "context" not in payload
        assert set(payload["state"]).isdisjoint({"private", "raw"})
        return [{"action": "allow"}]
    monkeypatch.setattr(lifecycle, "invoke_hook", decision)
    assert memory_context_allowed(provider="external", query="private query",
                                  context="private raw context", session_id="s")
    assert seen and all(s for s in seen)
