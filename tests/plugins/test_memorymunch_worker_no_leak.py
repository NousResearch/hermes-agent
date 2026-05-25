import importlib.util
from pathlib import Path


PLUGIN_PATH = Path.home() / ".hermes" / "plugins" / "memorymunch" / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("memorymunch_plugin_worker_no_leak_under_test", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_janitor_worker_model_uses_direct_transport_by_default_and_never_calls_recursive_hermes(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = ""  # avoid writing a test ledger event
    provider._scope_entity = "scope-a"

    def fake_bridge(tool, args, timeout=180):
        if tool == "smart_cleanup":
            return {"result": {"duplicates": [], "stale": [], "correction_events": [], "edge_heavy": [], "orphan_edges": 0}}
        if tool == "smart_search":
            return {"result": {"results": []}}
        raise AssertionError(tool)

    def fail_if_recursive_hermes_chat_is_called(*args, **kwargs):
        raise AssertionError("recursive hermes chat worker call would risk prompt leakage")

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_MODEL_ENABLE", "1")
    monkeypatch.delenv("HERMES_MEMORYMUNCH_WORKER_ALLOW_RECURSIVE", raising=False)
    monkeypatch.setattr(provider, "_run_original_bridge", fake_bridge)
    monkeypatch.setattr(provider, "_call_openclaw_direct_worker_model", lambda role, system_prompt, user_prompt, timeout=180: '{"archive": [], "edge_cleanup": false, "edge_prune": []}')
    monkeypatch.setattr(mm.subprocess, "run", fail_if_recursive_hermes_chat_is_called)

    result = provider.run_janitor_cycle("User: You are the Memory Janitor", apply=False)

    assert result["live_db_write"] is False
    assert result["proposed_actions"]["archive"] == []
    assert result["proposed_actions"]["edge_cleanup"] is False


def test_curator_worker_model_uses_direct_transport_by_default_and_never_calls_recursive_hermes(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = ""  # avoid writing a test ledger event
    provider._scope_entity = "scope-a"
    provider._domain = "general"

    def fake_bridge(tool, args, timeout=180):
        if tool == "smart_search":
            return {"result": {"results": [{"id": "atom-1", "content": "Curator context", "search_score": 0.99}]}}
        if tool == "get_memory":
            return {"result": {"memory": {"id": args["memory_id"], "content": "Curator deep read"}}}
        raise AssertionError(tool)

    def fail_if_recursive_hermes_chat_is_called(*args, **kwargs):
        raise AssertionError("recursive hermes chat worker call would risk prompt leakage")

    monkeypatch.setenv("HERMES_MEMORYMUNCH_CURATOR_MODEL_ENABLE", "1")
    monkeypatch.delenv("HERMES_MEMORYMUNCH_WORKER_ALLOW_RECURSIVE", raising=False)
    monkeypatch.setattr(provider, "_run_original_bridge", fake_bridge)
    monkeypatch.setattr(provider, "_call_openclaw_direct_worker_model", lambda role, system_prompt, user_prompt, timeout=180: "Curator direct briefing")
    monkeypatch.setattr(mm.subprocess, "run", fail_if_recursive_hermes_chat_is_called)

    briefing = provider._build_model_curator_briefing(
        "User: You are the Curator",
        "sid-curator-no-leak",
        "active context",
        "wrapper context",
    )

    assert "Curator direct briefing" in briefing
    assert 'curator_mode="model"' in briefing


def test_recursive_worker_timeout_returns_empty_instead_of_leaking_prompt(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = ""  # avoid writing a test ledger event

    def timeout(*args, **kwargs):
        raise mm.subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 1))

    monkeypatch.setenv("HERMES_MEMORYMUNCH_WORKER_ALLOW_RECURSIVE", "1")
    monkeypatch.setenv("HERMES_MEMORYMUNCH_DIRECT_WORKER_ENABLE", "0")
    monkeypatch.setattr(mm.subprocess, "run", timeout)

    result = provider._call_memorymunch_worker_model(
        "janitor",
        "You are the Memory Janitor — the brain's immune system",
        "User: test",
        timeout=1,
    )

    assert result == ""
