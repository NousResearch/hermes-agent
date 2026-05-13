"""Tests for the Open Brain memory plugin."""

import json
import threading
from plugins.memory.openbrain import (
    OpenBrainMemoryProvider,
    _safe_to_capture,
)


def test_provider_identity_and_schemas(monkeypatch):
    monkeypatch.setattr("plugins.memory.openbrain._openbrain_mcp_configured", lambda: True)

    provider = OpenBrainMemoryProvider()

    assert provider.name == "openbrain"
    assert provider.is_available() is True
    assert [schema["name"] for schema in provider.get_tool_schemas()] == [
        "openbrain_search",
        "openbrain_capture",
        "openbrain_recent",
        "openbrain_stats",
    ]
    capture_schema = provider.get_tool_schemas()[1]
    assert capture_schema["parameters"]["properties"]["use_policy"]["enum"] == ["evidence", "reference"]
    assert provider.get_config_schema() == []
    assert "Open Brain Memory" in provider.system_prompt_block()
    assert "Never store raw transcripts" in provider.system_prompt_block()


def test_safe_to_capture_rejects_short_long_and_secretish_content():
    assert _safe_to_capture("too short") == (False, "too short")
    ok, reason = _safe_to_capture("x" * 6001)
    assert ok is False
    assert reason == "too long"
    ok, reason = _safe_to_capture("Remember this API_KEY should never be copied into memory")
    assert ok is False
    assert "secret" in reason
    assert _safe_to_capture("A durable project decision with enough safe context.")[0] is True


def test_handle_search_routes_to_openbrain_mcp(monkeypatch):
    provider = OpenBrainMemoryProvider()
    provider.initialize("sess-1", platform="telegram", agent_identity="marshall")

    calls = []

    def fake_dispatch(name, args):
        calls.append((name, args))
        return json.dumps({"result": json.dumps({"thoughts": [{"content": "Known fact"}]})})

    monkeypatch.setattr("plugins.memory.openbrain._ensure_mcp_tools", lambda: None)
    monkeypatch.setattr("plugins.memory.openbrain.registry.get_entry", lambda name: object())
    monkeypatch.setattr("plugins.memory.openbrain.registry.dispatch", fake_dispatch)

    raw = provider.handle_tool_call("openbrain_search", {"query": "known", "limit": 3})
    parsed = json.loads(raw)

    assert parsed == {"thoughts": [{"content": "Known fact"}]}
    assert calls == [("mcp_open_brain_search_thoughts", {"query": "known", "limit": 3, "threshold": 0.5})]


def test_capture_stamps_policy_and_refuses_secretish_content(monkeypatch):
    provider = OpenBrainMemoryProvider()
    provider.initialize("sess-42", platform="telegram", agent_identity="marshall")

    calls = []

    def fake_dispatch(name, args):
        calls.append((name, args))
        return json.dumps({"result": json.dumps({"id": "thought-1"})})

    monkeypatch.setattr("plugins.memory.openbrain._ensure_mcp_tools", lambda: None)
    monkeypatch.setattr("plugins.memory.openbrain.registry.get_entry", lambda name: object())
    monkeypatch.setattr("plugins.memory.openbrain.registry.dispatch", fake_dispatch)

    refused = provider.handle_tool_call("openbrain_capture", {"content": "password should not be saved here"})
    assert "Refusing to capture" in refused
    assert calls == []

    refused_instruction = provider.handle_tool_call(
        "openbrain_capture",
        {
            "content": "A human has not approved this instruction-grade memory.",
            "use_policy": "instruction",
        },
    )
    assert "instruction-grade" in refused_instruction
    assert calls == []

    raw = provider.handle_tool_call(
        "openbrain_capture",
        {
            "content": "The openbrain provider is active for Hermes agent memory.",
            "scope": "workspace",
            "use_policy": "evidence",
        },
    )

    assert json.loads(raw) == {"id": "thought-1"}
    assert calls[0][0] == "mcp_open_brain_capture_thought"
    stamped = calls[0][1]["content"]
    assert "Hermes agent memory" in stamped
    assert "scope=workspace" in stamped
    assert "use_policy=evidence" in stamped
    assert "session=sess-42" in stamped
    assert "The openbrain provider is active" in stamped


def test_prefetch_formats_search_results(monkeypatch):
    provider = OpenBrainMemoryProvider()

    monkeypatch.setattr("plugins.memory.openbrain._openbrain_mcp_configured", lambda: True)
    monkeypatch.setattr("plugins.memory.openbrain._ensure_mcp_tools", lambda: None)
    monkeypatch.setattr("plugins.memory.openbrain.registry.get_entry", lambda name: object())
    monkeypatch.setattr(
        "plugins.memory.openbrain.registry.dispatch",
        lambda name, args: json.dumps(
            {
                "result": json.dumps(
                    {
                        "thoughts": [
                            {
                                "content": "Prior durable decision.",
                                "similarity": 0.8123,
                                "metadata": {"type": "observation", "topics": ["hermes", "memory"]},
                            }
                        ]
                    }
                )
            }
        ),
    )

    provider.queue_prefetch("Hermes memory")
    result = provider.prefetch("Hermes memory")

    assert "Open Brain Recall" in result
    assert "Prior durable decision" in result
    assert "score=0.81" in result
    assert "type=observation" in result


def test_session_switch_updates_provenance_and_clears_prefetch(monkeypatch):
    provider = OpenBrainMemoryProvider()
    provider.initialize("old-session", platform="telegram", agent_identity="marshall")
    provider._prefetch_result = "stale recall"

    calls = []

    def fake_dispatch(name, args):
        calls.append((name, args))
        return json.dumps({"result": json.dumps({"id": "thought-2"})})

    monkeypatch.setattr("plugins.memory.openbrain._ensure_mcp_tools", lambda: None)
    monkeypatch.setattr("plugins.memory.openbrain.registry.get_entry", lambda name: object())
    monkeypatch.setattr("plugins.memory.openbrain.registry.dispatch", fake_dispatch)

    provider.on_session_switch("new-session", parent_session_id="old-session", reset=True)
    assert provider.prefetch("anything") == ""

    provider.handle_tool_call(
        "openbrain_capture",
        {"content": "The session switch updates Open Brain memory provenance."},
    )

    assert "session=new-session" in calls[0][1]["content"]


def test_session_switch_invalidates_inflight_prefetch(monkeypatch):
    provider = OpenBrainMemoryProvider()
    release_dispatch = threading.Event()

    monkeypatch.setattr("plugins.memory.openbrain._openbrain_mcp_configured", lambda: True)
    monkeypatch.setattr("plugins.memory.openbrain._ensure_mcp_tools", lambda: None)
    monkeypatch.setattr("plugins.memory.openbrain.registry.get_entry", lambda name: object())

    def slow_dispatch(name, args):
        release_dispatch.wait(timeout=2)
        return json.dumps({"result": json.dumps({"thoughts": [{"content": "old-query recall"}]})})

    monkeypatch.setattr("plugins.memory.openbrain.registry.dispatch", slow_dispatch)

    provider.queue_prefetch("old query")
    provider.on_session_switch("new-session", parent_session_id="old-session", reset=True)
    release_dispatch.set()

    assert provider.prefetch("new query") == ""


def test_post_setup_activates_provider(monkeypatch, tmp_path):
    provider = OpenBrainMemoryProvider()
    saved = []

    monkeypatch.setattr("plugins.memory.openbrain._openbrain_mcp_configured", lambda: True)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda config: saved.append(config.copy()))

    config = {"memory": {"provider": ""}}
    provider.post_setup(str(tmp_path), config)

    assert config["memory"]["provider"] == "openbrain"
    assert saved[-1]["memory"]["provider"] == "openbrain"


def test_discovery_loads_openbrain_provider():
    from plugins.memory import discover_memory_providers, load_memory_provider

    providers = discover_memory_providers()
    names = {name for name, _desc, _available in providers}
    assert "openbrain" in names

    provider = load_memory_provider("openbrain")
    assert isinstance(provider, OpenBrainMemoryProvider)
