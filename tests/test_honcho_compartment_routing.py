"""Honcho named-compartment routing tests.

These tests keep ops/personal isolation at the provider boundary: selecting a
compartment must choose a different manager/session, not just pass a tag to the
same workspace.
"""

import json

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig, HonchoCompartmentConfig


class _FakeManager:
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def get_or_create(self, session_key):
        self.calls.append(("get_or_create", session_key))
        return object()

    def search_context(self, session_key, query, max_tokens=800, peer="user"):
        self.calls.append(("search_context", session_key, query, max_tokens, peer))
        return f"{self.name}:{session_key}:{query}:{peer}"

    def create_conclusion(self, session_key, conclusion, peer="user"):
        self.calls.append(("create_conclusion", session_key, conclusion, peer))
        return True


def _provider_with_ready_managers():
    provider = HonchoMemoryProvider()
    provider._session_initialized = True
    provider._session_key = "primary-session"
    provider._manager = _FakeManager("primary")
    provider._compartment_session_keys = {"ops": "ops-session"}
    provider._compartment_managers = {"ops": _FakeManager("ops")}
    provider._config = HonchoClientConfig(
        workspace_id="personal-prod",
        api_key="personal-key",
        base_url="https://honcho.example.test",
        compartments={
            "ops": HonchoCompartmentConfig(
                name="ops",
                workspace_id="ops-prod",
                api_key="ops-key",
                base_url="https://honcho.example.test",
            )
        },
    )
    return provider


def test_honcho_tool_schema_exposes_compartment_selector():
    provider = HonchoMemoryProvider()
    schemas = {schema["name"]: schema for schema in provider.get_tool_schemas()}

    for tool_name in [
        "honcho_profile",
        "honcho_search",
        "honcho_reasoning",
        "honcho_context",
        "honcho_conclude",
    ]:
        assert "compartment" in schemas[tool_name]["parameters"]["properties"]


def test_honcho_search_routes_to_requested_ready_compartment_manager():
    provider = _provider_with_ready_managers()

    result = json.loads(
        provider.handle_tool_call(
            "honcho_search",
            {"query": "ports", "peer": "user", "compartment": "ops"},
        )
    )

    assert result == {"result": "ops:ops-session:ports:user"}
    assert provider._manager.calls == []
    assert provider._compartment_managers["ops"].calls == [
        ("search_context", "ops-session", "ports", 800, "user")
    ]


def test_honcho_conclude_routes_write_to_requested_ready_compartment_manager():
    provider = _provider_with_ready_managers()

    result = json.loads(
        provider.handle_tool_call(
            "honcho_conclude",
            {"conclusion": "ops memory belongs in ops", "compartment": "ops"},
        )
    )

    assert result == {"result": "Conclusion saved for user: ops memory belongs in ops"}
    assert provider._manager.calls == []
    assert provider._compartment_managers["ops"].calls == [
        ("create_conclusion", "ops-session", "ops memory belongs in ops", "user")
    ]


def test_unknown_honcho_compartment_returns_tool_error_without_primary_fallback():
    provider = _provider_with_ready_managers()

    result = json.loads(
        provider.handle_tool_call(
            "honcho_search",
            {"query": "family", "compartment": "personal-but-missing"},
        )
    )

    assert result["error"] == "Unknown Honcho compartment: personal-but-missing"
    assert provider._manager.calls == []
    assert provider._compartment_managers["ops"].calls == []


def test_honcho_tool_lazily_initializes_configured_compartment_manager(monkeypatch):
    provider = HonchoMemoryProvider()
    provider._session_initialized = True
    provider._session_key = "primary-session"
    provider._lazy_init_session_id = "session-123"
    provider._lazy_init_kwargs = {"session_title": "Compartment Test"}
    provider._manager = _FakeManager("primary")
    provider._config = HonchoClientConfig(
        workspace_id="personal-prod",
        api_key="personal-key",
        base_url="https://honcho.example.test",
        compartments={
            "ops": HonchoCompartmentConfig(
                name="ops",
                workspace_id="ops-prod",
                api_key="ops-key",
                base_url="https://honcho.example.test",
            )
        },
    )

    built = []

    def fake_get_honcho_client(cfg):
        built.append((cfg.host, cfg.workspace_id, cfg.api_key))
        return object()

    class FakeSessionManager(_FakeManager):
        def __init__(self, **kwargs):
            super().__init__(kwargs["config"].workspace_id)
            self.kwargs = kwargs

    monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", fake_get_honcho_client)
    monkeypatch.setattr("plugins.memory.honcho.session.HonchoSessionManager", FakeSessionManager)

    result = json.loads(
        provider.handle_tool_call(
            "honcho_search",
            {"query": "routing", "compartment": "ops"},
        )
    )

    assert result == {"result": "ops-prod:Compartment-Test:routing:user"}
    assert built == [("hermes:ops", "ops-prod", "ops-key")]
    assert provider._compartment_managers["ops"].calls == [
        ("get_or_create", "Compartment-Test"),
        ("search_context", "Compartment-Test", "routing", 800, "user"),
    ]
