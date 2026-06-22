"""Status-only external adapter tests for Memory v2 evals."""

from __future__ import annotations

import socket

from plugins.memory.memory_v2.evals.adapters import external_adapter_status


EXPECTED_ADAPTERS = {"mem0", "zep_graphiti", "letta", "langmem", "cognee", "supermemory"}


def test_external_adapter_status_reports_expected_adapters():
    status = external_adapter_status()

    assert set(status) == EXPECTED_ADAPTERS
    for name, payload in status.items():
        assert payload["module"], name
        assert isinstance(payload["available"], bool), name
        assert isinstance(payload["module_available"], bool), name
        assert "required_env" in payload, name
        assert isinstance(payload["env_available"], bool), name


def test_external_adapter_status_is_safe_without_dependencies(monkeypatch):
    def fail_network_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("external_adapter_status must not perform network calls")

    monkeypatch.setattr(socket, "create_connection", fail_network_call)
    monkeypatch.setattr(socket, "socket", fail_network_call)

    status = external_adapter_status()

    assert EXPECTED_ADAPTERS.issubset(status)
    assert all("available" in payload for payload in status.values())
    assert all("module_available" in payload for payload in status.values())
    assert all("env_available" in payload for payload in status.values())


def test_external_adapter_status_is_readiness_only_when_modules_and_env_are_missing(monkeypatch):
    from plugins.memory.memory_v2.evals import adapters

    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda _module: None)
    monkeypatch.setattr(adapters.os, "getenv", lambda _env: None)

    status = external_adapter_status()

    assert status["mem0"] == {
        "available": False,
        "module": "mem0",
        "module_available": False,
        "required_env": "MEM0_API_KEY",
        "env_available": False,
    }
    assert status["zep_graphiti"]["required_env"] == "ZEP_API_KEY"
    assert status["letta"]["required_env"] == "LETTA_API_KEY"
    assert status["supermemory"]["required_env"] == "SUPERMEMORY_API_KEY"
    assert status["langmem"]["env_available"] is True
    assert status["cognee"]["env_available"] is True
