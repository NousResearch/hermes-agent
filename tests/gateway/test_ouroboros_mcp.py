import json
import math
from types import SimpleNamespace

import pytest

import gateway.ouroboros_mcp as ouroboros_mcp


class FakeRegistry:
    def __init__(self, entries=None):
        self.entries = entries or {}
        self.lookups = []

    def get_entry(self, name):
        self.lookups.append(name)
        return self.entries.get(name)


def _entry(*, toolset="mcp-ouroboros", check_fn=None):
    return SimpleNamespace(toolset=toolset, check_fn=check_fn)


def _install_registry(monkeypatch, entry):
    registry_name = ouroboros_mcp.ouroboros_registry_tool_name("ouroboros_interview")
    fake_registry = FakeRegistry({registry_name: entry} if entry is not None else {})
    monkeypatch.setattr(ouroboros_mcp, "registry", fake_registry)
    return registry_name, fake_registry


def _install_registry_entries(monkeypatch, entries):
    fake_registry = FakeRegistry(entries)
    monkeypatch.setattr(ouroboros_mcp, "registry", fake_registry)
    return fake_registry


def _install_handler_factory(monkeypatch, *, response='{"success": true}', raises=None):
    calls = []

    def fake_factory(server_name, raw_tool_name, timeout):
        calls.append(("factory", server_name, raw_tool_name, timeout))

        def fake_handler(args):
            calls.append(("handler", args))
            if raises is not None:
                raise raises
            return response

        return fake_handler

    monkeypatch.setattr(ouroboros_mcp, "_handler_factory", fake_factory)
    return calls


def test_registry_name_mapping_for_ouroboros_interview():
    assert (
        ouroboros_mcp.ouroboros_registry_tool_name("ouroboros_interview")
        == "mcp_ouroboros_ouroboros_interview"
    )


def test_success_path_calls_registry_gated_handler(monkeypatch):
    registry_name, fake_registry = _install_registry(monkeypatch, _entry())
    calls = _install_handler_factory(monkeypatch, response=json.dumps({"result": "ok"}))

    result = ouroboros_mcp.call_ouroboros_tool(
        "ouroboros_interview", {"initial_context": "hello"}, timeout=12.5
    )

    assert fake_registry.lookups == [registry_name]
    assert calls == [
        ("factory", "ouroboros", "ouroboros_interview", 12.5),
        ("handler", {"initial_context": "hello"}),
    ]
    assert result == {"result": "ok", "success": True}


def test_missing_registry_entry_returns_error_without_calling_handler(monkeypatch):
    registry_name, fake_registry = _install_registry(monkeypatch, None)
    monkeypatch.setattr(ouroboros_mcp, "_discover_mcp_tools", lambda: None)
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})

    assert fake_registry.lookups == [registry_name, registry_name]
    assert calls == []
    assert result["success"] is False
    assert result["tool"] == "ouroboros_interview"
    assert result["registry_name"] == registry_name
    assert "registered" in result["error"] or "available" in result["error"]


def test_missing_registry_entry_lazily_discovers_and_retries(monkeypatch):
    registry_name, fake_registry = _install_registry(monkeypatch, None)
    calls = _install_handler_factory(monkeypatch, response=json.dumps({"result": "ok"}))

    def fake_discover():
        fake_registry.entries[registry_name] = _entry()

    monkeypatch.setattr(ouroboros_mcp, "_discover_mcp_tools", fake_discover)

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})

    assert fake_registry.lookups == [registry_name, registry_name]
    assert calls == [
        ("factory", "ouroboros", "ouroboros_interview", 45.0),
        ("handler", {}),
    ]
    assert result == {"result": "ok", "success": True}


def test_lazy_discovery_failure_returns_safe_error_without_handler(monkeypatch):
    _registry_name, _fake_registry = _install_registry(monkeypatch, None)
    calls = _install_handler_factory(monkeypatch)

    def fake_discover():
        raise RuntimeError("discovery boom")

    monkeypatch.setattr(ouroboros_mcp, "_discover_mcp_tools", fake_discover)

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})

    assert calls == []
    assert result["success"] is False
    assert "discovery" in result["error"].lower()
    assert "RuntimeError" in result["error"]


def test_wrong_toolset_returns_error_without_calling_handler(monkeypatch):
    registry_name, _fake_registry = _install_registry(monkeypatch, _entry(toolset="mcp-other"))
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})

    assert calls == []
    assert result["success"] is False
    assert result["registry_name"] == registry_name
    assert "toolset" in result["error"]
    assert "mcp-ouroboros" in result["error"]


def test_check_fn_false_returns_unavailable_error_without_calling_handler(monkeypatch):
    _registry_name, _fake_registry = _install_registry(
        monkeypatch, _entry(check_fn=lambda: False)
    )
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})

    assert calls == []
    assert result["success"] is False
    assert "available" in result["error"] or "connected" in result["error"]


def test_non_dict_args_returns_validation_error_without_handler(monkeypatch):
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", ["bad"])

    assert calls == []
    assert result["success"] is False
    assert "args" in result["error"]
    assert "dict" in result["error"]


@pytest.mark.parametrize("bad_timeout", [0, -1, math.inf, -math.inf, math.nan])
def test_invalid_timeout_returns_validation_error_without_handler(monkeypatch, bad_timeout):
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool(
        "ouroboros_interview", {}, timeout=bad_timeout
    )

    assert calls == []
    assert result["success"] is False
    assert "timeout" in result["error"]


def test_incorrectly_prefixed_tool_name_is_rejected_without_handler(monkeypatch):
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool(
        "mcp_ouroboros_ouroboros_interview", {}
    )

    assert calls == []
    assert result["success"] is False
    assert "raw" in result["error"] or "prefix" in result["error"]


def test_sanitized_name_collision_raw_tool_name_is_rejected_without_handler(monkeypatch):
    registry_name = "mcp_ouroboros_foo_bar"
    fake_registry = _install_registry_entries(monkeypatch, {registry_name: _entry()})
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool("foo-bar", {})

    assert fake_registry.lookups == []
    assert calls == []
    assert result["success"] is False
    assert "tool" in result["error"]


def test_non_ouroboros_raw_tool_name_is_rejected_without_handler(monkeypatch):
    registry_name = "mcp_ouroboros_list_prompts"
    fake_registry = _install_registry_entries(monkeypatch, {registry_name: _entry()})
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool("list_prompts", {})

    assert fake_registry.lookups == []
    assert calls == []
    assert result["success"] is False
    assert "ouroboros_" in result["error"]


def test_invalid_tool_name_error_does_not_echo_sensitive_object(monkeypatch):
    class SecretToolName:
        def __str__(self):
            return "bad opaque-secret-value"

    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool(
        SecretToolName(),  # pyright: ignore[reportArgumentType]
        {},
    )

    assert calls == []
    assert result["success"] is False
    assert isinstance(result["tool"], str)
    assert "opaque-secret-value" not in result["tool"]


def test_handler_non_json_text_is_wrapped_as_success_result(monkeypatch):
    _registry_name, _fake_registry = _install_registry(monkeypatch, _entry())
    _calls = _install_handler_factory(monkeypatch, response="plain text")

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})

    assert result == {"success": True, "result": "plain text"}


def test_handler_exception_returns_safe_error_without_traceback(monkeypatch):
    _registry_name, _fake_registry = _install_registry(monkeypatch, _entry())
    raw_secret = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    _calls = _install_handler_factory(
        monkeypatch,
        raises=RuntimeError(f"boom {raw_secret}"),
    )

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})
    serialized = json.dumps(result)

    assert result["success"] is False
    assert "RuntimeError" in result["error"]
    assert "Traceback" not in serialized
    assert raw_secret not in serialized


def test_check_fn_exception_returns_safe_error_without_calling_handler(monkeypatch):
    def unavailable():
        raise RuntimeError("availability boom")

    _registry_name, _fake_registry = _install_registry(
        monkeypatch, _entry(check_fn=unavailable)
    )
    calls = _install_handler_factory(monkeypatch)

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})
    serialized = json.dumps(result)

    assert calls == []
    assert result["success"] is False
    assert "RuntimeError" in result["error"]
    assert "availability boom" in result["error"]
    assert "Traceback" not in serialized


def test_json_error_payload_strings_are_redacted(monkeypatch):
    _registry_name, _fake_registry = _install_registry(monkeypatch, _entry())
    raw_secret = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    _calls = _install_handler_factory(
        monkeypatch,
        response=json.dumps({"error": f"failed with token {raw_secret}"}),
    )

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})
    serialized = json.dumps(result)

    assert result["success"] is False
    assert raw_secret not in serialized
    assert "failed with token" in result["error"]


def test_json_payload_sensitive_keys_are_structurally_redacted(monkeypatch):
    _registry_name, _fake_registry = _install_registry(monkeypatch, _entry())
    _calls = _install_handler_factory(
        monkeypatch,
        response=json.dumps(
            {
                "token": "opaque-secret-value",
                "nested": {
                    "api_key": "plainvalue123",
                    "visible": "keep-me",
                    "items": [
                        {
                            "authorization": "Bearer plainvalue456",
                            "name": "public-name",
                        }
                    ],
                },
                "credentials": {"username": "alice", "password": "hunter2"},
            }
        ),
    )

    result = ouroboros_mcp.call_ouroboros_tool("ouroboros_interview", {})

    assert result["success"] is True
    assert result["token"] == "[REDACTED]"
    assert result["nested"]["api_key"] == "[REDACTED]"
    assert result["nested"]["visible"] == "keep-me"
    assert result["nested"]["items"][0]["authorization"] == "[REDACTED]"
    assert result["nested"]["items"][0]["name"] == "public-name"
    assert result["credentials"] == "[REDACTED]"
