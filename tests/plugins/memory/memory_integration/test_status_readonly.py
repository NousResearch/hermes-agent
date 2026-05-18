from __future__ import annotations

import json
import sys


def test_status_tool_returns_bounded_json_before_initialize(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    raw = provider.handle_tool_call(
        "memory_integration_status",
        {},
        config={"memory_integration": {"vault": {"mode": "shared"}}},
        hermes_home=tmp_path,
    )
    status = json.loads(raw)

    assert isinstance(raw, str)
    assert len(raw) < 4096
    assert status["provider"] == "memory-integration"
    assert status["ok"] is False
    assert status["initialized"] is False
    assert "not_initialized" in status["diagnostics"]


def test_status_does_not_create_sidecar_directory(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()
    sidecar_dir = tmp_path / "memory-integration"

    provider.handle_tool_call(
        "memory_integration_status",
        {},
        config={"memory_integration": {"vault": {"mode": "shared"}}},
        hermes_home=tmp_path,
    )

    assert not sidecar_dir.exists()


def test_initialize_is_read_only(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    provider.initialize("session-1", hermes_home=str(tmp_path))

    assert provider.handle_tool_call(
        "memory_integration_status",
        {},
        config={"memory_integration": {"vault": {"mode": "shared"}}},
        hermes_home=tmp_path,
    )
    assert not (tmp_path / "memory-integration").exists()


def test_adapter_unavailable_is_reported(load_provider, monkeypatch, tmp_path):
    provider = load_provider()
    monkeypatch.delitem(sys.modules, "vault_adapter", raising=False)

    status = provider.status(
        config={"memory_integration": {"vault": {"mode": "shared"}}},
        hermes_home=tmp_path,
    )

    assert status["ok"] is False
    assert "adapter_unavailable" in status["diagnostics"]


def test_typed_adapter_error_is_reported_without_traceback(load_provider, monkeypatch, tmp_path):
    import types

    provider = load_provider()
    module = types.ModuleType("vault_adapter")

    class VaultPathError(Exception):
        pass

    def resolve_vault_path(**kwargs):
        raise VaultPathError("/secret/path is invalid and should be bounded")

    setattr(module, "resolve_vault_path", resolve_vault_path)
    monkeypatch.setitem(sys.modules, "vault_adapter", module)

    status = provider.status(
        config={"memory_integration": {"vault": {"mode": "shared"}}},
        hermes_home=tmp_path,
    )

    assert status["ok"] is False
    assert "vault_path_error" in status["diagnostics"]
    assert status["error_type"] == "VaultPathError"
    assert "Traceback" not in json.dumps(status)


def test_tool_call_loads_runtime_config_from_hermes_home(load_provider, fake_vault_adapter, monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "memory_integration:\n"
        "  vault:\n"
        "    mode: shared\n"
        "  status:\n"
        "    include_absolute_paths: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    provider = load_provider()
    provider.initialize("session-1")

    raw = provider.handle_tool_call("memory_integration_status", {})
    status = json.loads(raw)

    assert status["ok"] is True
    assert status["vault"]["mode"] == "shared"
    assert status["vault"]["root"] == str(fake_vault_adapter)


def test_status_output_is_bounded_for_long_user_config(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path))
    raw = provider.handle_tool_call(
        "memory_integration_status",
        {},
        config={
            "memory_integration": {
                "vault": {"mode": "shared", "memory_subdir": "x" * 20000},
                "status": {"include_absolute_paths": True},
            }
        },
        hermes_home=tmp_path,
    )
    status = json.loads(raw)

    assert len(raw) < 4096
    assert "output_truncated" in status["diagnostics"]
    assert status["vault"]["memory_subdir"].endswith("...")


def test_unknown_tool_returns_json_error(load_provider):
    provider = load_provider()

    result = json.loads(provider.handle_tool_call("unexpected", {}))

    assert result["ok"] is False
    assert result["error"] == "unknown_tool"
