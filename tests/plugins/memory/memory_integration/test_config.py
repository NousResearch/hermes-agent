from __future__ import annotations

from pathlib import Path

import pytest


def test_missing_mode_is_not_configured(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    status = provider.status(config={"memory_integration": {}}, hermes_home=tmp_path)

    assert status["ok"] is False
    assert status["configured"] is False
    assert "mode_required" in status["diagnostics"]


def test_shared_mode_defaults_memory_subdir_and_sidecar(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    status = provider.status(
        config={"memory_integration": {"vault": {"mode": "shared"}}},
        hermes_home=tmp_path,
    )

    assert status["ok"] is False
    assert status["configured"] is True
    assert status["initialized"] is False
    assert status["vault"]["mode"] == "shared"
    assert status["vault"]["root"] == "<redacted>"
    assert status["vault"]["memory_subdir"] == "wiki/memory-integration"
    assert status["vault"]["memory_root"] == "<redacted>"
    assert status["sidecar"]["path"] == "<redacted>"
    assert "not_initialized" in status["diagnostics"]


def test_initialized_shared_mode_is_ok(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path))

    status = provider.status(config={"memory_integration": {"vault": {"mode": "shared"}}})

    assert status["ok"] is True
    assert status["initialized"] is True
    assert "not_initialized" not in status["diagnostics"]


def test_dedicated_mode_rejects_memory_subdir(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    status = provider.status(
        config={"memory_integration": {"vault": {"mode": "dedicated", "memory_subdir": "ignored"}}},
        hermes_home=tmp_path,
    )

    assert status["ok"] is False
    assert "memory_subdir_not_allowed" in status["diagnostics"]


def test_path_redaction_when_absolute_paths_disabled(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    status = provider.status(
        config={
            "memory_integration": {
                "vault": {"mode": "shared"},
                "status": {"include_absolute_paths": False},
            }
        },
        hermes_home=tmp_path,
    )

    assert status["vault"]["root"] == "<redacted>"
    assert status["vault"]["memory_root"] == "<redacted>"
    assert status["sidecar"]["path"] == "<redacted>"


def test_absolute_paths_require_opt_in(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    status = provider.status(
        config={
            "memory_integration": {
                "vault": {"mode": "shared"},
                "status": {"include_absolute_paths": True},
            }
        },
        hermes_home=tmp_path,
    )

    assert status["vault"]["root"] == str(fake_vault_adapter)
    assert status["vault"]["memory_root"] == str(fake_vault_adapter / "wiki/memory-integration")
    assert status["sidecar"]["path"] == str(tmp_path / "memory-integration" / "memory_integration.db")


def test_boolean_string_false_is_safe_false(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    status = provider.status(
        config={
            "memory_integration": {
                "vault": {"mode": "shared"},
                "status": {"include_absolute_paths": "false"},
            }
        },
        hermes_home=tmp_path,
    )

    assert status["vault"]["root"] == "<redacted>"


def test_boolean_string_true_opts_in(load_provider, fake_vault_adapter, tmp_path):
    provider = load_provider()

    status = provider.status(
        config={
            "memory_integration": {
                "vault": {"mode": "shared"},
                "status": {"include_absolute_paths": "yes"},
            }
        },
        hermes_home=tmp_path,
    )

    assert status["vault"]["root"] == str(fake_vault_adapter)


@pytest.mark.parametrize("path", ["relative/vault"])
def test_explicit_vault_path_must_be_absolute(load_provider, tmp_path, path):
    provider = load_provider()

    status = provider.status(
        config={"memory_integration": {"vault": {"mode": "shared", "path": path}}},
        hermes_home=tmp_path,
    )

    assert status["ok"] is False
    assert "vault_path_invalid" in status["diagnostics"]
