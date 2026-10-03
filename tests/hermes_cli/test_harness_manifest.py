"""Tests for the typed harness manifest and its config-loader integration."""

import pytest

from hermes_cli import harness_manifest as harness


def _reset_config_caches():
    from hermes_cli import config as config_mod, config_effective

    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()


def test_missing_manifest_is_stock_and_fingerprint_is_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = harness.show_state()
    assert state["overlays"] == []
    assert state["values"]["agent.gateway_timeout"] == 1800
    assert len(state["fingerprint"]) == 64
    assert harness.fingerprint() == state["fingerprint"]


def test_set_revert_and_runtime_cache_invalidation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _reset_config_caches()
    from hermes_cli import config

    assert config.load_config()["agent"]["max_turns"] is None
    harness.set_value("agent.max_turns", 777, overlay="explore", reason="bounded trial")
    assert config.load_config()["agent"]["max_turns"] == 777
    assert config.load_config()["agent"]["max_turns"] == 777
    harness.revert_overlay("explore")
    assert config.load_config()["agent"]["max_turns"] is None


def test_invalid_or_security_sensitive_keys_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(harness.HarnessManifestError, match="Unknown"):
        harness.set_value("security.redact_secrets", False, overlay="bad", reason="no")
    with pytest.raises(harness.HarnessManifestError, match="expects an integer"):
        harness.set_value("agent.max_turns", "777", overlay="bad", reason="no")
    with pytest.raises(harness.HarnessManifestError, match="<="):
        harness.set_value("compression.threshold", 2, overlay="bad", reason="no")


def test_stale_overlay_is_inactive_and_explained(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manifest = {
        "schema_version": 1,
        "stock_revision": "old-revision",
        "overlays": [{
            "id": "old", "name": "old", "reason": "old trial",
            "authored_against": "old-revision", "created_at": "2026-01-01T00:00:00Z",
            "values": {"agent.gateway_timeout": 1},
        }],
    }
    harness.save_manifest(manifest)
    state = harness.show_state()
    assert state["overlays"][0]["active"] is False
    assert state["values"]["agent.gateway_timeout"] == 1800
    explained = harness.explain("agent.gateway_timeout")
    assert explained["sources"] == []


def test_loaders_apply_same_active_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("display:\n  skin: test\n", encoding="utf-8")
    harness.set_value("compression.threshold", 0.7, overlay="compress", reason="trial")
    _reset_config_caches()
    from hermes_cli import config
    from hermes_cli.config_effective import load_user_config_effective

    assert config.load_config()["compression"]["threshold"] == pytest.approx(0.7)
    assert load_user_config_effective(tmp_path / "config.yaml")["compression"]["threshold"] == pytest.approx(0.7)


def test_fingerprint_changes_with_overlay_and_manifest_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    before = harness.fingerprint()
    harness.set_value("delegation.max_iterations", 99, overlay="delegate", reason="trial")
    after = harness.fingerprint()
    assert before != after
    assert harness.load_manifest()["overlays"][0]["values"] == {"delegation.max_iterations": 99}
    if hasattr((tmp_path / "harness.yaml").stat(), "st_mode"):
        # Windows does not expose POSIX permission semantics; on POSIX this is owner-only.
        import os
        if os.name == "posix":
            assert (tmp_path / "harness.yaml").stat().st_mode & 0o077 == 0

