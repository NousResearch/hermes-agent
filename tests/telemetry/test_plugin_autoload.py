"""Telemetry plugin auto-load gating: on by default, off when telemetry.local=false."""

from __future__ import annotations

import hermes_cli.plugins as plugins_mod


def test_local_enabled_defaults_true(monkeypatch):
    monkeypatch.setattr(plugins_mod, "load_config", lambda: {}, raising=False)
    # With no telemetry section, the default is on.
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {}, raising=False
    )
    assert plugins_mod._telemetry_local_enabled() is True


def test_local_disabled_when_config_false(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"telemetry": {"local": False}},
        raising=False,
    )
    assert plugins_mod._telemetry_local_enabled() is False


def test_local_enabled_when_config_true(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"telemetry": {"local": True}},
        raising=False,
    )
    assert plugins_mod._telemetry_local_enabled() is True


def test_malformed_config_defaults_on(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"telemetry": "not a dict"},
        raising=False,
    )
    assert plugins_mod._telemetry_local_enabled() is True


def test_plugin_manifest_is_discoverable():
    """The bundled telemetry plugin.yaml exists and declares the lifecycle hooks."""
    from pathlib import Path
    import hermes_cli.plugins as p
    bundled = p.get_bundled_plugins_dir()
    manifest = bundled / "telemetry" / "plugin.yaml"
    assert manifest.exists(), f"missing {manifest}"
    text = manifest.read_text(encoding="utf-8")
    for hook in ("post_api_request", "post_tool_call", "on_session_finalize"):
        assert hook in text
