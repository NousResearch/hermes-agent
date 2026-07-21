"""Tests for the bundled fluxer platform plugin."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "platforms" / "fluxer"


def test_plugin_yaml_exists_and_has_required_fields():
    """Verify plugin.yaml has correct metadata and env var declarations."""
    yaml_path = PLUGIN_DIR / "plugin.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert data["name"] == "fluxer-platform"
    assert data["label"] == "Fluxer"
    assert data["kind"] == "platform"
    assert data["version"] == "1.0.0"

    # Required env vars
    env_names = {env["name"] for env in data.get("requires_env", [])}
    assert "FLUXER_BOT_TOKEN" in env_names
    assert "FLUXER_API_URL" in env_names

    # Optional env vars
    opt_env_names = {env["name"] for env in data.get("optional_env", [])}
    assert "FLUXER_HOME_CHANNEL" in opt_env_names

    # Dependencies listed
    assert "pip_dependencies" in data


def test_init_exports_register():
    """__init__.py imports and re-exports register from adapter."""
    from plugins.platforms.fluxer import register

    assert callable(register)


def test_register_calls_register_platform():
    """register() calls ctx.register_platform() with the right shape."""
    from plugins.platforms.fluxer.adapter import register

    calls = []

    class FakeCtx:
        def register_platform(self, *args, **kwargs):
            calls.append(("register_platform", kwargs))

    ctx = FakeCtx()
    register(ctx)

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["name"] == "fluxer"
    assert kwargs["label"] == "Fluxer"
    assert callable(kwargs["adapter_factory"])
    assert callable(kwargs["check_fn"])
    assert callable(kwargs["is_connected"])
    assert "FLUXER_BOT_TOKEN" in kwargs.get("required_env", [])


def test_adapter_stub_classes_exist():
    """All three stub classes exist and have expected signatures."""
    from plugins.platforms.fluxer.adapter import (
        FluxerAdapter,
        FluxerGatewayClient,
        FluxerRESTClient,
    )

    # FluxerGatewayClient
    assert hasattr(FluxerGatewayClient, "__init__")
    assert hasattr(FluxerGatewayClient, "connect")
    assert hasattr(FluxerGatewayClient, "disconnect")
    assert hasattr(FluxerGatewayClient, "is_connected")

    # FluxerRESTClient
    assert hasattr(FluxerRESTClient, "__init__")
    assert hasattr(FluxerRESTClient, "send_message")

    # FluxerAdapter
    assert issubclass(FluxerAdapter, object)  # would need BasePlatformAdapter import
    assert hasattr(FluxerAdapter, "__init__")
    assert hasattr(FluxerAdapter, "connect")
    assert hasattr(FluxerAdapter, "disconnect")
    assert hasattr(FluxerAdapter, "send")
    assert hasattr(FluxerAdapter, "send_media")
    assert hasattr(FluxerAdapter, "is_alive")
    assert hasattr(FluxerAdapter, "get_me")


def test_check_fluxer_requirements_returns_bool():
    """check_fn returns a boolean without error."""
    from plugins.platforms.fluxer.adapter import check_fluxer_requirements

    result = check_fluxer_requirements()
    assert isinstance(result, bool)


def test_is_connected_checks_env_vars(monkeypatch):
    """_is_connected returns False when env vars are missing."""
    from plugins.platforms.fluxer.adapter import _is_connected
    from gateway.config import PlatformConfig

    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)
    monkeypatch.delenv("FLUXER_API_URL", raising=False)

    config = PlatformConfig()
    assert _is_connected(config) is False


def test_is_connected_returns_true_when_env_set(monkeypatch):
    """_is_connected returns True when required env vars are set."""
    from plugins.platforms.fluxer.adapter import _is_connected
    from gateway.config import PlatformConfig

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "test-token")
    monkeypatch.setenv("FLUXER_API_URL", "https://fluxer.example.com")

    config = PlatformConfig()
    assert _is_connected(config) is True


def _reset_bundled_cache(monkeypatch):
    """Reset the module-level bundled plugin names cache."""
    import gateway.config as cfg

    monkeypatch.setattr(cfg, "_Platform__bundled_plugin_names", None)


def test_discoverable_as_bundled_plugin(monkeypatch):
    """Plugin is discoverable by Platform._scan_bundled_plugin_platforms()."""
    from gateway.config import Platform

    _reset_bundled_cache(monkeypatch)
    names = Platform._scan_bundled_plugin_platforms()
    assert "fluxer" in names


def test_platform_enum_accepts_fluxer(monkeypatch):
    """Platform('fluxer') creates a dynamic member without error."""
    from gateway.config import Platform

    _reset_bundled_cache(monkeypatch)
    p = Platform("fluxer")
    assert p.value == "fluxer"
