"""Pure auxiliary config parity tests for gateway-adjacent paths.

These tests use temp ``HERMES_HOME`` config files and dummy provider/model
strings only.  They do not launch the gateway, agents, cron, providers, live
models, or read the real profile config/credentials.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from gateway.config import GatewayConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import config as hermes_config
from hermes_cli.config import cfg_get


REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_RUN_PATH = REPO_ROOT / "gateway" / "run.py"
AUXILIARY_ENV_KEYS = (
    "AUXILIARY_VISION_PROVIDER",
    "AUXILIARY_VISION_MODEL",
    "AUXILIARY_VISION_BASE_URL",
    "AUXILIARY_WEB_EXTRACT_PROVIDER",
    "AUXILIARY_WEB_EXTRACT_MODEL",
    "AUXILIARY_WEB_EXTRACT_BASE_URL",
    "AUXILIARY_APPROVAL_PROVIDER",
    "AUXILIARY_APPROVAL_MODEL",
    "AUXILIARY_APPROVAL_BASE_URL",
    "AUXILIARY_PLUGIN_AUDIT_PROVIDER",
    "AUXILIARY_PLUGIN_AUDIT_MODEL",
    "AUXILIARY_PLUGIN_AUDIT_BASE_URL",
    "AUXILIARY_COMPRESSION_PROVIDER",
    "AUXILIARY_COMPRESSION_MODEL",
)


@pytest.fixture(autouse=True)
def isolated_auxiliary_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in AUXILIARY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_hermes_home(tmp_path: Path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    token = set_hermes_home_override(home)
    hermes_config._RAW_CONFIG_CACHE.clear()
    hermes_config._LOAD_CONFIG_CACHE.clear()
    try:
        yield home
    finally:
        hermes_config._RAW_CONFIG_CACHE.clear()
        hermes_config._LOAD_CONFIG_CACHE.clear()
        reset_hermes_home_override(token)


def _write_config(home: Path, config: dict[str, Any]) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _gateway_bridge_source() -> str:
    """Extract only gateway/run.py's startup config bridge block."""

    source = GATEWAY_RUN_PATH.read_text(encoding="utf-8")
    start_marker = "# Bridge config.yaml values into the environment so os.getenv() picks them up."
    end_marker = "from gateway.config import ("
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _install_fake_plugin_aux_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real plugin discovery while preserving gateway bridge semantics."""

    fake_plugins = types.ModuleType("hermes_cli.plugins")
    fake_plugins.get_plugin_auxiliary_tasks = lambda: [  # type: ignore[attr-defined]
        {"key": "plugin_audit"}
    ]
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)


def _run_gateway_bridge(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_plugin_aux_registry(monkeypatch)
    ns = {
        "__file__": str(GATEWAY_RUN_PATH),
        "_hermes_home": home,
        "os": os,
        "Path": Path,
        "sys": sys,
        "json": __import__("json"),
    }
    exec(compile(_gateway_bridge_source(), str(GATEWAY_RUN_PATH), "exec"), ns)


def test_full_config_loader_preserves_auxiliary_slots_from_temp_home(
    temp_hermes_home: Path,
) -> None:
    _write_config(
        temp_hermes_home,
        {
            "model": {"provider": "dummy-main-provider", "default": "dummy-main-model"},
            "auxiliary": {
                "vision": {
                    "provider": "dummy-vision-provider",
                    "model": "dummy-vision-model",
                    "base_url": "http://127.0.0.1:65535/vision/v1",
                },
                "approval": {
                    "provider": "dummy-approval-provider",
                    "model": "dummy-approval-model",
                    "timeout": 17,
                },
                "compression": {
                    "provider": "dummy-compression-provider",
                    "model": "dummy-compression-model",
                    "extra_body": {"routing_hint": "dummy"},
                },
                "goal_judge": {"max_tokens": 321},
            },
        },
    )

    raw = hermes_config.read_raw_config()
    loaded = hermes_config.load_config()

    assert raw["auxiliary"]["vision"]["provider"] == "dummy-vision-provider"
    assert loaded["auxiliary"]["vision"]["provider"] == "dummy-vision-provider"
    assert loaded["auxiliary"]["vision"]["model"] == "dummy-vision-model"
    assert loaded["auxiliary"]["vision"]["base_url"] == "http://127.0.0.1:65535/vision/v1"
    # Defaults are deep-merged around user overrides, so consumers see a complete slot.
    assert loaded["auxiliary"]["vision"]["timeout"] == 120
    assert loaded["auxiliary"]["approval"]["timeout"] == 17
    assert loaded["auxiliary"]["compression"]["extra_body"] == {"routing_hint": "dummy"}
    assert cfg_get(loaded, "auxiliary", "goal_judge", "max_tokens") == 321


def test_gateway_auxiliary_bridge_uses_temp_config_and_plugin_registry_without_live_calls(
    temp_hermes_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        temp_hermes_home,
        {
            "auxiliary": {
                "vision": {
                    "provider": "dummy-vision-provider",
                    "model": "dummy-vision-model",
                    "base_url": "http://127.0.0.1:65535/vision/v1",
                },
                "web_extract": {
                    "provider": "auto",
                    "model": "dummy-web-model",
                    "base_url": "http://127.0.0.1:65535/web/v1",
                },
                "approval": {
                    "provider": "dummy-approval-provider",
                    "model": "dummy-approval-model",
                },
                "plugin_audit": {
                    "provider": "dummy-plugin-provider",
                    "model": "dummy-plugin-model",
                    "base_url": "http://127.0.0.1:65535/plugin/v1",
                },
                # Compression is config-file-only for agent/auxiliary_client consumers.
                "compression": {
                    "provider": "dummy-compression-provider",
                    "model": "dummy-compression-model",
                },
            }
        },
    )

    _run_gateway_bridge(temp_hermes_home, monkeypatch)

    assert os.environ["AUXILIARY_VISION_PROVIDER"] == "dummy-vision-provider"
    assert os.environ["AUXILIARY_VISION_MODEL"] == "dummy-vision-model"
    assert os.environ["AUXILIARY_VISION_BASE_URL"] == "http://127.0.0.1:65535/vision/v1"
    assert "AUXILIARY_WEB_EXTRACT_PROVIDER" not in os.environ
    assert os.environ["AUXILIARY_WEB_EXTRACT_MODEL"] == "dummy-web-model"
    assert os.environ["AUXILIARY_WEB_EXTRACT_BASE_URL"] == "http://127.0.0.1:65535/web/v1"
    assert os.environ["AUXILIARY_APPROVAL_PROVIDER"] == "dummy-approval-provider"
    assert os.environ["AUXILIARY_APPROVAL_MODEL"] == "dummy-approval-model"
    assert os.environ["AUXILIARY_PLUGIN_AUDIT_PROVIDER"] == "dummy-plugin-provider"
    assert os.environ["AUXILIARY_PLUGIN_AUDIT_MODEL"] == "dummy-plugin-model"
    assert os.environ["AUXILIARY_PLUGIN_AUDIT_BASE_URL"] == "http://127.0.0.1:65535/plugin/v1"
    assert "AUXILIARY_COMPRESSION_PROVIDER" not in os.environ
    assert "AUXILIARY_COMPRESSION_MODEL" not in os.environ


def test_gateway_approval_warning_surface_reads_auxiliary_approval_from_full_config(
    temp_hermes_home: Path,
) -> None:
    _write_config(
        temp_hermes_home,
        {
            "approvals": {"mode": "manual"},
            "security": {"tirith_enabled": False},
            "auxiliary": {
                "approval": {
                    "provider": "dummy-approval-provider",
                    "model": "dummy-approval-model",
                }
            },
        },
    )

    loaded = hermes_config.load_config()
    approval_slot = cfg_get(loaded, "auxiliary", "approval", default=None)

    assert approval_slot["provider"] == "dummy-approval-provider"
    assert approval_slot["model"] == "dummy-approval-model"

    gateway_source = GATEWAY_RUN_PATH.read_text(encoding="utf-8")
    assert "from hermes_cli.config import load_config as _load_full_config" in gateway_source
    assert 'cfg_get(_appr_cfg, "auxiliary", "approval", default=None)' in gateway_source


def test_gateway_config_contract_excludes_auxiliary_routing_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(GatewayConfig)}
    forbidden_auxiliary_fields = {
        "auxiliary",
        "auxiliary_vision_provider",
        "auxiliary_vision_model",
        "auxiliary_web_extract_provider",
        "auxiliary_web_extract_model",
        "auxiliary_approval_provider",
        "auxiliary_approval_model",
        "auxiliary_compression_provider",
        "auxiliary_compression_model",
    }

    assert field_names.isdisjoint(forbidden_auxiliary_fields)

    cfg = GatewayConfig.from_dict(
        {
            "auxiliary": {
                "vision": {
                    "provider": "dummy-vision-provider",
                    "model": "dummy-vision-model",
                },
                "approval": {
                    "provider": "dummy-approval-provider",
                    "model": "dummy-approval-model",
                },
            }
        }
    )

    assert cfg.to_dict().keys().isdisjoint(forbidden_auxiliary_fields)
