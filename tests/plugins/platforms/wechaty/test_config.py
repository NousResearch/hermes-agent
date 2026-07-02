"""Config, env enablement, and YAML bridge tests for Wechaty."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wechaty import adapter as wechaty_adapter
from plugins.platforms.wechaty.adapter import (
    _apply_yaml_config,
    _env_enablement,
    check_requirements,
    is_connected,
    validate_config,
)


def test_validate_config_requires_puppet_or_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WECHATY_PUPPET", raising=False)
    monkeypatch.delenv("WECHATY_PUPPET_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("WECHATY_TOKEN", raising=False)
    cfg = PlatformConfig(enabled=True, extra={})
    assert validate_config(cfg) is False

    cfg_puppet = PlatformConfig(
        enabled=True, extra={"puppet": "wechaty-puppet-wechat4u"}
    )
    assert validate_config(cfg_puppet) is True

    cfg_token = PlatformConfig(enabled=True, extra={"puppet_token": "tok"})
    assert validate_config(cfg_token) is True


def test_is_connected_mirrors_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WECHATY_PUPPET", raising=False)
    monkeypatch.delenv("WECHATY_PUPPET_SERVICE_TOKEN", raising=False)
    cfg = PlatformConfig(enabled=True, extra={"puppet": "wechaty-puppet-wechat4u"})
    assert is_connected(cfg) is validate_config(cfg)


def test_env_enablement_seeds_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WECHATY_PUPPET", "wechaty-puppet-wechat4u")
    monkeypatch.setenv("WECHATY_PUPPET_SERVICE_TOKEN", "secret")
    monkeypatch.setenv("WECHATY_HOME_CHANNEL", "contact:abc")
    monkeypatch.setenv("WECHATY_HOME_CHANNEL_NAME", "Home Chat")
    seed = _env_enablement()
    assert seed is not None
    assert seed["puppet"] == "wechaty-puppet-wechat4u"
    assert seed["puppet_token"] == "secret"
    assert seed["home_channel"]["chat_id"] == "contact:abc"
    assert seed["home_channel"]["name"] == "Home Chat"


def test_apply_yaml_config_bridges_extra_without_overwriting_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WECHATY_PUPPET", raising=False)
    monkeypatch.delenv("WECHATY_MENTION_PATTERNS", raising=False)
    monkeypatch.setenv("WECHATY_ALLOWED_USERS", "existing-user")
    platform_cfg = PlatformConfig(
        enabled=True,
        extra={
            "puppet": "wechaty-puppet-wechat4u",
            "mention_patterns": "@hermes",
            "allowed_users": "user-a,user-b",
            "home_channel": {"chat_id": "room:xyz", "name": "Ops"},
        },
    )
    seed = _apply_yaml_config({}, platform_cfg)
    assert seed is not None
    assert os.environ["WECHATY_PUPPET"] == "wechaty-puppet-wechat4u"
    assert os.environ["WECHATY_MENTION_PATTERNS"] == "@hermes"
    assert os.environ["WECHATY_HOME_CHANNEL"] == "room:xyz"
    # env wins over YAML for allowed_users
    assert os.environ.get("WECHATY_ALLOWED_USERS") == "existing-user"
    assert "allowed_users" not in (seed or {})


def test_check_requirements_needs_node_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not wechaty_adapter.HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")
    monkeypatch.setattr(wechaty_adapter.shutil, "which", lambda _: "/usr/bin/node")
    monkeypatch.setattr(wechaty_adapter, "_SIDECAR_DIR", tmp_path)
    assert check_requirements() is False
    (tmp_path / "node_modules").mkdir()
    assert check_requirements() is True
