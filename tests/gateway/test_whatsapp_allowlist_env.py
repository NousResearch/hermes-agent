"""Regression tests for issue #61913.

The Node bridge reads WHATSAPP_ALLOWED_USERS / WHATSAPP_GROUP_ALLOWED_USERS
directly from the environment, but the Python adapter's __init__ only read
allow_from/group_allow_from from config.extra, so a bare-env-var setup
(no duplication under the whatsapp config block) was silently dropped.
"""
from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig


def _build_adapter(monkeypatch, tmp_path, extra=None, env=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setattr(WhatsAppAdapter, "_DEFAULT_BRIDGE_DIR", tmp_path / "bridge")

    for key in ("WHATSAPP_ALLOWED_USERS", "WHATSAPP_GROUP_ALLOWED_USERS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    config = PlatformConfig(enabled=True, extra=extra or {})
    return WhatsAppAdapter(config)


def test_dm_allowlist_falls_back_to_env_var(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch, tmp_path,
        extra={"dm_policy": "allowlist"},
        env={"WHATSAPP_ALLOWED_USERS": "6281234567890,6281111111111"},
    )

    assert adapter._allow_from == {"6281234567890", "6281111111111"}


def test_group_allowlist_falls_back_to_env_var(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch, tmp_path,
        extra={"group_policy": "allowlist"},
        env={"WHATSAPP_GROUP_ALLOWED_USERS": "120363001234567890@g.us"},
    )

    assert adapter._group_allow_from == {"120363001234567890@g.us"}


def test_config_extra_still_takes_precedence_over_env_var(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch, tmp_path,
        extra={"dm_policy": "allowlist", "allow_from": ["6281234567890"]},
        env={"WHATSAPP_ALLOWED_USERS": "6289999999999"},
    )

    assert adapter._allow_from == {"6281234567890"}


def test_no_env_var_and_no_config_gives_empty_allowlist(monkeypatch, tmp_path):
    adapter = _build_adapter(monkeypatch, tmp_path, extra={"dm_policy": "allowlist"})

    assert adapter._allow_from == set()

def test_explicit_empty_allow_from_does_not_fallback_to_env(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch, tmp_path,
        extra={"dm_policy": "allowlist", "allow_from": []},
        env={"WHATSAPP_ALLOWED_USERS": "6289999999999"},
    )

    assert adapter._allow_from == set()


def test_explicit_empty_group_allow_from_does_not_fallback_to_env(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch, tmp_path,
        extra={"group_policy": "allowlist", "group_allow_from": []},
        env={"WHATSAPP_GROUP_ALLOWED_USERS": "120363009999999999@g.us"},
    )

    assert adapter._group_allow_from == set()

