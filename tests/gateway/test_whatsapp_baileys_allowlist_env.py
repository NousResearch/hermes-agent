"""Regression: Baileys WhatsApp adapter must honor documented env allowlists.

Wizard / .env-only installs write WHATSAPP_ALLOWED_USERS and
WHATSAPP_GROUP_ALLOWED_USERS but leave PlatformConfig.extra empty. Before
this fix the adapter only read allow_from / group_allow_from from extra,
so dm_policy=allowlist + a non-empty env allowlist still ran with an empty
set and silently dropped every inbound after the bridge queued it.

Mirrors the WhatsApp Cloud salvage (PR #58504 / #58448) and adopts
key-presence semantics so explicit allow_from: [] stays deny-all even when
a stale env allowlist is present (review feedback on #61924).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import PlatformConfig


def _build_adapter(monkeypatch, tmp_path, extra=None, env=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setattr(WhatsAppAdapter, "_DEFAULT_BRIDGE_DIR", tmp_path / "bridge")

    for key in (
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_ALLOW_FROM",
        "WHATSAPP_ALLOW_ALL_USERS",
        "WHATSAPP_GROUP_ALLOWED_USERS",
        "WHATSAPP_GROUP_ALLOW_FROM",
        "WHATSAPP_DM_POLICY",
        "WHATSAPP_GROUP_POLICY",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    config = PlatformConfig(enabled=True, extra=extra or {})
    return WhatsAppAdapter(config)


def test_dm_allowlist_falls_back_to_env_var(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"dm_policy": "allowlist"},
        env={"WHATSAPP_ALLOWED_USERS": "15551234567,15557654321"},
    )
    assert adapter._allow_from == {"15551234567", "15557654321"}


def test_group_allowlist_falls_back_to_env_var(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"group_policy": "allowlist"},
        env={"WHATSAPP_GROUP_ALLOWED_USERS": "120363001234567890@g.us"},
    )
    assert adapter._group_allow_from == {"120363001234567890@g.us"}


def test_group_allow_from_legacy_env_alias(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        env={"WHATSAPP_GROUP_ALLOW_FROM": "120363009999999999@g.us"},
    )
    assert "120363009999999999@g.us" in adapter._group_allow_from


def test_explicit_empty_allow_from_does_not_fallback_to_env(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"dm_policy": "allowlist", "allow_from": []},
        env={"WHATSAPP_ALLOWED_USERS": "15551234567"},
    )
    # Explicit empty config must stay deny-all (no silent widen via env).
    assert adapter._allow_from == set()


def test_explicit_empty_group_allow_from_does_not_fallback_to_env(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"group_policy": "allowlist", "group_allow_from": []},
        env={"WHATSAPP_GROUP_ALLOWED_USERS": "120363001234567890@g.us"},
    )
    assert adapter._group_allow_from == set()


def test_config_extra_wins_over_env(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"allow_from": ["15550000001"]},
        env={"WHATSAPP_ALLOWED_USERS": "15559999999"},
    )
    assert adapter._allow_from == {"15550000001"}


def test_allow_all_users_opts_in_when_no_allowlist_configured(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        env={"WHATSAPP_ALLOW_ALL_USERS": "true"},
    )
    assert adapter._allow_from == {"*"}


def test_allow_all_users_does_not_override_explicit_empty(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"allow_from": []},
        env={"WHATSAPP_ALLOW_ALL_USERS": "true"},
    )
    assert adapter._allow_from == set()
