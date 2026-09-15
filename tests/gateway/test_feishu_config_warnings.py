"""Startup diagnostics for unusable Feishu group-admission settings."""

from __future__ import annotations

import asyncio
import logging

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


_POLICY_WARNING = "all human group messages will be rejected"


def _config(**extra: object) -> PlatformConfig:
    return PlatformConfig(
        enabled=True,
        extra={"app_id": "cli_test", "app_secret": "secret_test", **extra},
    )


def test_startup_warns_when_group_allowlist_cannot_admit_humans(monkeypatch, caplog):
    monkeypatch.delenv("FEISHU_GROUP_POLICY", raising=False)
    monkeypatch.delenv("FEISHU_ALLOWED_USERS", raising=False)
    monkeypatch.setattr("plugins.platforms.feishu.adapter._load_lark_oapi", lambda: False)

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.feishu.adapter"):
        asyncio.run(FeishuAdapter(_config()).connect())

    assert _POLICY_WARNING in caplog.text


@pytest.mark.parametrize(
    ("env", "extra"),
    [
        ({"FEISHU_ALLOWED_USERS": "ou_operator"}, {}),
        ({"FEISHU_GROUP_POLICY": "open"}, {}),
        ({}, {"admins": ["ou_admin"]}),
        ({}, {"group_rules": {"oc_chat": {"policy": "open"}}}),
    ],
)
def test_startup_skips_group_warning_when_human_access_is_configured(
    monkeypatch, caplog, env, extra,
):
    monkeypatch.delenv("FEISHU_GROUP_POLICY", raising=False)
    monkeypatch.delenv("FEISHU_ALLOWED_USERS", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("plugins.platforms.feishu.adapter._load_lark_oapi", lambda: False)

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.feishu.adapter"):
        asyncio.run(FeishuAdapter(_config(**extra)).connect())

    assert _POLICY_WARNING not in caplog.text
