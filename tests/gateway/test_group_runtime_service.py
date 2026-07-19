"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
from __future__ import annotations
import pytest

pytestmark = pytest.mark.dead_runtime_service


from gateway.group_runtime_service import (
    qq_group_message_allowed,
    qq_policy_has_runtime_override,
    resolve_qq_effective_group_policy,
    resolve_weixin_effective_group_policy,
    weixin_group_message_allowed,
)


def test_qq_group_message_allowed_respects_overlay_policy_and_allowlist():
    assert qq_group_message_allowed(
        "123",
        allow_all_groups=False,
        allowed_groups=set(),
        has_policy=False,
        overlay_active=True,
    )
    assert qq_group_message_allowed(
        "123",
        allow_all_groups=False,
        allowed_groups=set(),
        has_policy=True,
        overlay_active=False,
    )
    assert not qq_group_message_allowed(
        "123",
        allow_all_groups=False,
        allowed_groups={"456"},
        has_policy=False,
        overlay_active=False,
    )
    assert qq_group_message_allowed(
        "123",
        allow_all_groups=False,
        allowed_groups={"123"},
        has_policy=False,
        overlay_active=False,
    )


def test_qq_policy_has_runtime_override_detects_non_default_policy_bits():
    assert not qq_policy_has_runtime_override({"mode": "default", "purge_raw_after_rollup": True})
    assert qq_policy_has_runtime_override({"mode": "collect_only", "purge_raw_after_rollup": True})
    assert qq_policy_has_runtime_override({"mode": "default", "daily_report_target": "qq_napcat:dm:1"})


def test_resolve_qq_effective_group_policy_merges_overlay_only_without_override():
    policy = {"mode": "default", "archive_enabled": False, "purge_raw_after_rollup": True}
    overlay = {
        "active": True,
        "archive_enabled": True,
        "daily_report_enabled": True,
        "workers": [
            {
                "daily_report_enabled": True,
                "daily_report_target": "qq_napcat:dm:179033731",
                "manual_report_target": "qq_napcat:dm:179033731",
                "notify_target": "qq_napcat:dm:179033731",
            }
        ],
    }

    merged = resolve_qq_effective_group_policy(
        "123",
        policy_loader=lambda _gid: dict(policy),
        default_policy_loader=lambda gid: {"mode": "default", "group_id": gid},
        overlay_loader=lambda _gid: overlay,
    )

    assert merged["mode"] == "collect_only"
    assert merged["archive_enabled"] is True
    assert merged["daily_report_enabled"] is True
    assert merged["daily_report_target"] == "qq_napcat:dm:179033731"
    assert merged["manual_report_targets"] == ["qq_napcat:dm:179033731"]
    assert merged["notify_targets"] == ["qq_napcat:dm:179033731"]


def test_resolve_qq_effective_group_policy_falls_back_to_default_on_loader_error():
    fallback = resolve_qq_effective_group_policy(
        "123",
        policy_loader=lambda _gid: (_ for _ in ()).throw(RuntimeError("boom")),
        default_policy_loader=lambda gid: {"mode": "default", "group_id": gid},
        overlay_loader=lambda _gid: {"active": False},
    )

    assert fallback == {"mode": "default", "group_id": "123"}


def test_weixin_group_message_allowed_respects_mode_and_allowlist():
    assert weixin_group_message_allowed(
        "room-1",
        has_policy=True,
        group_policy_mode="disabled",
        group_allow_from=set(),
    )
    assert not weixin_group_message_allowed(
        "room-1",
        has_policy=False,
        group_policy_mode="disabled",
        group_allow_from=set(),
    )
    assert not weixin_group_message_allowed(
        "room-1",
        has_policy=False,
        group_policy_mode="allowlist",
        group_allow_from={"room-2"},
    )
    assert weixin_group_message_allowed(
        "room-1",
        has_policy=False,
        group_policy_mode="allowlist",
        group_allow_from={"room-1"},
    )


def test_resolve_weixin_effective_group_policy_falls_back_to_default_on_error():
    policy = resolve_weixin_effective_group_policy(
        "room-1",
        policy_loader=lambda _gid: (_ for _ in ()).throw(RuntimeError("boom")),
        default_policy_loader=lambda gid: {"mode": "default", "chat_id": gid},
    )

    assert policy == {"mode": "default", "chat_id": "room-1"}
