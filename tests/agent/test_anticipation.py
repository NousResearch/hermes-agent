"""Tests for anticipation data types and validation."""

from math import inf, nan

import pytest

from agent.anticipation import (
    AnticipationLoopConfig,
    AnticipationPermission,
    AnticipationRuntimeConfig,
    parse_loop_config,
    parse_permission,
    parse_bool,
)


@pytest.mark.parametrize("permission", [permission.value for permission in AnticipationPermission])
def test_parse_permission_accepts_all_permission_values(permission):
    assert parse_permission(permission).value == permission


def test_parse_permission_accepts_enum_value():
    assert parse_permission(AnticipationPermission.DRAFT) is AnticipationPermission.DRAFT


def test_parse_permission_rejects_invalid_value_with_allowed_values():
    with pytest.raises(ValueError) as excinfo:
        parse_permission("do_whatever")

    message = str(excinfo.value)
    assert "do_whatever" in message
    assert "suggest" in message
    assert "execute_safe" in message


def test_parse_loop_config_builds_typed_loop_config():
    loop = parse_loop_config(
        "stale_task_resurfacer",
        {
            "enabled": True,
            "schedule": "0 9 * * *",
            "permission": "draft",
            "min_confidence": 0.8,
            "lookback_days": 21,
        },
    )

    assert loop == AnticipationLoopConfig(
        name="stale_task_resurfacer",
        enabled=True,
        schedule="0 9 * * *",
        permission=AnticipationPermission.DRAFT,
        min_confidence=0.8,
        lookback_days=21,
    )


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", ""])
def test_parse_bool_treats_false_strings_as_false(value):
    assert parse_bool(value) is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on"])
def test_parse_bool_treats_true_strings_as_true(value):
    assert parse_bool(value) is True


def test_parse_loop_config_does_not_enable_quoted_false():
    loop = parse_loop_config(
        "router_monitor",
        {
            "enabled": "false",
            "schedule": "manual",
            "permission": "ask_to_execute",
            "min_confidence": 0.8,
            "lookback_days": 1,
        },
    )

    assert loop.enabled is False


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_parse_loop_config_rejects_confidence_outside_unit_interval(confidence):
    with pytest.raises(ValueError, match="min_confidence"):
        parse_loop_config(
            "stale_task_resurfacer",
            {
                "enabled": False,
                "schedule": "",
                "permission": "suggest",
                "min_confidence": confidence,
            },
        )


@pytest.mark.parametrize("confidence", [nan, inf, -inf])
def test_parse_loop_config_rejects_non_finite_confidence(confidence):
    with pytest.raises(ValueError, match="min_confidence"):
        parse_loop_config(
            "stale_task_resurfacer",
            {
                "enabled": False,
                "schedule": "",
                "permission": "suggest",
                "min_confidence": confidence,
            },
        )


def test_parse_loop_config_requires_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_loop_config("stale_task_resurfacer", None)


def test_runtime_config_rejects_non_finite_min_confidence():
    with pytest.raises(ValueError, match="min_confidence"):
        AnticipationRuntimeConfig(
            enabled=True,
            loop_enabled=True,
            loop_permission=AnticipationPermission.SUGGEST,
            min_confidence=nan,
            quiet_hours_enabled=False,
            quiet_hours_start="22:00",
            quiet_hours_end="08:00",
            max_per_day=3,
            min_minutes_between=120,
        )


def test_runtime_config_rejects_negative_budget_values():
    with pytest.raises(ValueError, match="max_per_day"):
        AnticipationRuntimeConfig(
            enabled=True,
            loop_enabled=True,
            loop_permission=AnticipationPermission.SUGGEST,
            min_confidence=0.7,
            quiet_hours_enabled=False,
            quiet_hours_start="22:00",
            quiet_hours_end="08:00",
            max_per_day=-1,
            min_minutes_between=120,
        )
