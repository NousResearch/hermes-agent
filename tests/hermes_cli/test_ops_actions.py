from datetime import datetime, timedelta, timezone

import pytest


def _approved_record(**overrides):
    now = datetime.now(timezone.utc)
    record = {
        "id": "appr_test",
        "status": "approved",
        "risk_label": "Read-only",
        "target": "read_only_status_probe",
        "proposed_action": "read_only_status_probe",
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }
    record.update(overrides)
    return record


def test_action_registry_lists_only_fixed_actions():
    from hermes_cli.ops_actions import ACTIONS, get_action

    assert "read_only_status_probe" in ACTIONS
    assert "shell" not in ACTIONS
    assert "gateway_restart" not in ACTIONS
    assert get_action("read_only_status_probe").risk_label == "Read-only"


def test_unknown_fixed_action_is_rejected():
    from hermes_cli.ops_actions import ActionError, get_action

    with pytest.raises(ActionError, match="Unknown ops action"):
        get_action("shell")


def test_default_config_denies_action_preflight():
    from hermes_cli.ops_actions import ActionError, preflight_action_config

    with pytest.raises(ActionError, match="disabled"):
        preflight_action_config({}, "read_only_status_probe")


def test_enabled_config_requires_exact_allowlist():
    from hermes_cli.ops_actions import ActionError, preflight_action_config

    config = {"ops_center": {"action_execution_enabled": True, "allowed_actions": []}}
    with pytest.raises(ActionError, match="not allowlisted"):
        preflight_action_config(config, "read_only_status_probe")

    allowed = {"ops_center": {"action_execution_enabled": True, "allowed_actions": ["read_only_status_probe"]}}
    result = preflight_action_config(allowed, "read_only_status_probe")
    assert result["allowed"] is True
    assert result["would_execute"] is False


def test_approval_preflight_requires_approved_unexpired_matching_record():
    from hermes_cli.ops_actions import preflight_approval_for_action

    action = preflight_approval_for_action(_approved_record(), "read_only_status_probe")

    assert action["allowed"] is True
    assert action["would_execute"] is False
    assert action["action"]["name"] == "read_only_status_probe"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "pending"}, "must be approved"),
        ({"risk_label": "Live-service"}, "risk label"),
        ({"target": "other_action", "proposed_action": "other_action"}, "does not match"),
        ({"expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()}, "expired"),
    ],
)
def test_approval_preflight_rejects_ineligible_records(overrides, message):
    from hermes_cli.ops_actions import ActionError, preflight_approval_for_action

    with pytest.raises(ActionError, match=message):
        preflight_approval_for_action(_approved_record(**overrides), "read_only_status_probe")
