"""Tests for QQ social auto-handling policy storage."""

from gateway.qq_social_policy import (
    clear_social_policy,
    describe_social_policy_state,
    get_social_policy,
    set_social_policy,
)


def test_default_social_policy_starts_disabled():
    policy = get_social_policy()

    assert policy["auto_approve_friend_requests"] is False
    assert policy["auto_approve_group_add_requests"] is False
    assert policy["auto_approve_group_invites"] is False
    assert policy["notify_target"] is None


def test_set_and_clear_social_policy_round_trip():
    updated = set_social_policy(
        auto_approve_friend_requests=True,
        auto_approve_group_add_requests=True,
        auto_approve_group_invites=True,
        notify_target="qq_napcat:dm:179033731",
        notes="test social policy",
        updated_by="tester",
    )

    assert updated["auto_approve_friend_requests"] is True
    assert updated["auto_approve_group_add_requests"] is True
    assert updated["auto_approve_group_invites"] is True
    assert updated["notify_target"] == "qq_napcat:dm:179033731"
    assert updated["notes"] == "test social policy"
    assert updated["updated_by"] == "tester"

    cleared = clear_social_policy(updated_by="tester")

    assert cleared["auto_approve_friend_requests"] is False
    assert cleared["auto_approve_group_add_requests"] is False
    assert cleared["auto_approve_group_invites"] is False
    assert cleared["notify_target"] is None
    assert cleared["notes"] == ""
    assert cleared["updated_by"] == "tester"


def test_describe_social_policy_state_reports_enabled_scopes_and_notify_status():
    policy = set_social_policy(
        auto_approve_friend_requests=True,
        auto_approve_group_invites=True,
        notify_target="qq_napcat:dm:179033731",
        notes="ops can keep an eye on auto-handled requests",
        updated_by="tester",
    )

    state = describe_social_policy_state(policy)

    assert state["auto_approval_enabled"] is True
    assert state["enabled_scope_count"] == 2
    assert state["enabled_scopes"] == ["friend_requests", "group_invites"]
    assert state["notify_configured"] is True
    assert state["notify_target"] == "qq_napcat:dm:179033731"
    assert state["updated_by"] == "tester"
