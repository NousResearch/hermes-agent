"""Tests for QQ social request persistence."""

from gateway.qq_social_requests import (
    QqSocialRequestStore,
    describe_social_request_state,
    record_social_request_event,
    summarize_social_requests,
    update_social_request_status,
)


def test_record_group_request_persists_pending_request():
    record = record_social_request_event(
        {
            "post_type": "request",
            "request_type": "group",
            "sub_type": "invite",
            "group_id": 987654321,
            "user_id": 179033731,
            "comment": "来群里聊项目",
            "flag": "group-flag-1",
            "time": 1713012345,
        }
    )

    assert record["request_key"] == "group:group-flag-1"
    assert record["request_type"] == "group"
    assert record["sub_type"] == "invite"
    assert record["group_id"] == "987654321"
    assert record["user_id"] == "179033731"
    assert record["status"] == "pending"

    stored = QqSocialRequestStore().get_request("group:group-flag-1")
    assert stored is not None
    assert stored["comment"] == "来群里聊项目"


def test_duplicate_request_event_does_not_clear_handled_status():
    first = record_social_request_event(
        {
            "post_type": "request",
            "request_type": "friend",
            "user_id": 456789,
            "comment": "加个好友",
            "flag": "friend-flag-1",
            "time": 1713012345,
        }
    )
    handled = update_social_request_status(
        first["request_key"],
        status="approved",
        handled_by="test",
        note="已通过",
    )
    repeated = record_social_request_event(
        {
            "post_type": "request",
            "request_type": "friend",
            "user_id": 456789,
            "comment": "加个好友",
            "flag": "friend-flag-1",
            "time": 1713012399,
        }
    )

    assert handled["status"] == "approved"
    assert repeated["status"] == "approved"
    assert repeated["handled_by"] == "test"
    assert repeated["decision_note"] == "已通过"


def test_describe_social_requests_state_exposes_actionability_and_counts():
    pending_group = record_social_request_event(
        {
            "post_type": "request",
            "request_type": "group",
            "sub_type": "add",
            "group_id": 987654321,
            "user_id": 179033731,
            "comment": "想进群看看",
            "flag": "group-flag-state-1",
            "time": 1713012345,
        }
    )
    approved_friend = update_social_request_status(
        record_social_request_event(
            {
                "post_type": "request",
                "request_type": "friend",
                "user_id": 456789,
                "comment": "认识一下",
                "flag": "friend-flag-state-1",
                "time": 1713012401,
            }
        )["request_key"],
        status="approved",
        handled_by="qq_napcat:auto_social_policy",
        handled_via="auto_social_policy",
        note="按社交自动处理策略自动通过好友请求。",
    )

    pending_state = describe_social_request_state(pending_group)
    approved_state = describe_social_request_state(approved_friend)
    summary = summarize_social_requests([pending_group, approved_friend])

    assert pending_state["is_pending"] is True
    assert pending_state["request_kind"] == "group_add_request"
    assert pending_state["available_actions"] == ["approve_request", "reject_request"]
    assert approved_state["handled_automatically"] is True
    assert approved_state["handled_via"] == "auto_social_policy"
    assert approved_state["available_actions"] == []
    assert summary["total"] == 2
    assert summary["actionable"] == 1
    assert summary["by_status"] == {"pending": 1, "approved": 1, "rejected": 0, "ignored": 0}
    assert summary["by_type"] == {"friend": 1, "group": 1}
