"""Enterprise permissions fail closed on missing scope, malformed grants and unsupported chats."""

from gateway.enterprise_permissions import allows, allows_inbound


def _policy(*grants):
    return {"enterprise_permissions": {"enabled": True, "grants": list(grants)}}


def _grant(scope, kind, resource, *, read=False, write=False):
    return {"scope_id": scope, "kind": kind, "id": resource, "read": read, "write": write}


def test_opt_in_preserves_existing_behavior():
    assert allows_inbound({}, platform="slack", scope_id="", user_id="", chat_id="", chat_kind="channel")


def test_slack_sender_and_channel_must_both_be_granted():
    policy = _policy(_grant("T1", "user", "U1", read=True), _grant("T1", "channel", "C1", read=True))
    assert allows_inbound(policy, platform="slack", scope_id="T1", user_id="U1", chat_id="C1", chat_kind="channel")
    assert not allows_inbound(policy, platform="slack", scope_id="T1", user_id="U2", chat_id="C1", chat_kind="channel")
    assert not allows_inbound(policy, platform="slack", scope_id="T2", user_id="U1", chat_id="C1", chat_kind="channel")
    assert not allows_inbound(policy, platform="slack", scope_id="T1", user_id="U1", chat_id="C2", chat_kind="channel")
    assert not allows_inbound(policy, platform="slack", scope_id="T1", user_id="U1", chat_id="G1", chat_kind="unsupported")


def test_teams_personal_group_and_channel_are_distinct():
    policy = _policy(_grant("tenant", "user", "aad-user", read=True),
                     _grant("tenant", "group_chat", "conversation", read=True))
    assert allows_inbound(policy, platform="teams", scope_id="tenant", user_id="aad-user",
                          chat_id="personal", chat_kind="dm")
    assert allows_inbound(policy, platform="teams", scope_id="tenant", user_id="aad-user",
                          chat_id="conversation", chat_kind="group_chat")
    assert not allows_inbound(policy, platform="teams", scope_id="tenant", user_id="aad-user",
                              chat_id="conversation", chat_kind="channel")


def test_read_does_not_grant_write_and_bad_config_denies():
    policy = _policy(_grant("T1", "channel", "C1", read=True))
    assert not allows(policy, platform="slack", scope_id="T1", kind="channel", resource_id="C1", action="write")
    assert not allows({"enterprise_permissions": {"enabled": True, "grants": "all"}},
                      platform="slack", scope_id="T1", kind="channel", resource_id="C1", action="read")
    assert not allows({"enterprise_permissions": {"enabled": "true", "grants": [
        _grant("T1", "channel", "C1", read=True)]}}, platform="slack", scope_id="T1",
        kind="channel", resource_id="C1", action="read")
    assert not allows(policy, platform="slack", scope_id="", kind="channel", resource_id="C1", action="read")
