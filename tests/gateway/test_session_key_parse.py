from gateway.run import _parse_session_key


def test_parse_session_key_accepts_profile_namespace():
    assert _parse_session_key("agent:coder:discord:group:c1:u1") == {
        "namespace": "coder",
        "platform": "discord",
        "chat_type": "group",
        "chat_id": "c1",
        "participant_id": "u1",
    }


def test_parse_session_key_decodes_scope_segment():
    assert _parse_session_key("agent:main:discord:group:scope=guild-a:c1:u1") == {
        "namespace": "main",
        "platform": "discord",
        "chat_type": "group",
        "scope_id": "guild-a",
        "chat_id": "c1",
        "participant_id": "u1",
    }
