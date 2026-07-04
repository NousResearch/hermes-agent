from run_agent import AIAgent


def _roles(messages):
    return [message["role"] for message in messages]


def test_injects_user_after_system_when_absent():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "assistant", "content": "A"},
        {"role": "tool", "content": "T"},
    ]

    AIAgent._ensure_user_anchor(messages)

    assert _roles(messages)[:2] == ["system", "user"]


def test_noop_when_user_already_present():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    before = [dict(message) for message in messages]

    AIAgent._ensure_user_anchor(messages)

    assert messages == before


def test_injects_at_head_when_no_system():
    messages = [
        {"role": "assistant", "content": "A"},
        {"role": "tool", "content": "T"},
    ]

    AIAgent._ensure_user_anchor(messages)

    assert _roles(messages)[0] == "user"


def test_empty_list_is_noop():
    messages = []

    AIAgent._ensure_user_anchor(messages)

    assert messages == []


def test_anchor_content_is_normalized_parts():
    messages = [{"role": "tool", "content": "T"}]

    AIAgent._ensure_user_anchor(messages)

    user = next(message for message in messages if message["role"] == "user")
    assert isinstance(user["content"], list)
    assert user["content"][0]["type"] == "text"
