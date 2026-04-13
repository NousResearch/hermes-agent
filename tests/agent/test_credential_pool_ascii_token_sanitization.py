from agent.credential_pool import PooledCredential


def test_from_dict_strips_non_ascii_from_access_token():
    entry = PooledCredential.from_dict(
        "copilot",
        {
            "id": "abc123",
            "label": "test",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual",
            "access_token": "tok123│tail",
        },
    )
    assert entry.access_token == "tok123tail"


def test_from_dict_strips_non_ascii_from_refresh_and_agent_key():
    entry = PooledCredential.from_dict(
        "nous",
        {
            "id": "abc123",
            "label": "test",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual",
            "access_token": "access",
            "refresh_token": "refresh│token",
            "agent_key": "agent│key",
        },
    )
    assert entry.refresh_token == "refreshtoken"
    assert entry.agent_key == "agentkey"


def test_from_dict_removes_internal_whitespace_from_tokens():
    entry = PooledCredential.from_dict(
        "copilot",
        {
            "id": "abc123",
            "label": "test",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual",
            "access_token": "tok en\twith\nspaces",
        },
    )
    assert entry.access_token == "tokenwithspaces"
