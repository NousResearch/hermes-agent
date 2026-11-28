"""Configured Codex usage ceilings are read by stable credential ID."""

from agent.usage_admission import policy


def test_codex_pool_usage_limit_is_read_by_credential_id(monkeypatch):
    from agent import credential_pool

    monkeypatch.setattr(
        credential_pool,
        "_load_config_safe",
        lambda: {
            "credential_pool_usage_limits": {
                "openai-codex": {"a1b2c3": 80},
            }
        },
    )

    assert policy.get_pool_usage_limits("openai-codex") == {
        "a1b2c3": 80.0,
    }


def test_codex_pool_usage_limits_ignore_invalid_rows(monkeypatch):
    from agent import credential_pool

    monkeypatch.setattr(
        credential_pool,
        "_load_config_safe",
        lambda: {
            "credential_pool_usage_limits": {
                "openai-codex": {
                    "": 80,
                    "zero": 0,
                    "negative": -1,
                    "too-high": 101,
                    "not-a-number": "nope",
                    "not-finite": float("inf"),
                    "boolean": True,
                    "string-number": "75",
                    "valid": 50.5,
                }
            }
        },
    )

    assert policy.get_pool_usage_limits("openai-codex") == {
        "string-number": 75.0,
        "valid": 50.5,
    }
