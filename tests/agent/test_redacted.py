import logging

import pytest

from agent.credential_pool import PooledCredential
from agent.redacted import Redacted


def test_redacted_hides_value_in_logging_and_repr(caplog):
    secret = Redacted.make("sk-test-super-secret-value")

    assert str(secret) == "<redacted>"
    assert repr(secret) == "<redacted>"
    assert f"{secret}" == "<redacted>"
    assert Redacted.value(secret) == "sk-test-super-secret-value"

    logger = logging.getLogger("tests.redacted")
    with caplog.at_level(logging.INFO, logger="tests.redacted"):
        logger.info("credential=%s repr=%r", secret, secret)

    assert "<redacted>" in caplog.text
    assert "sk-test-super-secret-value" not in caplog.text


def test_redacted_value_rejects_non_redacted_object():
    with pytest.raises(ValueError, match="not in registry"):
        Redacted.value(object())  # type: ignore[arg-type]


def test_pooled_credential_repr_does_not_include_tokens():
    credential = PooledCredential(
        provider="openai",
        id="cred-1",
        label="prod",
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-live-super-secret-token",
        refresh_token="refresh-super-secret-token",
        agent_key="agent-super-secret-token",
        extra={"api_key": "extra-super-secret-token", "scope": "read"},
    )

    rendered = repr(credential)

    assert "sk-live-super-secret-token" not in rendered
    assert "refresh-super-secret-token" not in rendered
    assert "agent-super-secret-token" not in rendered
    assert "extra-super-secret-token" not in rendered
    assert "access_token='<redacted>'" in rendered
    assert "refresh_token='<redacted>'" in rendered
    assert "agent_key='<redacted>'" in rendered
    assert "'api_key': '<redacted>'" in rendered
    assert "'scope': 'read'" in rendered
