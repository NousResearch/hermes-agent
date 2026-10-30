"""Operator-tunable rate-limit patience: ``agent.max_retry_wait_seconds`` bounds every single retry
wait, and ``agent.rotate_credential_on_first_rate_limit`` rotates a multi-credential pool on the
first 429 instead of retrying the same quota-walled credential once. Both default to today's behavior.
"""
import json
import types
from unittest.mock import MagicMock

import pytest

from agent.agent_init import normalize_max_retry_wait
from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.error_classifier import FailoverReason
from agent.turn_recovery import compute_error_backoff

TOKENS = ("tok-account-a", "tok-account-b")


@pytest.mark.parametrize("raw, expected", [
    (None, None), (10, 10.0), ("10", 10.0), (0, 0.0), (2.5, 2.5),
    (-1, None), ("soon", None), (True, None), ([], None),
])
def test_max_retry_wait_normalizes_to_non_negative_seconds_or_none(raw, expected):
    assert normalize_max_retry_wait(raw) == expected


def _agent(**attrs):
    agent = MagicMock()
    from agent.status_output import StatusOutputMixin
    for name in ("_emit_diagnostic_wait", "_buffer_diagnostic_status"):
        setattr(agent, name, getattr(StatusOutputMixin, name).__get__(agent))
    agent._client_log_context.return_value = ""
    for name, value in attrs.items():
        setattr(agent, name, value)
    return agent


def _backoff(agent, monkeypatch, retry_after):
    monkeypatch.setattr("agent.retry_utils.provider_retry_after_seconds", lambda err: retry_after)
    return compute_error_backoff(
        agent, Exception("HTTP 429"), retry_count=1, max_retries=3, is_rate_limited=True,
        is_zai_coding_overload=False, base_url="https://example.test/v1", model="test/model",
    )


def test_max_retry_wait_bounds_a_long_retry_after(monkeypatch):
    # Unset (an attribute that is not a number, as on any agent built before the setting): built-in 600s cap.
    assert _backoff(_agent(), monkeypatch, 3600.0) == 600.0
    assert _backoff(_agent(_max_retry_wait_s=None), monkeypatch, 3600.0) == 600.0
    # Set: no single wait exceeds it, Retry-After included.
    assert _backoff(_agent(_max_retry_wait_s=10.0), monkeypatch, 600.0) == 10.0
    # Shorter waits are untouched.
    assert _backoff(_agent(_max_retry_wait_s=10.0), monkeypatch, 3.0) == 3.0


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": f"cred-{i}", "label": f"acct-{i}", "auth_type": "oauth", "priority": i, "source": "manual",
         "access_token": tok, "refresh_token": f"rt-{i}", "expires_at_ms": 4_000_000_000_000}
        for i, tok in enumerate(TOKENS)
    ]}}))
    from agent.credential_pool import load_pool
    return load_pool("openai-codex")


def _pool_agent(pool, **attrs):
    return types.SimpleNamespace(
        provider="openai-codex", model="gpt-5.3-codex", base_url="https://chatgpt.com/backend-api/codex",
        api_key=TOKENS[0], _credential_pool=pool, _credential_pool_entry_id="cred-0",
        _swap_credential=MagicMock(return_value=True), **attrs,
    )


def test_first_429_retries_the_same_credential_by_default(pool):
    agent = _pool_agent(pool)
    assert recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, classified_reason=FailoverReason.rate_limit,
    ) == (False, True)
    agent._swap_credential.assert_not_called()


def test_rotate_on_first_rate_limit_moves_to_the_next_credential_at_once(pool):
    agent = _pool_agent(pool, _rotate_on_first_429=True)
    assert recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, classified_reason=FailoverReason.rate_limit,
    ) == (True, False)
    agent._swap_credential.assert_called_once()
    assert agent._swap_credential.call_args.args[0].id == "cred-1"
