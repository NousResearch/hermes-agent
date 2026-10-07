"""z.ai coding-plan quota 429s (error code 1310) must rotate immediately.

The message "Weekly/Monthly Limit Exhausted. Your limit will reset at <ts>" is a
semipermanent account cap, not a transient throttle: retrying the same key once before
rotating just burns a request. The ``limit exhausted`` usage-limit token makes the
rate-limit recovery branch rotate on the FIRST 429.
"""

from types import SimpleNamespace

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential
from agent.error_classifier import FailoverReason

ZAI_1310_MESSAGE = (
    "Error code: 429 - {'error': {'code': '1310', 'message': "
    "'Weekly/Monthly Limit Exhausted. Your limit will reset at 2026-09-10 22:41:42'}}"
)


def _pool_two_keys():
    failed = PooledCredential(
        id="failed", provider="zai", label="GLM_API_KEY", auth_type="api_key",
        priority=0, source="env:GLM_API_KEY", access_token="key-failed",
    )
    healthy = PooledCredential(
        id="healthy", provider="zai", label="ZAI_API_KEY", auth_type="api_key",
        priority=1, source="env:ZAI_API_KEY", access_token="key-healthy",
    )
    return CredentialPool("zai", [failed, healthy]), failed, healthy


def test_zai_limit_exhausted_rotates_on_first_429():
    pool, failed, healthy = _pool_two_keys()
    agent = SimpleNamespace(
        provider="zai", api_key="key-failed", _credential_pool=pool,
        _credential_pool_entry_id="failed",
        _swap_credential=lambda entry: None,
    )
    recovered, has_retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False,
        classified_reason=FailoverReason.rate_limit,
        error_context={"message": ZAI_1310_MESSAGE},
    )
    # Immediate rotation: no same-key retry burned on a weekly/monthly cap.
    assert (recovered, has_retried) == (True, False)
    stored = {entry.id: entry for entry in pool.entries()}
    assert stored["failed"].last_status == "exhausted"
    assert stored["failed"].last_error_code == 429
    assert stored["healthy"].last_status is None


def test_zai_limit_exhausted_marks_failed_key_not_healthy():
    """The exhausted marking must land on the key that served the request, so the
    surviving key stays available for the rotation."""
    pool, failed, healthy = _pool_two_keys()
    agent = SimpleNamespace(
        provider="zai", api_key="key-failed", _credential_pool=pool,
        _credential_pool_entry_id="failed",
        _swap_credential=lambda entry: None,
    )
    recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False,
        classified_reason=FailoverReason.rate_limit,
        error_context={"message": ZAI_1310_MESSAGE},
    )
    stored = {entry.id: entry for entry in pool.entries()}
    assert stored["failed"].last_status == "exhausted"
    assert stored["healthy"].last_status is None
    assert pool.has_available()
