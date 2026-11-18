"""A Nous 503 ``usage_window_unavailable`` is a usage-check outage, not a bad credential:
credential-pool recovery must neither rotate nor mark (exhaust/bench) the credential for it,
and the eager fallback must not wait on a pool rotation that cannot help."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.error_classifier import classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


_BODY = {"error": {"code": "usage_window_unavailable",
                   "message": "Hermes usage checks are temporarily unavailable"}}


def _classified_503():
    err = _APIError("Hermes usage checks are temporarily unavailable", status_code=503, body=_BODY)
    return classify_api_error(err, provider="nous", base_url="https://inference-api.nousresearch.com/v1",
                              model="Hermes-4-405B")


class _Pool:
    provider = "nous"

    def __init__(self):
        self.mark_calls: list[dict] = []
        self.entry = SimpleNamespace(id="cred-1", runtime_api_key="k1", last_status=None, priority=0)

    def current(self):
        return self.entry

    def entries(self):
        return [self.entry]

    def mark_exhausted_and_rotate(self, **kwargs):
        self.mark_calls.append(kwargs)
        return SimpleNamespace(id="cred-2", priority=1)


def _agent(pool):
    return SimpleNamespace(
        provider="nous", base_url="https://inference-api.nousresearch.com/v1", model="Hermes-4-405B",
        api_key="k1", _credential_pool=pool, _credential_pool_entry_id="cred-1",
        _swap_credential=MagicMock(), _credential_pool_revert_id=None,
    )


def test_classifier_tags_the_503_as_a_usage_window_outage():
    c = _classified_503()
    assert c.error_context.get("usage_window_outage") is True


def test_pool_context_keeps_the_outage_flag():
    from agent.turn_recovery import _pool_error_context
    ctx = _pool_error_context(_agent(_Pool()), _classified_503(), {})
    assert ctx.get("usage_window_outage") is True


def test_503_usage_window_never_rotates_or_marks_the_credential():
    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.turn_recovery import _pool_error_context

    pool = _Pool()
    agent = _agent(pool)
    c = _classified_503()
    ctx = _pool_error_context(agent, c, {})
    has_retried = False
    for _ in range(3):  # first attempt, then the "retried once" path that would rotate
        recovered, has_retried = recover_with_credential_pool(
            agent, status_code=503, has_retried_429=has_retried,
            classified_reason=c.reason, error_context=ctx,
        )
        assert recovered is False
        has_retried = True
    assert pool.mark_calls == []
    agent._swap_credential.assert_not_called()


def test_503_usage_window_eager_fallback_does_not_wait_on_pool(monkeypatch):
    import agent.conversation_loop as loop
    from agent.turn_recovery import _pool_may_help

    # A pool that *would* claim it can recover: the outage must still not wait on it.
    monkeypatch.setattr(loop, "_ra", lambda: SimpleNamespace(_pool_may_recover_from_rate_limit=lambda p: True))
    assert _pool_may_help(_agent(_Pool()), _classified_503()) is False
