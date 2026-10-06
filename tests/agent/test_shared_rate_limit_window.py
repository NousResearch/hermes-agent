"""Agents sharing one key share its rate-limit window.

A delegate_task fan-out runs its children as threads of one process on the parent's key. When one
child is told to wait out a 429 (Retry-After, or the adaptive long backoff), a sibling that hits the
same limit must not retry on its own 2-4 s schedule inside that window: every early retry re-trips
the limit and spends one of the sibling's few attempts, so the whole batch dies together.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_recovery import compute_error_backoff

BASE_URL = "https://api.z.ai/api/coding/paas/v4"


def _agent(api_key):
    agent = MagicMock()
    agent.api_key = api_key
    agent._client_log_context.return_value = ""
    return agent


def _rate_limit(retry_after=None):
    err = Exception("Error code: 429 - {'error': {'code': '1302', 'message': 'Rate limit reached for requests'}}")
    err.status_code = 429
    err.response = SimpleNamespace(headers={"retry-after": str(retry_after)} if retry_after else {})
    return err


def _wait(agent, err, *, is_rate_limited=True):
    return compute_error_backoff(
        agent, err, retry_count=1, max_retries=3, is_rate_limited=is_rate_limited,
        is_zai_coding_overload=False, base_url=BASE_URL, model="glm-5.3",
    )


def test_sibling_on_the_same_key_waits_out_the_armed_window():
    assert _wait(_agent("key-a"), _rate_limit(retry_after=45)) == 45

    # First retry of a sibling would be ~2-3 s on its own; it joins the 45 s window instead.
    assert 40 <= _wait(_agent("key-a"), _rate_limit()) <= 45
    # Another key is metered separately and keeps its own short schedule.
    assert _wait(_agent("key-b"), _rate_limit()) < 10


def test_non_rate_limit_errors_do_not_join_the_window():
    _wait(_agent("key-a"), _rate_limit(retry_after=45))

    server_error = Exception("HTTP 502 Bad Gateway")
    assert _wait(_agent("key-a"), server_error, is_rate_limited=False) < 10
