"""Vertex AI capacity 429s get their own backoff ladder; no other provider changes.

Vertex answers a CAPACITY shortage on the shared PayGo pool with HTTP 429
``RESOURCE_EXHAUSTED`` and no ``Retry-After`` header. Google's guidance for those 429s is
exponential backoff with jitter plus smoothed traffic — an immediate retry re-enters the same
contended window, and the generic ladder (2s base) was measured re-429ing on attempts 2 and 3.

The invariant that matters most: every NON-Vertex 429 keeps exactly the wait it had before.
"""

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import retry_utils

VERTEX_BASE = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi"


class _HttpError(Exception):
    """Provider error shaped like the SDK's: an HTTP status plus a JSON body."""

    def __init__(self, status_code: int, body: dict):
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _vertex_capacity_error():
    return _HttpError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Resource exhausted: quota exceeded for the shared PayGo pool",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )


def _openai_429():
    return _HttpError(429, {"error": {"message": "Rate limit reached for gpt-4o", "type": "rate_limit_error"}})


def _agent():
    agent = MagicMock()
    from agent.status_output import StatusOutputMixin
    for name in ("_emit_diagnostic_wait", "_buffer_diagnostic_status"):
        setattr(agent, name, getattr(StatusOutputMixin, name).__get__(agent))
    agent._client_log_context.return_value = ""
    return agent


@pytest.mark.real_retry_backoff
def test_vertex_capacity_429_waits_at_least_15s_on_the_first_retry():
    """The first retry after a Vertex capacity 429 must leave the contended window: the generic
    ladder waited 5.6s and was 429'd again on attempts 2 and 3."""
    from agent.turn_recovery import compute_error_backoff

    wait = compute_error_backoff(
        _agent(), _vertex_capacity_error(), retry_count=1, max_retries=5,
        is_rate_limited=True, is_zai_coding_overload=False,
        base_url=VERTEX_BASE, model="google/gemini-2.5-pro",
    )
    assert wait >= 15.0, f"first Vertex capacity retry waited only {wait}s"


@pytest.mark.real_retry_backoff
def test_vertex_capacity_ladder_is_strictly_increasing():
    """Each further attempt must wait strictly longer than the last — a flat or shrinking
    ladder re-enters the same window."""
    from agent.retry_utils import vertex_capacity_backoff

    waits = [vertex_capacity_backoff(attempt) for attempt in (1, 2, 3, 4)]
    assert all(later > earlier for earlier, later in itertools.pairwise(waits)), waits
    # The ladder tops out at 120s (plus jitter) rather than growing without bound.
    assert waits[-1] <= 120.0 * 1.2


@pytest.mark.real_retry_backoff
def test_non_vertex_429_keeps_the_generic_wait(monkeypatch):
    """A non-Vertex 429 (OpenAI-shaped) is UNCHANGED: the wait is exactly what the generic
    jittered ladder produces, and no Vertex policy is claimed."""
    from agent.turn_recovery import compute_error_backoff

    # Deterministic jitter so the wait can be compared exactly: the generic ladder's
    # base_delay for the first attempt is 2s.
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    agent = _agent()
    wait = compute_error_backoff(
        agent, _openai_429(), retry_count=1, max_retries=3,
        is_rate_limited=True, is_zai_coding_overload=False,
        base_url="https://api.openai.com/v1", model="gpt-4o",
    )
    assert wait == pytest.approx(2.0, abs=0.001)
    status = agent._buffer_status.call_args.args[0]
    assert "Vertex" not in status


@pytest.mark.real_retry_backoff
def test_vertex_429_without_resource_exhausted_text_is_not_treated_as_capacity():
    """The Vertex branch is narrow: a 429 from an aiplatform host that does not say
    RESOURCE_EXHAUSTED keeps the generic wait."""
    from agent.turn_recovery import compute_error_backoff

    err = SimpleNamespace(status_code=429, body={"error": {"message": "Invalid argument: bad prompt"}})
    wait = compute_error_backoff(
        _agent(), err, retry_count=1, max_retries=3,
        is_rate_limited=True, is_zai_coding_overload=False,
        base_url=VERTEX_BASE, model="google/gemini-2.5-pro",
    )
    assert wait < 15.0


def test_vertex_capacity_detection_is_narrow(monkeypatch):
    """Only an aiplatform host + 429 + RESOURCE_EXHAUSTED text qualifies."""
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    from agent.retry_utils import is_vertex_capacity_exhausted, vertex_capacity_backoff

    assert is_vertex_capacity_exhausted(base_url=VERTEX_BASE, error=_vertex_capacity_error())
    # A look-alike host must not qualify (substring on the URL would match it).
    assert not is_vertex_capacity_exhausted(
        base_url="https://aiplatform.googleapis.com.evil.test/v1", error=_vertex_capacity_error())
    assert not is_vertex_capacity_exhausted(base_url="https://api.openai.com/v1", error=_vertex_capacity_error())
    assert not is_vertex_capacity_exhausted(base_url=VERTEX_BASE, error=_openai_429())
    # The ladder values are the contract: 15 / 30 / 60 / 120 seconds.
    assert [vertex_capacity_backoff(a) for a in (1, 2, 3, 4)] == [15.0, 30.0, 60.0, 120.0]
