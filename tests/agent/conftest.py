"""Fast-path fixtures shared across tests/agent/.

Many tests in this directory exercise the retry/backoff paths in the
agent loop. Production code uses ``jittered_backoff(base_delay=5.0)``
with a ``while time.time() < sleep_end`` loop — a single retry test
spends 5+ seconds of real wall-clock time on backoff waits.

Mocking ``jittered_backoff`` to return 0.0 collapses the while-loop
to a no-op (``time.time() < time.time() + 0`` is false immediately),
which handles the most common case without touching ``time.sleep``.

We deliberately DO NOT mock ``time.sleep`` here — some tests
(test_interrupt_propagation, test_primary_runtime_restore, etc.) use
the real ``time.sleep`` for threading coordination or assert that it
was called with specific values. Tests that want to additionally
fast-path direct ``time.sleep(N)`` calls in production code should
monkeypatch ``run_agent.time.sleep`` locally (see
``test_anthropic_error_handling.py`` for the pattern).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_structured_output_memo(monkeypatch):
    """The aux client remembers routes that rejected ``response_format`` for the whole process;
    a rejection recorded by one test must not strip the field from the next test's request."""
    from agent import auxiliary_structured_output
    monkeypatch.setattr(auxiliary_structured_output, "_REJECTED_ROUTES", set())


@pytest.fixture(autouse=True)
def _fast_retry_backoff(request, monkeypatch):
    """Short-circuit retry backoff for all tests in this directory.

    Tests that assert on the real backoff value opt out with
    ``@pytest.mark.real_retry_backoff``.
    """
    if request.node.get_closest_marker("real_retry_backoff"):
        return
    # The agent.turn_* retry paths import ``jittered_backoff`` lazily from
    # ``agent.retry_utils``; patch it there so rate-limit / invalid-response /
    # server-error retries don't burn real wall-clock seconds.
    from agent import retry_utils as _retry_utils
    monkeypatch.setattr(_retry_utils, "jittered_backoff", lambda *a, **k: 0.0)
