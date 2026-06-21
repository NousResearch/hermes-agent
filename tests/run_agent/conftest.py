"""Fast-path fixtures shared across tests/run_agent/.

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
def _fast_retry_backoff(monkeypatch):
    """Short-circuit retry backoff for all tests in this directory."""
    try:
        import run_agent
    except ImportError:
        return

    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)
    # The conversation loop was extracted out of run_agent.py into
    # ``agent.conversation_loop``, which imports ``jittered_backoff``
    # directly (``from agent.retry_utils import jittered_backoff``).
    # Patching ``run_agent.jittered_backoff`` alone misses every retry
    # path under the new module — tests that exercise rate-limit /
    # invalid-response / server-error retries burn real wall-clock
    # seconds per retry. Patch both for full coverage.
    try:
        from agent import conversation_loop as _conv_loop
        monkeypatch.setattr(_conv_loop, "jittered_backoff", lambda *a, **k: 0.0)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Shared AIAgent fixtures (moved here from the former monolithic
# test_run_agent.py when it was split into per-theme files). Fixtures in a
# conftest auto-inject into every test module in this directory by name.
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock, patch  # noqa: E402

from run_agent import AIAgent  # noqa: E402

from tests.run_agent._run_agent_helpers import _make_tool_defs  # noqa: E402,F401


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


@pytest.fixture()
def agent_with_memory_tool():
    """Agent whose valid_tool_names includes 'memory'."""
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search", "memory"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-k...7890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a
