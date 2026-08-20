"""Regression test — and exhaustive stress test — for the fix to an
FD-reuse-unsafe hard close that had been reintroduced on the "closed
client" auto-recovery path.

run_agent.py::AIAgent has two places that swap out the shared
``self.client`` (the OpenAI SDK client used by the chat_completions /
OpenAI route):

- ``_replace_primary_openai_client`` (credential rotation) retires the old
  client via ``_retire_shared_openai_client``, which only shuts down the
  pooled sockets and explicitly defers ``client.close()`` / FD release to
  garbage collection. Its own docstring explains why: the shared client
  has no single owning thread, other workers may still be unwinding SSL
  BIOs on it, and hard-closing releases raw FDs the kernel can immediately
  recycle into an unrelated ``open()`` (e.g. ``kanban.db``) — the
  SQLite-header-corruption bug tracked as #29507 / #70773, covered by
  ``tests/run_agent/test_70773_shared_client_fd_corruption.py``.

- ``_ensure_primary_openai_client`` (lazy recovery when a stale/closed
  client is detected before a request) used to call ``_close_openai_client``
  directly on the *old, shared* client instead — the exact hard-close
  pattern the sibling function exists to avoid, and one the #70773 test
  file never exercised.

Fix: ``_ensure_primary_openai_client`` now also retires via
``_retire_shared_openai_client`` instead of hard-closing.
"""

from __future__ import annotations

import threading
from typing import Any

from run_agent import AIAgent


class _FakeOpenAIClient:
    """Not a ``unittest.mock.Mock`` on purpose — AIAgent._is_openai_client_closed
    special-cases ``Mock`` instances to always report "not closed", which
    would defeat this test."""

    def __init__(self, *, is_closed: bool = False) -> None:
        self.is_closed = is_closed
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _bare_agent(*, client: Any) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "test-openai-client-reuse"
    agent.client = client
    agent._client_kwargs = {}
    agent.provider = "openai"
    agent.base_url = "https://api.openai.com/v1"
    agent.model = "gpt-test"
    return agent


class TestEnsurePrimaryClientRetiresSharedClient:
    def test_ensure_primary_does_not_hard_close_the_stale_shared_client(
        self, monkeypatch
    ):
        """Proves the fix: recovering from a closed shared client retires
        it (FD-reuse-safe: shutdown + defer to GC) instead of calling
        client.close() directly, matching _replace_primary_openai_client."""
        old_client = _FakeOpenAIClient(is_closed=True)
        new_client = _FakeOpenAIClient(is_closed=False)
        agent = _bare_agent(client=old_client)

        monkeypatch.setattr(
            agent, "_create_openai_client", lambda *a, **kw: new_client
        )
        monkeypatch.setattr(agent, "_force_close_tcp_sockets", lambda client: 0)

        result = agent._ensure_primary_openai_client(reason="test-recovery")

        assert result is new_client
        assert agent.client is new_client
        assert old_client.close_calls == 0, (
            "a hard close() landed on the old *shared* client — this is "
            "exactly what _retire_shared_openai_client's docstring (and "
            "#29507/#70773) says must never happen for a shared client "
            "with unknown borrowers"
        )

    def test_replace_primary_does_not_hard_close_the_old_shared_client(
        self, monkeypatch
    ):
        """Contrast case: the credential-rotation call site gets this
        right — it must NOT call close() on the old shared client."""
        old_client = _FakeOpenAIClient(is_closed=False)
        new_client = _FakeOpenAIClient(is_closed=False)
        agent = _bare_agent(client=old_client)
        agent._client_lock = threading.RLock()

        monkeypatch.setattr(
            agent, "_create_openai_client", lambda *a, **kw: new_client
        )
        monkeypatch.setattr(agent, "_force_close_tcp_sockets", lambda client: 0)

        ok = agent._replace_primary_openai_client(reason="test-rotation")

        assert ok is True
        assert agent.client is new_client
        assert old_client.close_calls == 0, (
            "credential rotation must defer FD release to GC, not hard-close "
            "a shared client another thread may still be borrowing"
        )


class TestEnsurePrimaryClientExhaustive:
    """Exhaustion / soak test requested alongside the fix: repeats the
    recover-a-closed-shared-client path many times, including with real
    concurrent borrower threads still "reading" the old client (simulating
    an in-flight stream) while recovery runs, to make sure the fix holds
    up under repeated and concurrent pressure — not just a single lucky
    pass. Every agent/client pair is local to its iteration and dropped
    immediately after, so memory stays flat across iterations."""

    ITERATIONS = 300

    def test_no_hard_close_across_many_recoveries_with_concurrent_borrowers(
        self, monkeypatch
    ):
        failures: list[tuple[int, str]] = []

        for i in range(self.ITERATIONS):
            old_client = _FakeOpenAIClient(is_closed=True)
            new_client = _FakeOpenAIClient(is_closed=False)
            agent = _bare_agent(client=old_client)

            monkeypatch.setattr(
                agent, "_create_openai_client", lambda *a, **kw: new_client
            )
            monkeypatch.setattr(
                agent, "_force_close_tcp_sockets", lambda client: 0
            )

            # Hold a real borrower reference across retirement. The start and
            # release barriers guarantee that recovery overlaps the borrower;
            # merely starting a thread here would allow it to finish before
            # _ensure_primary_openai_client runs and would not test the race.
            borrower_started = threading.Event()
            borrower_release = threading.Event()
            borrower_done = threading.Event()

            def _borrower(client=old_client):
                borrower_started.set()
                borrower_release.wait(timeout=5)
                _ = client.is_closed
                borrower_done.set()

            t = threading.Thread(target=_borrower, daemon=True)
            t.start()

            if not borrower_started.wait(timeout=5):
                failures.append((i, "borrower thread never started"))
                borrower_release.set()
                t.join(timeout=5)
                continue

            shutdown_calls = []

            def _shutdown_only(client):
                shutdown_calls.append(client)
                return 0

            monkeypatch.setattr(agent, "_force_close_tcp_sockets", _shutdown_only)
            try:
                result = agent._ensure_primary_openai_client(
                    reason=f"exhaustive-{i}"
                )
            finally:
                borrower_release.set()

            t.join(timeout=5)
            if not borrower_done.is_set():
                failures.append((i, "borrower thread never finished"))
            elif result is not new_client:
                failures.append((i, "did not return the new client"))
            elif shutdown_calls != [old_client]:
                failures.append(
                    (i, f"unexpected shutdown calls: {shutdown_calls!r}")
                )
            elif old_client.close_calls != 0:
                failures.append(
                    (i, f"hard close() called {old_client.close_calls}x")
                )

        assert not failures, (
            f"{len(failures)}/{self.ITERATIONS} iterations failed "
            f"(showing up to 5): {failures[:5]}"
        )
