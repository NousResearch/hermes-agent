"""#107475: ``AIAgent.close()`` must not hard-close the shared OpenAI client while a request is in flight.

``_drop_shared_client`` handed the shared client straight to ``_close_openai_client(shared=True)`` — the
FD-releasing close — even though the per-request teardown already refuses to do that while a worker still
owns the pool (its ``in_use`` flag). A scheduled cron turn and an auxiliary context-compression request
overlapping left the in-flight request mute on a closed pool.

Fix: ``close()`` goes through ``_close_shared_openai_client``, which retires (socket ``shutdown()``, FDs
left to GC/the owning worker) while a request is in flight and hard-closes only when the agent is idle.
"""
import threading

import pytest

import run_agent


class _FakeSharedClient:
    """Stand-in for the shared OpenAI/httpx client; only ``close()`` is observable in this path."""

    def __init__(self):
        self.is_closed = False
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        self.is_closed = True


def _build_agent(client):
    """Agent skeleton whose only live resource is the shared client (no sessions/processes/memory)."""
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.provider = "openai"
    agent.model = "gpt-5"
    agent.base_url = "https://api.openai.com/v1"
    agent.quiet_mode = True
    agent.client = client
    agent._model_request_active = threading.Event()
    return agent


@pytest.fixture
def retired(monkeypatch):
    """Record retirement calls instead of touching real sockets."""
    calls = []
    monkeypatch.setattr(
        run_agent.AIAgent,
        "_retire_shared_openai_client",
        lambda self, client, *, reason: calls.append((client, reason)),
    )
    return calls


def test_close_retires_shared_client_while_turn_request_in_flight(retired):
    """The turn path brackets its provider call with ``_model_request_active``."""
    client = _FakeSharedClient()
    agent = _build_agent(client)
    agent._model_request_active.set()

    agent.close()

    assert client.close_calls == 0, "in-flight request still owns the pool's FDs"
    assert [c for c, _ in retired] == [client]
    assert retired[0][1] == "agent_close_in_flight"
    assert agent.client is None


def test_close_retires_shared_client_while_inline_request_registered(retired):
    """The inline cron/delegated path keeps ``_active_request_abort`` registered for the whole call."""
    client = _FakeSharedClient()
    agent = _build_agent(client)
    agent._active_request_abort = lambda reason: None

    agent.close()

    assert client.close_calls == 0, "in-flight request still owns the pool's FDs"
    assert retired[0][1] == "agent_close_in_flight"
    assert agent.client is None


def test_close_hard_closes_shared_client_when_idle(retired):
    """Idle agent: close() keeps its hard-close semantics (it is the resource-owning boundary)."""
    client = _FakeSharedClient()
    agent = _build_agent(client)

    agent.close()

    assert client.close_calls == 1
    assert retired == []
    assert agent.client is None
