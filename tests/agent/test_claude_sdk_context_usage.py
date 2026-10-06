"""Ground-truth context usage on the claude-agent-sdk lane.

Hermes' own estimate is unusable here: api_messages holds full tool payloads
that are never sent to the CLI, so it over-reports by an order of magnitude.
``ClaudeAgentSdkSession.context_usage()`` asks the CLI instead.

It feeds status lines and compaction heuristics, so every failure mode must
degrade to None rather than raise into a status path.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.transports import claude_agent_sdk_session as M


USAGE = {
    "totalTokens": 111_000,
    "maxTokens": 170_000,
    "contextWindow": 200_000,
    "percentage": 55.5,
    "model": "claude-opus-5",
    "isAutoCompactEnabled": True,
}


@pytest.fixture
def session():
    """A session with a live loop thread and no real CLI behind it."""
    loops = []

    def _build(client):
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        loops.append(loop)
        s = M.ClaudeAgentSdkSession.__new__(M.ClaudeAgentSdkSession)
        s._client = client
        s._loop = loop
        s._loop_thread = thread
        return s

    yield _build
    for loop in loops:
        loop.call_soon_threadsafe(loop.stop)


class _Client:
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises

    async def get_context_usage(self):
        if self._raises is not None:
            raise self._raises
        return self._result


def test_returns_the_cli_reported_usage(session):
    s = session(_Client(result=USAGE))

    usage = s.context_usage()

    assert usage == USAGE
    assert usage["totalTokens"] == 111_000
    assert usage["isAutoCompactEnabled"] is True


def test_none_when_no_client(session):
    s = session(_Client(result=USAGE))
    s._client = None

    assert s.context_usage() is None


def test_none_when_no_loop(session):
    s = session(_Client(result=USAGE))
    s._loop = None

    assert s.context_usage() is None


def test_none_on_older_sdk_without_the_method(session):
    class _Old:
        pass

    s = session(_Old())

    assert s.context_usage() is None


def test_none_when_the_query_raises(session):
    s = session(_Client(raises=RuntimeError("transport closed")))

    assert s.context_usage() is None


def test_none_when_the_result_is_not_a_mapping(session):
    s = session(_Client(result=["not", "a", "dict"]))

    assert s.context_usage() is None
