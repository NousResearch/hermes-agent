"""Regression for #80646: the ``agent_context`` handed to memory providers follows the platform
(``cron`` / ``subagent`` skip writes per the ``MemoryProvider.initialize`` contract) instead of a
hardcoded ``"primary"`` that let cron turns land in stores configured to skip them.
"""

from types import SimpleNamespace

import pytest

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _memory_provider_init_kwargs


def _fake_agent():
    """The attribute surface ``_memory_provider_init_kwargs`` reads."""
    return SimpleNamespace(
        session_id="sess-80646", _session_db=None, _emit_warning=None, _emit_status=None,
        session_cwd=None, **{f"_{name}": None for name in _GATEWAY_IDENTITY_PARAMS},
    )


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("cron", "cron"), ("subagent", "subagent"), ("telegram", "primary"), (None, "primary")],
)
def test_agent_context_follows_the_platform(platform, expected):
    assert _memory_provider_init_kwargs(_fake_agent(), platform)["agent_context"] == expected


def _agent_with_session_source(source):
    """Agent whose session DB reports *source* for the current session."""

    class _DB:
        def get_session(self, session_id):
            return {"id": session_id, "source": source}

    return SimpleNamespace(
        session_id="sess-peer",
        _session_db=_DB(),
        _emit_warning=None,
        _emit_status=None,
        session_cwd=None,
        **{f"_{name}": None for name in _GATEWAY_IDENTITY_PARAMS},
    )


def test_peer_dm_session_is_non_primary():
    """``hermes peer dm`` is agent-to-agent traffic, not the user talking.

    Those sessions are stamped ``source="bot_peer_dm"``. Running them as a
    primary context made providers auto-extract "facts" from another agent's
    instructions and debugging chatter, which is exactly what polluted the
    store on 22/11/26. A non-primary context makes providers skip writes.
    """
    kwargs = _memory_provider_init_kwargs(_agent_with_session_source("bot_peer_dm"), "local")
    assert kwargs["agent_context"] == "peer"


def test_non_peer_session_stays_primary():
    kwargs = _memory_provider_init_kwargs(_agent_with_session_source("desktop"), "local")
    assert kwargs["agent_context"] == "primary"


def test_platform_wins_over_peer_source():
    """An explicit cron/subagent platform keeps its own context."""
    kwargs = _memory_provider_init_kwargs(_agent_with_session_source("bot_peer_dm"), "cron")
    assert kwargs["agent_context"] == "cron"


def test_session_db_failure_fails_soft():
    """A session-lookup failure must not break provider init."""

    class _Boom:
        def get_session(self, session_id):
            raise RuntimeError("state db gone")

    agent = SimpleNamespace(
        session_id="sess-boom",
        _session_db=_Boom(),
        _emit_warning=None,
        _emit_status=None,
        session_cwd=None,
        **{f"_{name}": None for name in _GATEWAY_IDENTITY_PARAMS},
    )
    assert _memory_provider_init_kwargs(agent, "local")["agent_context"] == "primary"
