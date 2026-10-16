"""Opt-in ``fallback.sticky_session`` keeps a fired fallback for the rest of the session (#23094).

Turn-scoped fallback restores the primary at the next turn, so one incident costs two full
prompt-cache rewrites. With the flag on, ``restore_primary_runtime`` leaves the fallback in
place until the session changes. The flag is read from a real ``config.yaml`` in the test's
isolated ``HERMES_HOME`` through the loader the agent uses.
"""

from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import get_hermes_home
from run_agent import AIAgent

_FALLBACK = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}


@pytest.fixture
def sticky_config():
    def write(enabled: bool) -> None:
        (get_hermes_home() / "config.yaml").write_text(
            f"fallback:\n  sticky_session: {'true' if enabled else 'false'}\n", encoding="utf-8")
    return write


def _agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
    ):
        agent = AIAgent(
            api_key="test-key-12345678", base_url="https://my-llm.example.com/v1", provider="custom",
            model="primary-model", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=_FALLBACK,
        )
    agent.client = MagicMock()
    return agent


def _fire_fallback(agent: AIAgent) -> None:
    client = MagicMock(api_key="fallback-key-1234", base_url="https://openrouter.ai/api/v1")
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, None)):
        assert agent._try_activate_fallback() is True
    assert agent.model == _FALLBACK["model"]


def _next_turn_restore(agent: AIAgent) -> bool:
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        return agent._restore_primary_runtime()


def test_sticky_session_keeps_the_fallback_across_turns(sticky_config):
    sticky_config(True)
    agent = _agent()
    _fire_fallback(agent)

    assert _next_turn_restore(agent) is False
    assert _next_turn_restore(agent) is False
    assert (agent.model, agent.provider) == (_FALLBACK["model"], _FALLBACK["provider"])


def test_sticky_session_ends_when_the_session_changes(sticky_config):
    """``/new`` / ``/resume`` on a long-lived agent swap ``session_id``: the new session starts on the primary."""
    sticky_config(True)
    agent = _agent()
    _fire_fallback(agent)
    agent.session_id = "a-new-session"

    assert _next_turn_restore(agent) is True
    assert agent.model == "primary-model"


@pytest.mark.parametrize("configured", [False, None])
def test_fallback_stays_turn_scoped_by_default(sticky_config, configured):
    if configured is not None:
        sticky_config(configured)
    agent = _agent()
    _fire_fallback(agent)

    assert _next_turn_restore(agent) is True
    assert agent.model == "primary-model"


def test_model_once_restore_is_not_held_by_sticky_session(sticky_config):
    """`/model --once` reuses the restore path with ``_fallback_activated``; no provider fallback fired."""
    sticky_config(True)
    agent = _agent()
    agent.model, agent.provider = "temporary-model", "openrouter"
    agent._fallback_activated = True

    assert _next_turn_restore(agent) is True
    assert agent.model == "primary-model"
