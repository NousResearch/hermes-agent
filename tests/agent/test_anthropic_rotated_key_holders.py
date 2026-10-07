"""A rotated Anthropic key must reach every holder that copied the old one (#121756).

turn_context publishes the auxiliary main runtime at turn start; the credential refresh runs later
(pre-dispatch or after a 401), possibly in a worker context copied from the turn's. ContextCompressor
keeps its own key from startup. A pool-bound agent rotates through the pool and must never adopt the
singleton token, which may belong to another account.
"""

from __future__ import annotations

import contextvars
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux
from agent.turn_context import _publish_runtime_main
from run_agent import AIAgent

OLD, NEW = "sk-ant-oat01-old", "sk-ant-oat01-new"


@pytest.fixture
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    a.api_mode, a.provider, a.model = "anthropic_messages", "anthropic", "claude-opus-4-6"
    a.base_url = a._anthropic_base_url = "https://api.anthropic.com"
    a.api_key = a._anthropic_api_key = OLD
    a._anthropic_client, a._is_anthropic_oauth = MagicMock(), True
    a._credential_pool, a._credential_pool_entry_id = None, None
    a.context_compressor = SimpleNamespace(api_key=OLD)
    yield a
    aux.clear_runtime_main()


def _rotate(agent, how: str) -> None:
    with (
        patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=NEW),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        if how == "refresh":
            assert agent._try_refresh_anthropic_client_credentials() is True
        else:
            agent._credential_pool, agent._credential_pool_entry_id = MagicMock(), "entry-1"
            entry = SimpleNamespace(id="entry-2", runtime_api_key=NEW, runtime_base_url="https://api.anthropic.com")
            assert agent._swap_credential(entry) is True


@pytest.mark.parametrize("how", ["refresh", "pool-rotation"])
def test_mid_turn_rotation_reaches_every_key_holder(agent, how):
    _publish_runtime_main(agent)                                   # turn start
    contextvars.copy_context().run(_rotate, agent, how)           # the request's worker context

    assert agent.api_key == NEW
    assert agent.context_compressor.api_key == NEW
    assert aux._normalize_main_runtime(None)["api_key"] == NEW   # what `auto` aux calls route with


def test_pool_bound_agent_never_adopts_the_singleton_token(agent):
    agent._credential_pool, agent._credential_pool_entry_id = MagicMock(), "entry-1"
    _publish_runtime_main(agent)

    with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=NEW):
        assert agent._try_refresh_anthropic_client_credentials() is False

    assert (agent._anthropic_api_key, agent.api_key, agent._credential_pool_entry_id) == (OLD, OLD, "entry-1")
    assert aux._normalize_main_runtime(None)["api_key"] == OLD
