"""Focused tests for the startup context-length floor."""

from types import SimpleNamespace

import pytest

from agent import agent_init
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH


@pytest.fixture
def low_context_agent():
    return SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=32_768),
        _ollama_num_ctx=None,
        base_url="http://127.0.0.1:11434/v1",
        provider="ollama",
        _config_context_length=32_768,
        model="qwen2.5:7b",
    )


def test_below_floor_still_raises_without_override(low_context_agent):
    with pytest.raises(ValueError, match="below the minimum"):
        agent_init._enforce_minimum_context(low_context_agent)


def test_explicit_override_allows_low_context_and_warns(low_context_agent, caplog):
    low_context_agent.allow_context_below_minimum = True

    with caplog.at_level("WARNING"):
        agent_init._enforce_minimum_context(low_context_agent)

    assert any("below the recommended" in record.message for record in caplog.records)
    assert any("tool use may be unreliable" in record.message for record in caplog.records)


def test_override_requires_explicit_context_length(low_context_agent):
    low_context_agent.allow_context_below_minimum = True
    low_context_agent._config_context_length = None

    with pytest.raises(ValueError, match="below the minimum"):
        agent_init._enforce_minimum_context(low_context_agent)


def test_lmstudio_explicit_low_context_remains_allowed(low_context_agent):
    low_context_agent.provider = "lmstudio"
    agent_init._enforce_minimum_context(low_context_agent)


@pytest.mark.parametrize("override", [False, True])
def test_context_at_floor_is_unaffected(low_context_agent, override):
    low_context_agent.context_compressor.context_length = MINIMUM_CONTEXT_LENGTH
    low_context_agent.allow_context_below_minimum = override
    agent_init._enforce_minimum_context(low_context_agent)
