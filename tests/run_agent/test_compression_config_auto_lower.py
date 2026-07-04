"""Regression: auto-lower must skip when context_length is user-configured.

Issue #58407: ``auxiliary.compression.context_length`` in config.yaml set an
input budget for the summarizer, but the auto-lower logic treated it as a
capability limit and lowered the session threshold — causing near-useless
compression loops on long sessions.

When the aux context value originates from the user's explicit config
(``_aux_compression_context_length_config is not None``), the auto-lower
should be skipped.
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


@pytest.fixture(autouse=True)
def _stable_aux_provider_config():
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("auto", None, None, None, None),
    ):
        yield


def _make_agent(
    *,
    compression_enabled: bool = True,
    threshold_percent: float = 0.50,
    main_context: int = 800_000,
    aux_context_config: int | None = None,
) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model = "test-main-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-test"
    agent.api_mode = "chat_completions"
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.compression_enabled = compression_enabled
    agent._print_fn = None
    agent.suppress_status_output = False
    agent._stream_consumers = []
    agent._executing_tools = False
    agent._mute_post_response = False
    agent.status_callback = None
    agent.tool_progress_callback = None
    agent._compression_warning = None
    agent._aux_compression_context_length_config = aux_context_config
    agent._custom_providers = []
    agent.tools = []

    compressor = MagicMock(spec=ContextCompressor)
    compressor.context_length = main_context
    compressor.threshold_tokens = int(main_context * threshold_percent)
    agent.context_compressor = compressor

    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_auto_lower_skipped_when_context_length_from_config(mock_get_client, mock_ctx_len):
    """Issue #58407: user-configured context_length is an input budget,
    not a capability limit.  Auto-lower must NOT engage."""
    # Main model: 800K context, threshold at 50% = 400K
    # Aux model: 128K context (returned by mock)
    # Config: auxiliary.compression.context_length = 128000
    agent = _make_agent(
        main_context=800_000,
        threshold_percent=0.50,
        aux_context_config=128_000,
    )
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "deepseek/deepseek-v4-flash")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    # Threshold must NOT be lowered — the 128K came from user config
    assert agent.context_compressor.threshold_tokens == 400_000
    # No "Auto-lowered" message
    assert not any("Auto-lowered" in m for m in messages)


@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_auto_lower_still_engages_when_no_config_override(mock_get_client, mock_ctx_len):
    """Without explicit config, auto-lower should still work (existing behavior)."""
    agent = _make_agent(
        main_context=800_000,
        threshold_percent=0.50,
        aux_context_config=None,  # no user config
    )
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "deepseek/deepseek-v4-flash")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    # Threshold IS lowered — aux model's detected limit is below threshold
    assert agent.context_compressor.threshold_tokens == 128_000
    assert any("Auto-lowered" in m for m in messages)
