"""A plugin context engine owns its own compaction trigger.

The auxiliary-compression feasibility probe lowers the live threshold to the
auxiliary model's context window, because the built-in compressor summarises a
window with that model. A plugin engine (llmslim) computes its own budget from
``context_length * target_percent`` and never calls the auxiliary model at all.

Lowering such an engine's trigger fires it before there is anything to
compress: the request is dominated by the system prompt and the tool schemas,
the message body is still tiny, and ``compress()`` returns the transcript
unchanged. The host then records ``no_progress`` and aborts the compaction on
every turn (#113737).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import conversation_compression as cc
from agent.context_compressor import ContextCompressor


class _PluginContextEngine:
    """Stand-in for a plugin context engine (llmslim): owns its own trigger and budget."""

    name = "llmslim"

    def __init__(self, *, context_length: int = 1_000_000, threshold_percent: float = 0.65) -> None:
        self.context_length = context_length
        self.threshold_percent = threshold_percent
        self.threshold_tokens = int(context_length * threshold_percent)
        self.target_percent = 0.15


def _make_agent(*, main_context: int = 1_000_000) -> SimpleNamespace:
    agent = SimpleNamespace()
    agent.model = "test-main-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-test"
    agent.api_mode = "chat_completions"
    agent.compression_enabled = True
    agent.session_id = "session-test"
    agent._compression_warning = None
    agent._last_feasibility_notice = None
    agent._aux_compression_context_length_config = None
    agent._custom_providers = []
    agent._emit_status = lambda message: None
    agent._emit_diagnostic_status = lambda message: None
    agent._current_main_runtime = lambda: None
    agent.context_compressor = MagicMock(spec=ContextCompressor)
    agent.context_compressor.context_length = main_context
    return agent


def test_plugin_context_engine_trigger_is_not_lowered_to_the_aux_window():
    """The aux window must not become a plugin engine's trigger: it has no aux summariser."""
    agent = _make_agent()
    engine = agent.context_compressor = _PluginContextEngine()
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)), \
         patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", return_value=80_000):
        cc.check_compression_model_feasibility(agent)
    assert engine.threshold_tokens == 650_000
    assert engine.threshold_percent == 0.65


def test_builtin_compressor_still_gets_the_aux_window_lowering():
    """The built-in compressor summarises with the aux model, so it keeps the lowering."""
    agent = _make_agent(main_context=200_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=200_000, threshold_percent=0.50, quiet_mode=True,
    )
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)), \
         patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", return_value=80_000):
        cc.check_compression_model_feasibility(agent)
    assert compressor.threshold_tokens == 80_000
