"""Auxiliary feasibility must respect a context engine's own trigger policy."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine
from agent.conversation_compression import check_compression_model_feasibility


class ChunkedContextEngine(ContextEngine):
    name = "chunked"
    context_length = 1_000_000
    threshold_tokens = 750_000
    threshold_percent = 0.75

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage["prompt_tokens"]

    def should_compress(self, prompt_tokens=None):
        return (prompt_tokens or self.last_prompt_tokens) >= self.threshold_tokens

    def compress(self, messages, **kwargs):
        return messages


@pytest.mark.parametrize("builtin", [False, True], ids=["plugin", "builtin"])
def test_aux_window_only_clamps_builtin_engine_and_preserves_notice(builtin):
    engine = (
        ContextCompressor(
            "main-model", config_context_length=1_000_000,
            threshold_percent=0.75, quiet_mode=True,
        )
        if builtin else ChunkedContextEngine()
    )
    notices = []
    agent = SimpleNamespace(
        compression_enabled=True, context_compressor=engine,
        model="main-model", provider="openai", _custom_providers=[],
        _aux_compression_context_length_config=272_000,
        _current_main_runtime=dict, _emit_diagnostic_status=notices.append,
    )
    client = SimpleNamespace(base_url="https://example.com/v1", api_key="")
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("openai", "aux-model", None, None, None),
    ), patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(client, "aux-model"),
    ), patch("agent.model_metadata.get_model_context_length", return_value=272_000):
        check_compression_model_feasibility(agent)

    assert len(notices) == 1
    assert agent._compression_warning == notices[0]
    assert "272,000" in notices[0] and "750,000" in notices[0]
    assert "aux-model" in notices[0]
    from gateway.run import _TELEGRAM_NOISY_STATUS_RE
    assert _TELEGRAM_NOISY_STATUS_RE.search(notices[0])  # stays in logs, out of gateway chats, as on main
    assert engine.threshold_tokens == (272_000 if builtin else 750_000)
    assert engine.threshold_percent == (0.272 if builtin else 0.75)
    if builtin:
        assert engine._aux_context_ceiling == 272_000
        assert "Auto-lowered this session's threshold to 272,000" in notices[0]
    else:
        assert not hasattr(engine, "_aux_context_ceiling")
        assert not engine.should_compress(300_000)
