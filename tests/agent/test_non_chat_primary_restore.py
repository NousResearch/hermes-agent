"""A primary already known to be non-chat must not be restored on the next turn."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent

_TEST_KEY = "test-" + "key-12345678"


def _make_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key=_TEST_KEY,
            base_url="https://my-llm.example.com/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model={"provider": "zai", "model": "glm-5.2"},
        )
        agent.client = MagicMock()
        return agent


def _restore_after_fallback(primary_model):
    agent = _make_agent()
    agent._primary_runtime["model"] = primary_model
    agent._primary_runtime["provider"] = "alibaba-token-plan"
    fb_client = MagicMock()
    fb_client.api_key = "fallback-" + "key-1234"
    fb_client.base_url = "https://fallback.example.com/v1"
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fb_client, None)):
        assert agent._try_activate_fallback() is True
    emitted = []
    agent._emit_status = emitted.append
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        restored = agent._restore_primary_runtime()
    return agent, restored, emitted


def test_restore_skips_a_primary_already_known_to_be_non_chat():
    agent, restored, emitted = _restore_after_fallback("wan2.7-image-pro")

    assert restored is False
    assert (agent.provider, agent.model, agent._fallback_activated) == ("zai", "glm-5.2", True)
    assert not any("Primary model restored" in notice for notice in emitted)

    other, other_restored, _emitted = _restore_after_fallback("acme/text-to-image")
    assert other_restored is False
    assert other.model == "glm-5.2"


def test_restore_still_returns_a_chat_primary():
    _agent, restored, emitted = _restore_after_fallback("qwen3.7-plus")

    assert restored is True
    assert any("Primary model restored" in notice for notice in emitted)
