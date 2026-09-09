"""#106475: with a single credential, a Codex ChatGPT-account entitlement 400 must fail closed —
the rejected slug is skipped by the fallback walk and restore_primary_runtime must not switch back
and announce an unverified "Primary model restored" that would oscillate forever.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.fallback_cooldown import _mark_entitlement_rejected_model

from run_agent import AIAgent

# Assembled at runtime so no credential-shaped literal sits in source (scanner guard).
_TEST_KEY = "test-" + "key-12345678"


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key=_TEST_KEY, base_url="https://my-llm.example.com/v1", provider="custom",
            quiet_mode=True, skip_context_files=True, skip_memory=True, fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _entitlement_error(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=400,
        message=f"The '{model}' model is not supported when using Codex with a ChatGPT account.",
    )


def test_marker_fires_only_on_single_credential_entitlement_400_and_fallback_walk_skips_it():
    agent = _make_agent(fallback_model={"provider": "openai-codex", "model": "openai/gpt-5.6-sol"})
    agent.provider, agent.model = "openai-codex", "gpt-5.6-sol"
    for err in (
        SimpleNamespace(status_code=500, message="server error"),
        SimpleNamespace(status_code=429, message="rate limited"),
        SimpleNamespace(status_code=400, message="invalid request body"),
        SimpleNamespace(status_code=403, message="model is not supported when using Codex with a ChatGPT account."),
    ):
        assert _mark_entitlement_rejected_model(agent, err) is False
    # Multi-credential pool: another account may be entitled — leave it to rotation (#71970).
    pool = MagicMock()
    pool.entries.return_value = [object(), object()]
    agent._credential_pool = pool
    assert _mark_entitlement_rejected_model(agent, _entitlement_error("gpt-5.6-sol")) is False
    assert getattr(agent, "_entitlement_rejected_models", None) is None

    agent._credential_pool = None
    assert _mark_entitlement_rejected_model(agent, _entitlement_error("gpt-5.6-sol")) is True
    assert agent._entitlement_rejected_models == {("openai-codex", "gpt-5.6-sol")}
    # The fallback entry names the same slug in vendor-prefixed form: skipped, chain exhausts.
    with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
        assert agent._try_activate_fallback() is False
        resolve.assert_not_called()


def test_restore_primary_runtime_is_gated_on_rejected_primary_slug():
    fb_client = MagicMock()
    fb_client.api_key, fb_client.base_url = "fallback-" + "key-1234", "https://fallback.example.com/v1"

    def _run(rejected_slug):
        agent = _make_agent(fallback_model={"provider": "zai", "model": "glm-5.2"})
        agent._primary_runtime["model"], agent._primary_runtime["provider"] = "gpt-5.6-sol", "openai-codex"
        agent._entitlement_rejected_models = {("openai-codex", rejected_slug)}
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fb_client, None)):
            assert agent._try_activate_fallback() is True
        emitted = []
        agent._emit_status = emitted.append
        with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
            restored = agent._restore_primary_runtime()
        return agent, restored, emitted

    agent, restored, emitted = _run("gpt-5.6-sol")
    assert restored is False
    assert (agent.provider, agent.model, agent._fallback_activated) == ("zai", "glm-5.2", True)
    assert not any("Primary model restored" in n for n in emitted)

    _, restored, emitted = _run("some-other-slug")
    assert restored is True
    assert any("Primary model restored" in n for n in emitted)
