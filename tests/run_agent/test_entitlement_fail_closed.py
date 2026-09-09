"""#106475: with a single credential, a Codex ChatGPT-account entitlement 400 must fail
closed — the rejected slug is skipped by the fallback walk and restore_primary_runtime
must not switch back and announce an unverified "Primary model restored" that would
oscillate forever.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import chat_completion_helpers

from run_agent import AIAgent

# Assembled at runtime so no credential-shaped literal sits in source (scanner guard).
_TEST_KEY = "test-" + "key-12345678"
_FALLBACK_KEY = "fallback-" + "key-1234"


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(fallback_model=None, provider="custom", base_url="https://my-llm.example.com/v1"):
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key=_TEST_KEY,
            base_url=base_url,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _entitlement_error(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=400,
        message=(
            f"The '{model}' model is not supported when using Codex with a ChatGPT account."
        ),
    )


def _mock_resolve(base_url="https://fallback.example.com/v1", api_key=_FALLBACK_KEY):
    mock_client = MagicMock()
    mock_client.api_key = api_key
    mock_client.base_url = base_url
    return mock_client


class TestMarkEntitlementRejectedModel:
    def test_marks_codex_entitlement_400(self):
        agent = _make_agent()
        agent.provider = "openai-codex"
        agent.model = "gpt-5.6-sol"
        assert chat_completion_helpers._mark_entitlement_rejected_model(
            agent, _entitlement_error("gpt-5.6-sol")
        ) is True
        assert agent._entitlement_rejected_models == {("openai-codex", "gpt-5.6-sol")}

    def test_ignores_other_statuses_and_bodies(self):
        agent = _make_agent()
        agent.provider = "openai-codex"
        agent.model = "gpt-5.6-sol"
        for err in (
            SimpleNamespace(status_code=500, message="server error"),
            SimpleNamespace(status_code=429, message="rate limited"),
            SimpleNamespace(status_code=400, message="invalid request body"),
            SimpleNamespace(status_code=403, message="model is not supported when using Codex with a ChatGPT account."),
        ):
            assert chat_completion_helpers._mark_entitlement_rejected_model(agent, err) is False
        assert getattr(agent, "_entitlement_rejected_models", None) is None

    def test_multi_credential_pool_is_left_to_rotation(self):
        agent = _make_agent()
        agent.provider = "openai-codex"
        agent.model = "gpt-5.6-sol"
        pool = MagicMock()
        pool.entries.return_value = [object(), object()]
        agent._credential_pool = pool
        assert chat_completion_helpers._mark_entitlement_rejected_model(
            agent, _entitlement_error("gpt-5.6-sol")
        ) is False
        assert getattr(agent, "_entitlement_rejected_models", None) is None

    def test_single_credential_pool_still_marks(self):
        agent = _make_agent()
        agent.provider = "openai-codex"
        agent.model = "gpt-5.6-sol"
        pool = MagicMock()
        pool.entries.return_value = [object()]
        agent._credential_pool = pool
        assert chat_completion_helpers._mark_entitlement_rejected_model(
            agent, _entitlement_error("gpt-5.6-sol")
        ) is True


class TestFallbackWalkSkipsRejectedSlug:
    def test_rejected_entry_is_skipped_and_chain_exhausts(self):
        agent = _make_agent(
            fallback_model={"provider": "openai-codex", "model": "gpt-5.6-sol"},
        )
        agent._entitlement_rejected_models = {("openai-codex", "gpt-5.6-sol")}
        with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
            assert agent._try_activate_fallback() is False
            resolve.assert_not_called()

    def test_normalized_slug_form_is_also_skipped(self):
        agent = _make_agent(
            fallback_model={"provider": "openai-codex", "model": "openai/gpt-5.6-sol"},
        )
        # The runtime (and the marker) recorded the post-normalization slug.
        agent._entitlement_rejected_models = {("openai-codex", "gpt-5.6-sol")}
        with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
            assert agent._try_activate_fallback() is False
            resolve.assert_not_called()


class TestRestoreGate:
    def test_restore_does_not_switch_back_to_rejected_primary(self):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5.2"},
        )
        agent._primary_runtime["model"] = "gpt-5.6-sol"
        agent._primary_runtime["provider"] = "openai-codex"
        agent._entitlement_rejected_models = {("openai-codex", "gpt-5.6-sol")}
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)):
            assert agent._try_activate_fallback() is True
        assert agent._fallback_activated is True

        emitted = []
        agent._emit_status = emitted.append
        with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is False

        # Still on the fallback; no unverified "Primary model restored" claim.
        assert agent.provider == "zai"
        assert agent.model == "glm-5.2"
        assert agent._fallback_activated is True
        assert emitted == []

    def test_unrejected_primary_restores_normally(self):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5.2"},
        )
        agent._entitlement_rejected_models = {("openai-codex", "some-other-slug")}
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)):
            assert agent._try_activate_fallback() is True

        emitted = []
        agent._emit_status = emitted.append
        with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True
        assert any("Primary model restored" in notice for notice in emitted)
