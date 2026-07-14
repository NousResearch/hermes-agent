"""Regression tests for ``_resolve_task_provider_model`` provider priority.

Locks in the fix for a class of bugs where an explicit ``provider:`` setting
in ``auxiliary.<task>`` config (e.g. ``provider: bedrock``) was silently
rewritten to ``"custom"`` whenever a ``base_url`` and ``api_key`` were also
present.

The trap surfaces most visibly with AWS Bedrock — pointing the OpenAI HTTP
client at ``https://bedrock-runtime.<region>.amazonaws.com/chat/completions``
with a bearer-token returns a 200 response with an empty body, which then
crashes downstream as::

    Auxiliary vision: LLM returned invalid response (type=ChatCompletion):
    'ChatCompletion(id=None, choices=None, ...)'

Same trap applies to any provider whose adapter is incompatible with the
generic OpenAI chat-completions wire (``anthropic``, ``nous``,
``openai-codex``, ...). The fix preserves the explicit provider so its
adapter (boto3 Converse, AnthropicBedrock SDK, Codex Responses API, ...)
gets used instead.

Reference: ``Auxiliary vision broken on Bedrock with bearer token`` —
without this guard, ``call_llm(task='vision')`` and
``resolve_vision_provider_client(provider='bedrock', base_url=..., api_key=...)``
silently downgrade to ``provider='custom'`` and produce empty
``ChatCompletion`` objects.
"""

from unittest.mock import patch

from agent.auxiliary_client import _resolve_task_provider_model


class TestExplicitProviderWinsOverBaseUrl:
    """Explicit known providers must NOT be downgraded to ``"custom"``."""

    def test_explicit_bedrock_with_base_url_and_api_key_kept(self):
        provider, model, base_url, api_key, _ = _resolve_task_provider_model(
            task=None,
            provider="bedrock",
            model="us.anthropic.claude-sonnet-4-6",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key="ABSK-bearer-token-here",
        )
        assert provider == "bedrock"
        assert model == "us.anthropic.claude-sonnet-4-6"
        assert base_url == "https://bedrock-runtime.us-east-1.amazonaws.com"
        assert api_key == "ABSK-bearer-token-here"

    def test_explicit_anthropic_with_base_url_kept(self):
        provider, _, _, _, _ = _resolve_task_provider_model(
            task=None,
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-...",
        )
        assert provider == "anthropic"

    def test_explicit_openai_codex_with_base_url_kept(self):
        provider, _, _, _, _ = _resolve_task_provider_model(
            task=None,
            provider="openai-codex",
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
        )
        assert provider == "openai-codex"

    def test_no_provider_with_base_url_falls_back_to_custom(self):
        """Without an explicit provider, base_url still implies 'custom'."""
        provider, _, _, _, _ = _resolve_task_provider_model(
            task=None,
            provider=None,
            base_url="https://my-llm.example.com/v1",
            api_key="local-key",
        )
        assert provider == "custom"

    def test_explicit_custom_with_base_url_stays_custom(self):
        """Explicit ``provider='custom'`` is honored as-is."""
        provider, _, _, _, _ = _resolve_task_provider_model(
            task=None,
            provider="custom",
            base_url="https://my-llm.example.com/v1",
            api_key="local-key",
        )
        assert provider == "custom"

    def test_explicit_auto_with_base_url_falls_back_to_custom(self):
        """``provider='auto'`` is treated like missing provider."""
        provider, _, _, _, _ = _resolve_task_provider_model(
            task=None,
            provider="auto",
            base_url="https://my-llm.example.com/v1",
            api_key="local-key",
        )
        assert provider == "custom"


class TestTaskConfigBedrockPriority:
    """End-to-end regression: ``auxiliary.vision`` config with explicit
    ``provider: bedrock`` plus ``base_url`` + ``api_key`` must resolve to
    ``"bedrock"``, not ``"custom"``.

    This was the actual production bug — config said ``provider: bedrock``
    and Hermes still routed through the OpenAI HTTP client because the
    base_url+api_key heuristic ran first.
    """

    def test_bedrock_config_resolves_to_bedrock(self):
        bedrock_cfg = {
            "provider": "bedrock",
            "model": "us.anthropic.claude-sonnet-4-6",
            "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "api_key": "ABSK-bearer-token",
        }
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value=bedrock_cfg,
        ):
            provider, model, base_url, api_key, _ = _resolve_task_provider_model(
                task="vision"
            )
        assert provider == "bedrock"
        assert model == "us.anthropic.claude-sonnet-4-6"
        assert base_url == "https://bedrock-runtime.us-east-1.amazonaws.com"
        assert api_key == "ABSK-bearer-token"

    def test_custom_config_still_resolves_to_custom(self):
        """Plain custom-endpoint config (no provider key) still works."""
        custom_cfg = {
            "base_url": "https://my-llm.example.com/v1",
            "api_key": "local-key",
        }
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value=custom_cfg,
        ):
            provider, _, base_url, api_key, _ = _resolve_task_provider_model(
                task="vision"
            )
        assert provider == "custom"
        assert base_url == "https://my-llm.example.com/v1"
        assert api_key == "local-key"

    def test_anthropic_config_resolves_to_anthropic(self):
        cfg = {
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-ant-...",
        }
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value=cfg,
        ):
            provider, _, _, _, _ = _resolve_task_provider_model(task="vision")
        assert provider == "anthropic"
