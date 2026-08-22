from unittest.mock import patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm
from hermes_cli.auth import PROVIDER_REGISTRY, provider_api_key_env_hint


def test_provider_api_key_env_hint_uses_registry_and_safe_fallback():
    assert provider_api_key_env_hint("minimax-oauth") == "MINIMAX_API_KEY"
    assert PROVIDER_REGISTRY["minimax-oauth"].api_key_env_vars == ()
    assert provider_api_key_env_hint("alibaba") == "DASHSCOPE_API_KEY"
    assert provider_api_key_env_hint("future-provider") == "FUTURE-PROVIDER_API_KEY"


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_minimax_oauth_missing_credentials_names_supported_api_key_env(
    async_mode,
):
    kwargs = {
        "messages": [{"role": "user", "content": "summarize"}],
        "provider": "minimax-oauth",
        "model": "MiniMax-M2.1",
        "task": "compression",
    }
    with (
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(None, kwargs["model"]),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
            return_value=(None, None, None),
        ),
    ):
        with pytest.raises(
            RuntimeError, match="Set the MINIMAX_API_KEY environment variable"
        ):
            if async_mode:
                await async_call_llm(**kwargs)
            else:
                call_llm(**kwargs)
