from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux


def test_claude_sdk_main_runtime_never_resolves_anthropic_or_ambient_auto_route():
    runtime = {
        "runtime": "claude_agent_sdk",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_mode": "anthropic_messages",
    }

    with (
        patch.object(aux, "resolve_provider_client") as resolve,
        patch.object(aux, "_get_provider_chain") as ambient_chain,
        patch.object(aux, "_try_main_fallback_chain") as main_fallback,
    ):
        client, model = aux._resolve_auto(
            main_runtime=runtime,
            task="title_generation",
        )

    assert (client, model) == (None, None)
    resolve.assert_not_called()
    ambient_chain.assert_not_called()
    main_fallback.assert_not_called()


def test_runtime_identity_discriminates_auto_client_cache():
    hermes_key = aux._client_cache_key(
        "auto",
        async_mode=False,
        main_runtime={
            "runtime": "hermes",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        },
        task="title_generation",
    )
    subscription_key = aux._client_cache_key(
        "auto",
        async_mode=False,
        main_runtime={
            "runtime": "claude_agent_sdk",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        },
        task="title_generation",
    )

    assert hermes_key != subscription_key


def test_process_global_external_runtime_also_fails_closed_without_override():
    aux.set_runtime_main(
        "anthropic",
        "claude-sonnet-4-6",
        runtime="claude_agent_sdk",
    )
    try:
        with patch.object(aux, "resolve_provider_client") as resolve:
            assert aux._resolve_auto(task="compression") == (None, None)
        resolve.assert_not_called()
    finally:
        aux.clear_runtime_main()


def test_call_local_external_runtime_survives_global_interleaving_without_paid_fallback():
    class PaymentError(Exception):
        status_code = 402

    client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(completions=SimpleNamespace()),
    )

    def fail_after_other_session_overwrites_globals(**kwargs):
        aux.set_runtime_main("anthropic", "paid-model", runtime="hermes")
        raise PaymentError("credits exhausted")

    client.chat.completions.create = fail_after_other_session_overwrites_globals
    external_runtime = {
        "runtime": "claude_agent_sdk",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    }
    resolver = MagicMock(return_value=(None, None))
    try:
        with (
            patch.object(
                aux,
                "_resolve_task_provider_model",
                return_value=("openrouter", "explicit-aux", "", "", "chat_completions"),
            ),
            patch.object(aux, "_get_cached_client", return_value=(client, "explicit-aux")),
            patch.object(aux, "_try_configured_fallback_chain", return_value=(None, None, "")),
            patch.object(aux, "resolve_provider_client", resolver),
        ):
            with pytest.raises(PaymentError, match="credits exhausted"):
                aux.call_llm(
                    task="title_generation",
                    main_runtime=external_runtime,
                    messages=[{"role": "user", "content": "title"}],
                )
    finally:
        aux.clear_runtime_main()

    resolver.assert_not_called()


def test_auxiliary_fallback_does_not_reinterpret_external_agent_runtime_entry():
    with patch.object(aux, "resolve_provider_client") as resolve:
        client, model = aux._resolve_fallback_entry(
            {
                "runtime": "claude_agent_sdk",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            }
        )

    assert (client, model) == (None, None)
    resolve.assert_not_called()
