"""Oneshot direct-alias base_url must preserve query for credential-scope checks."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli import model_switch as ms


@pytest.fixture(autouse=True)
def _reset_direct_aliases():
    ms.DIRECT_ALIASES.clear()
    yield
    ms.DIRECT_ALIASES.clear()


def test_oneshot_alias_preserves_query_trailing_slash_in_explicit_base_url(
    monkeypatch,
):
    """hermes -z must not rstrip alias URLs before resolve_runtime_provider."""
    alias_url = "https://trusted.internal/v1?tenant=a/"
    captured: dict = {}

    def _fake_resolve_runtime_provider(**kwargs):
        captured.update(kwargs)
        return {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": kwargs.get("explicit_base_url"),
            "api_key": "no-key-required",
            "source": "test",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "ignored-default"}},
    )
    monkeypatch.setattr(
        "run_agent.AIAgent",
        lambda **kwargs: MagicMock(
            run_conversation=lambda *_a, **_k: {
                "final_response": "ok",
                "failed": False,
                "partial": False,
            }
        ),
    )
    monkeypatch.setattr("hermes_cli.oneshot._create_session_db_for_oneshot", lambda: None)

    ms.DIRECT_ALIASES["qalias"] = ms.DirectAlias(
        model="m",
        provider="custom:trusted-private",
        base_url=alias_url,
    )

    from hermes_cli.oneshot import _run_agent

    _run_agent("hello", model="qalias")

    assert captured["explicit_base_url"] == alias_url
    assert captured["requested"] == "custom:trusted-private"


def test_oneshot_named_custom_query_slash_mismatch_drops_secret(monkeypatch):
    """End-to-end: oneshot alias URL must not admit named secret on query mismatch."""
    import hermes_cli.runtime_provider as rp

    named_secret = "NAMED-PROVIDER-SECRET"
    alias_url = "https://trusted.internal/v1?tenant=a"
    agent_kwargs: dict = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    providers = [
        {
            "name": "trusted-private",
            "base_url": "https://trusted.internal/v1?tenant=a/",
            "api_key": named_secret,
        }
    ]
    monkeypatch.setattr(rp, "load_config", lambda: {"custom_providers": providers})
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("resolve_provider should not run for named custom providers")
        ),
    )
    monkeypatch.setattr(rp, "load_pool", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("pool must not be consulted when query scopes differ")
    ))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "ignored-default"}, "custom_providers": providers},
    )

    def _capture_agent(**kwargs):
        agent_kwargs.update(kwargs)
        return SimpleNamespace(
            run_conversation=lambda *_a, **_k: {
                "final_response": "ok",
                "failed": False,
                "partial": False,
            },
        )

    monkeypatch.setattr("run_agent.AIAgent", _capture_agent)
    monkeypatch.setattr("hermes_cli.oneshot._create_session_db_for_oneshot", lambda: None)

    ms.DIRECT_ALIASES["qalias"] = ms.DirectAlias(
        model="m",
        provider="custom:trusted-private",
        base_url=alias_url,
    )

    from hermes_cli.oneshot import _run_agent

    _run_agent("hello", model="qalias")

    assert agent_kwargs["base_url"] == alias_url
    assert agent_kwargs["api_key"] != named_secret
    assert agent_kwargs["api_key"] == "no-key-required"
