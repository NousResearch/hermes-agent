from agent.auxiliary_client import _build_call_kwargs
from agent.chat_completion_helpers import _enforce_summary_openrouter_zdr
from agent.openrouter_zdr import enforce_openrouter_zdr


def test_shared_enforcement_overrides_false(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.openrouter_zdr_enabled", lambda: True)
    kwargs = {"extra_body": {"provider": {"sort": "price", "zdr": False}}}
    enforce_openrouter_zdr(
        kwargs,
        is_openrouter=True,
        base_url="https://openrouter.ai/api/v1",
    )
    assert kwargs["extra_body"]["provider"] == {"sort": "price", "zdr": True}


def test_shared_enforcement_skips_native_gemini(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.openrouter_zdr_enabled", lambda: True)
    kwargs = {}
    enforce_openrouter_zdr(
        kwargs,
        is_openrouter=True,
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    assert kwargs == {}


def test_auxiliary_openrouter_enforces_zdr_after_caller_body(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.openrouter_zdr_enabled", lambda: True)
    kwargs = _build_call_kwargs(
        "openrouter",
        "anthropic/claude-sonnet-4.6",
        [{"role": "user", "content": "ping"}],
        extra_body={"provider": {"zdr": False, "sort": "price"}},
        base_url="https://openrouter.ai/api/v1",
    )
    assert kwargs["extra_body"]["provider"] == {"zdr": True, "sort": "price"}


def test_auxiliary_fallback_label_uses_openrouter_base_url(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.openrouter_zdr_enabled", lambda: True)
    kwargs = _build_call_kwargs(
        "fallback_chain[0](openrouter)",
        "anthropic/claude-sonnet-4.6",
        [{"role": "user", "content": "ping"}],
        base_url="https://openrouter.ai/api/v1",
    )
    assert kwargs["extra_body"]["provider"] == {"zdr": True}


def test_summary_direct_call_enforces_zdr(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.openrouter_zdr_enabled", lambda: True)
    agent = type(
        "Agent",
        (),
        {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "_is_openrouter_url": lambda self: True,
        },
    )()
    kwargs = {"extra_body": {"provider": {"zdr": False}}}
    _enforce_summary_openrouter_zdr(agent, kwargs)
    assert kwargs["extra_body"]["provider"]["zdr"] is True


def test_disabled_zdr_preserves_caller_policy(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.openrouter_zdr_enabled", lambda: False)
    kwargs = {"extra_body": {"provider": {"zdr": False}}}
    enforce_openrouter_zdr(
        kwargs,
        is_openrouter=True,
        base_url="https://openrouter.ai/api/v1",
    )
    assert kwargs["extra_body"]["provider"]["zdr"] is False