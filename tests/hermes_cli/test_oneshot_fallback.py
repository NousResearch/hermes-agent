"""#81209: ``hermes -z`` must consult the fallback chain at *resolution* time.

A quota-exhausted / expired primary raises ``AuthError`` from ``resolve_runtime_provider`` before
``AIAgent`` exists, so the mid-session ``fallback_model`` wiring never gets a chance. The shared
``resolve_runtime_with_fallback`` helper gives oneshot the gateway's resolution-time behaviour."""

import pytest

from hermes_cli.auth import AuthError
from hermes_cli.runtime_provider import resolve_runtime_with_fallback

_CFG = {"fallback_providers": [
    {"provider": "anthropic", "model": "claude-x", "api_key": "fb-key"},
    {"provider": "openai", "model": "gpt-x"},
]}


class TestResolveRuntimeWithFallback:
    def test_auth_error_walks_chain_in_order_and_re_raises_primary(self, monkeypatch):
        calls = []

        def fake_resolve(**kw):
            calls.append(kw)
            if kw.get("requested") == "openai-codex":
                raise AuthError("Codex provider quota exhausted (429); retry after 39750s.")
            if kw.get("requested") == "anthropic":
                raise AuthError("anthropic key missing")
            return {"provider": kw["requested"], "api_key": "k"}

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)
        runtime, entry = resolve_runtime_with_fallback(_CFG, requested="openai-codex", target_model="gpt-5.4")
        assert (runtime["provider"], entry["model"]) == ("openai", "gpt-x")
        # Chain walked in config order; the first entry got its inline api_key and its own model.
        assert [c.get("requested") for c in calls] == ["openai-codex", "anthropic", "openai"]
        assert calls[1]["explicit_api_key"] == "fb-key" and calls[1]["target_model"] == "claude-x"

        def all_fail(**kw):
            raise AuthError("primary down" if kw.get("requested") == "openai-codex" else "fallback down")

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", all_fail)
        with pytest.raises(AuthError, match="primary down"):  # primary-error precedence
            resolve_runtime_with_fallback(_CFG, requested="openai-codex")

    def test_misconfiguration_is_never_rerouted(self, monkeypatch, caplog):
        def typo(**kw):
            raise ValueError("Unknown provider 'antropic'")

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", typo)
        with pytest.raises(ValueError):
            resolve_runtime_with_fallback(_CFG, requested="antropic")
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {"provider": "p"})
        assert resolve_runtime_with_fallback({}) == ({"provider": "p"}, None)

        # A misconfigured *fallback* entry is skipped, but loudly: a typo must not vanish at debug level.
        def primary_down_first_entry_typo(**kw):
            if kw.get("requested") == "openai-codex":
                raise AuthError("primary down")
            if kw.get("requested") == "anthropic":
                raise ValueError("Unknown provider")
            return {"provider": kw["requested"]}

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", primary_down_first_entry_typo)
        with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
            _, entry = resolve_runtime_with_fallback(_CFG, requested="openai-codex")
        assert entry["provider"] == "openai"
        assert any(r.levelname == "WARNING" and "anthropic" in r.getMessage() for r in caplog.records)

    def test_halt_without_usable_chain_preserves_primary_error(self, monkeypatch, caplog):
        primary = AuthError("primary unavailable")
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: (_ for _ in ()).throw(primary),
        )
        monkeypatch.setattr(
            "hermes_cli.fallback_config.fallback_halt_active",
            lambda: (True, "fallback halt refusal"),
        )

        with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
            with pytest.raises(AuthError) as raised:
                resolve_runtime_with_fallback({}, requested="openai-codex")

        assert raised.value is primary
        assert "fallback halt refusal" not in caplog.text


def test_run_agent_falls_back_when_primary_resolution_raises_auth_error(monkeypatch):
    """End-to-end: ``_run_agent`` builds AIAgent against the fallback entry's provider/model (#81209)."""
    import hermes_cli.oneshot as oneshot_mod

    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __setattr__(self, name, _value):
            pass

        def run_conversation(self, _prompt, conversation_history=None):
            return {"final_response": "pong", "session_id": "s"}

        def close(self):
            pass

    def fake_resolve(**kw):
        # A config-sourced model resolves with requested=None: the ladder reads model.provider itself.
        if kw.get("requested") in (None, "openai-codex"):
            raise AuthError("Codex provider quota exhausted (429); retry after 39750s. Credentials are still valid.")
        return {"api_key": "fb", "base_url": None, "provider": kw["requested"], "api_mode": "chat", "credential_pool": None}

    cfg = {"model": {"default": "gpt-5.4", "provider": "openai-codex"}, **_CFG}
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: [])
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)

    text, _ = oneshot_mod._run_agent("Health check: reply pong.")

    assert text == "pong"
    assert (captured["provider"], captured["model"]) == ("anthropic", "claude-x")
    assert captured["api_key"] == "fb"
