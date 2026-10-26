"""The Kanban recovery probe must validate callable active-provider credentials on the wire."""

from __future__ import annotations

import httpx

from hermes_cli import doctor_connectivity as dc


def _runtime(credential, *, provider="openrouter", base_url="https://provider.example/v1", api_mode=""):
    return {
        "provider": provider,
        "api_key": credential,
        "base_url": base_url,
        "api_mode": api_mode,
    }


def test_callable_active_provider_credential_is_materialized_then_probed(monkeypatch):
    """A callable is healthy only if it mints a token that the active provider accepts."""
    credential = lambda: "minted-by-source"
    calls: list[tuple[str, dict, float]] = []

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda: _runtime(credential))
    monkeypatch.setattr("agent.command_token_source.materialize_probe_api_key", lambda value: "minted-token")

    def fake_get(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "get", fake_get)

    assert dc.probe_active_provider_credential(timeout=3.0) == {
        "ok": True, "provider": "openrouter", "reason": "", "detail": "HTTP 200",
    }
    assert calls == [(
        "https://provider.example/v1/models",
        {"Authorization": "Bearer minted-token", "User-Agent": dc._HERMES_USER_AGENT},
        3.0,
    )]


def test_unmaterializable_callable_active_provider_credential_never_probes(monkeypatch):
    """A failed mint remains unhealthy; resolving the provider alone must not unblock Kanban work."""
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda: _runtime(lambda: "unused"))
    monkeypatch.setattr("agent.command_token_source.materialize_probe_api_key", lambda value: "")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not request")))

    assert dc.probe_active_provider_credential() == {
        "ok": False,
        "provider": "openrouter",
        "reason": "no_credential",
        "detail": "No usable API key / base URL resolved for the active provider.",
    }


def test_native_anthropic_active_provider_uses_native_models_auth(monkeypatch):
    """A direct Anthropic route must use its native endpoint and x-api-key header, never Bearer."""
    runtime = _runtime(
        "native-key", provider="anthropic", base_url="https://api.anthropic.com/v1",
        api_mode="anthropic_messages",
    )
    calls: list[tuple[str, dict, object, float]] = []
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda: runtime)

    def fake_get(url, *, headers, params, timeout):
        calls.append((url, headers, params, timeout))
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "get", fake_get)

    assert dc.probe_active_provider_credential(timeout=3.0) == {
        "ok": True, "provider": "anthropic", "reason": "", "detail": "HTTP 200",
    }
    assert calls == [(
        "https://api.anthropic.com/v1/models",
        {"anthropic-version": "2023-06-01", "User-Agent": dc._HERMES_USER_AGENT, "x-api-key": "native-key"},
        None,
        3.0,
    )]
