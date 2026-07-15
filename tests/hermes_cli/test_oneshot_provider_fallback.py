"""Setup-time provider fallback for the shared resolver helper.

``resolve_runtime_provider_with_fallback`` centralizes the "primary provider
failed to authenticate at resolution time -> walk ``fallback_providers``"
recovery that the gateway and interactive CLI already perform, so oneshot
(``hermes -z``) workers stop dying on a missing primary credential when a
usable fallback is configured.
"""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp
from hermes_cli.auth import AuthError


def test_primary_success_returns_no_switch(monkeypatch):
    monkeypatch.setattr(
        rp, "resolve_runtime_provider", lambda **kw: {"provider": "zai", "api_key": "k"}
    )
    runtime, provider, model = rp.resolve_runtime_provider_with_fallback(
        requested="zai",
        fallback_chain=[{"provider": "opencode-go", "model": "qwen3.6-plus"}],
    )
    assert runtime["provider"] == "zai"
    # No switch happened, so the caller keeps its own provider/model.
    assert provider is None
    assert model is None


def test_autherror_falls_over_to_working_fallback(monkeypatch):
    def fake_resolve(**kw):
        requested = kw.get("requested")
        if requested == "nous":
            raise AuthError(
                "No access token found for Nous Portal login.",
                provider="nous",
                relogin_required=True,
            )
        return {"provider": requested, "api_key": "k"}

    monkeypatch.setattr(rp, "resolve_runtime_provider", fake_resolve)
    switched: list[tuple[str, str]] = []

    runtime, provider, model = rp.resolve_runtime_provider_with_fallback(
        requested="nous",
        fallback_chain=[
            {"provider": "", "model": "skip-me"},  # invalid entry is skipped
            {"provider": "zai", "model": "glm-5.2"},
        ],
        on_fallback=lambda p, m, exc: switched.append((p, m)),
    )

    assert provider == "zai"
    assert model == "glm-5.2"
    assert runtime["provider"] == "zai"
    assert switched == [("zai", "glm-5.2")]


def test_autherror_empty_chain_reraises_primary(monkeypatch):
    def fake_resolve(**kw):
        raise AuthError("no token", provider="nous", relogin_required=True)

    monkeypatch.setattr(rp, "resolve_runtime_provider", fake_resolve)
    with pytest.raises(AuthError):
        rp.resolve_runtime_provider_with_fallback(requested="nous", fallback_chain=[])


def test_autherror_all_fallbacks_fail_reraises_primary(monkeypatch):
    def fake_resolve(**kw):
        raise AuthError("every provider unusable", provider=kw.get("requested"))

    monkeypatch.setattr(rp, "resolve_runtime_provider", fake_resolve)
    with pytest.raises(AuthError):
        rp.resolve_runtime_provider_with_fallback(
            requested="nous",
            fallback_chain=[{"provider": "zai", "model": "glm-5.2"}],
        )


def test_non_autherror_propagates_without_trying_fallback(monkeypatch):
    calls: list[str | None] = []

    def fake_resolve(**kw):
        calls.append(kw.get("requested"))
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "resolve_runtime_provider", fake_resolve)
    with pytest.raises(RuntimeError):
        rp.resolve_runtime_provider_with_fallback(
            requested="nous",
            fallback_chain=[{"provider": "zai", "model": "glm-5.2"}],
        )
    # Fallback recovery is auth-only; a non-auth failure must not walk the chain.
    assert calls == ["nous"]
