"""Tests for Vertex AI runtime-provider resolution and profile registration.

Covers: provider-profile registration + aliases, alias canonicalization,
resolve_runtime_provider(vertex) minting an OAuth token, and the friendly
AuthError when credentials can't be resolved. No network calls.
"""

from __future__ import annotations

import pytest


def test_switch_model_resolves_vertex_profile_alias(monkeypatch):
    from hermes_cli import model_switch as ms
    from hermes_cli.model_switch import switch_model

    runtime_calls = []
    validation_calls = []

    def _resolve_runtime_provider(**kwargs):
        runtime_calls.append(kwargs)
        return {
            "api_key": "oauth-token",
            "base_url": (
                "https://us-central1-aiplatform.googleapis.com/"
                "v1beta1/openapi"
            ),
            "api_mode": "chat_completions",
        }

    def _validate_requested_model(model, provider, **kwargs):
        validation_calls.append((model, provider, kwargs))
        return {
            "accepted": True,
            "persist": False,
            "recognized": True,
            "message": "",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        _validate_requested_model,
    )
    monkeypatch.setattr(ms, "get_model_capabilities", lambda *_args: None)
    monkeypatch.setattr(ms, "get_model_info", lambda *_args: None)

    result = switch_model(
        raw_input="gemini-2.5-pro",
        current_provider="openrouter",
        current_model="google/gemini-2.5-flash",
        explicit_provider="vertex-ai",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.target_provider == "vertex"
    assert result.provider_changed is True
    assert result.api_key == "oauth-token"
    assert result.api_mode == "chat_completions"
    assert runtime_calls == [
        {"requested": "vertex", "target_model": "gemini-2.5-pro"}
    ]
    assert validation_calls[0][0:2] == ("gemini-2.5-pro", "vertex")


def test_resolve_runtime_provider_raises_autherror_when_unresolved(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(va, "get_vertex_config", lambda: (None, None))
    with pytest.raises(AuthError) as exc:
        rp.resolve_runtime_provider(requested="vertex")
    msg = str(exc.value)
    assert "OAuth2" in msg
    assert "not a static API key" in msg


def test_vertex_registered_in_provider_registry():
    """PROVIDER_REGISTRY (hermes_cli.auth) is what agent/auxiliary_client.py's
    resolve_provider_client() looks up before dispatching on auth_type. Without
    an entry here, the ``elif pconfig.auth_type == "vertex":`` branch there is
    unreachable dead code — every auxiliary Vertex call (vision, title
    generation, MoA reference/aggregator slots, ...) fails at the
    ``pconfig is None`` guard before ever reaching it."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get("vertex")
    assert cfg is not None
    assert cfg.auth_type == "vertex"


def test_vertex_registered_in_hermes_overlays():
    """hermes_cli.providers.get_provider("vertex") backs
    _preserve_provider_with_base_url() in agent/auxiliary_client.py, which
    decides whether a MoA slot's resolved Vertex (base_url, api_key) pair
    keeps its "vertex" provider identity or silently collapses to "custom" —
    losing the identity _refresh_provider_credentials() needs to re-mint an
    expired OAuth2 token on a 401."""
    from hermes_cli.providers import get_provider

    resolved = get_provider("vertex")
    assert resolved is not None
    assert resolved.auth_type == "vertex"
