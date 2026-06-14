"""Regression tests for #43866: reselecting the current model must not wipe credentials.

When the session-level model selector picks the SAME provider+model as the
global default, ``switch_model()`` takes the provider-unchanged branch and
re-resolves credentials via ``resolve_runtime_provider()``.  If that
resolution comes back empty (key lives only in the running agent's config,
named custom provider the resolver doesn't know, etc.), the result used to
carry ``api_key=""`` / ``base_url=""``.  The gateway stored those empty
strings in ``_session_model_overrides`` and applied them over the
correctly-resolved config defaults on the next message — so the session went
silently dead: no API call, no response.
"""

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _patch_common(monkeypatch, runtime):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: runtime,
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None
    )


def test_same_model_reselect_keeps_current_credentials(monkeypatch):
    """Empty resolver output must fall back to the working credentials."""
    _patch_common(
        monkeypatch, {"api_key": "", "base_url": "", "api_mode": ""}
    )

    result = switch_model(
        raw_input="openai/gpt-5.4-mini",
        current_provider="openrouter",
        current_model="openai/gpt-5.4-mini",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="sk-or-live-key",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.target_provider == "openrouter"
    assert result.api_key == "sk-or-live-key"
    assert result.base_url == "https://openrouter.ai/api/v1"


def test_same_provider_reselect_still_picks_up_rotated_credentials(monkeypatch):
    """Non-empty resolver output must still win (credential rotation)."""
    _patch_common(
        monkeypatch,
        {
            "api_key": "sk-or-rotated-key",
            "base_url": "https://openrouter.ai/api/v2",
            "api_mode": "chat_completions",
        },
    )

    result = switch_model(
        raw_input="openai/gpt-5.4-mini",
        current_provider="openrouter",
        current_model="openai/gpt-5.4-mini",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="sk-or-stale-key",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.api_key == "sk-or-rotated-key"
    assert result.base_url == "https://openrouter.ai/api/v2"
    assert result.api_mode == "chat_completions"
