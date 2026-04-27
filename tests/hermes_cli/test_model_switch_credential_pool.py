"""Regression tests for credential_pool propagation through switch_model.

The shared /model pipeline (``hermes_cli.model_switch.switch_model``) calls
``resolve_runtime_provider()`` for the target provider, which returns the
provider's credential pool alongside ``api_key`` / ``base_url`` / ``api_mode``.
Historically the pool was discarded, so the gateway's session-override
machinery had nothing to propagate — leaving the original provider's pool
live across a /model switch (#16678).

These tests pin the contract: ``ModelSwitchResult.credential_pool`` carries
the resolved pool for the target provider on both the explicit-provider
path and the same-provider re-resolve path.
"""

from unittest.mock import MagicMock

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _stub_model_metadata(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: _MOCK_VALIDATION,
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)


def test_switch_model_returns_credential_pool_on_explicit_provider(monkeypatch):
    """Explicit --provider switch must surface the new provider's pool.

    Without this, a 429 on the new provider rotates against the original
    provider's pool (or skips rotation entirely) — see #16678.
    """
    target_pool = MagicMock(name="codex_pool")

    def _fake_resolve_runtime_provider(**kwargs):
        assert kwargs["requested"] == "openai-codex"
        return {
            "api_key": "sk-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "provider": "openai-codex",
            "credential_pool": target_pool,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )
    _stub_model_metadata(monkeypatch)

    result = switch_model(
        raw_input="gpt-5.4",
        current_provider="anthropic",
        current_model="claude-sonnet-4-6",
        current_base_url="https://api.anthropic.com",
        current_api_key="sk-ant-old",
        explicit_provider="openai-codex",
    )

    assert result.success is True
    assert result.target_provider == "openai-codex"
    # Critical: the new provider's pool must round-trip through the result
    # so the gateway can store it on the session override.
    assert result.credential_pool is target_pool


def test_switch_model_returns_credential_pool_on_same_provider_reresolve(monkeypatch):
    """Same-provider switches still re-resolve runtime, including the pool.

    Models that don't change provider still go through resolve_runtime_provider
    so credential rotation / OpenCode base_url adjustments are picked up.
    The pool returned by that call must surface in the result.
    """
    pool = MagicMock(name="anthropic_pool")

    def _fake_resolve_runtime_provider(**kwargs):
        assert kwargs["requested"] == "anthropic"
        return {
            "api_key": "sk-ant-rotated",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
            "provider": "anthropic",
            "credential_pool": pool,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )
    _stub_model_metadata(monkeypatch)

    result = switch_model(
        raw_input="claude-opus-4-7",
        current_provider="anthropic",
        current_model="claude-sonnet-4-6",
        current_base_url="https://api.anthropic.com",
        current_api_key="sk-ant-old",
    )

    assert result.success is True
    assert result.target_provider == "anthropic"
    assert result.credential_pool is pool


def test_switch_model_default_credential_pool_is_none(monkeypatch):
    """When runtime resolution yields no pool, the result carries None.

    Some providers (single env-var key, custom local endpoints without a
    saved pool) genuinely have no pool — the gateway override layer must
    see ``None`` so it can clear the previous provider's pool rather than
    leaving stale rotation in place.
    """
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "sk-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "provider": "openai-codex",
            # No credential_pool key — older code paths sometimes omit it.
        },
    )
    _stub_model_metadata(monkeypatch)

    result = switch_model(
        raw_input="gpt-5.4",
        current_provider="anthropic",
        current_model="claude-sonnet-4-6",
        current_base_url="https://api.anthropic.com",
        current_api_key="sk-ant-old",
        explicit_provider="openai-codex",
    )

    assert result.success is True
    assert result.credential_pool is None
