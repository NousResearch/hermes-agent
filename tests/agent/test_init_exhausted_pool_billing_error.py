"""An exhausted (402) provider pool must be named as a billing failure at init.

#94785: a Desktop composer pick routed through OpenRouter while the OpenRouter account was out of
credits hit the generic "No LLM provider configured. Run `hermes model` …" message — actively
misleading, because a *different* provider (the config default) was configured and working and only
the session's override pool was burned. The ``openrouter`` / ``custom`` branch of
``_routed_client_kwargs`` has no provider-specific missing-credentials message, so it fell through to
the setup message; named providers fell through to the "no API key was found" one. Both must instead
name the exhausted pool, its provider and its last error code.
"""

import json
import time
from types import SimpleNamespace

import pytest

from auth.credential_pool import (
    STATUS_EXHAUSTED,
    STATUS_OK,
    CredentialPool,
    PooledCredential,
)


def _agent(provider="openrouter", model="z-ai/glm-5.3"):
    return SimpleNamespace(
        provider=provider,
        model=model,
        base_url=None,
        api_key=None,
        _fallback_activated=False,
    )


def _credential(provider="openrouter", **overrides):
    fields = dict(
        provider=provider,
        id="e1",
        label="test-cred",
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-test-0000000000",
    )
    fields.update(overrides)
    return PooledCredential(**fields)


def _exhausted_pool(provider="openrouter"):
    from hermes_cli.config_credentials import credential_pool_environment
    entry = _credential(
        provider,
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_code=402,
        last_error_message="You requested up to 65536 tokens, but can only afford 4282.",
    )
    return CredentialPool(provider, [entry], environment=credential_pool_environment())


def _healthy_pool(provider="openrouter"):
    from hermes_cli.config_credentials import credential_pool_environment
    return CredentialPool(provider, [_credential(provider, last_status=STATUS_OK)], environment=credential_pool_environment())


def _exhausted_pool_429(provider="openrouter"):
    from hermes_cli.config_credentials import credential_pool_environment
    entry = _credential(
        provider,
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_code=429,
        last_error_message="rate limited",
    )
    return CredentialPool(provider, [entry], environment=credential_pool_environment())


def _patch_no_routed_client(monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client", lambda *a, **k: (None, None)
    )
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: None
    )


@pytest.mark.parametrize(
    "provider, model",
    [
        ("openrouter", "z-ai/glm-5.3"),
        ("custom", "qwen3.5:4b"),
        ("zai", "glm-5.3"),
    ],
)
def test_exhausted_pool_raises_typed_billing_error(monkeypatch, provider, model):
    """Every provider with a fully burned pool names the billing failure, not setup/missing-key."""
    from agent import agent_init
    from agent.auxiliary_unavailable import ProviderCredentialsExhaustedError

    _patch_no_routed_client(monkeypatch)
    monkeypatch.setattr(
        "auth.credential_pool.load_pool", lambda p, environment=None: _exhausted_pool(provider)
    )

    with pytest.raises(ProviderCredentialsExhaustedError) as excinfo:
        agent_init._routed_client_kwargs(_agent(provider, model), None, 60)

    message = str(excinfo.value)
    assert excinfo.value.provider == provider
    assert f"'{provider}'" in message
    assert model in message
    assert "402" in message
    assert "exhausted" in message
    # The whole point of #94785: the setup / missing-key sentence must NOT be what the user sees.
    assert "No LLM provider configured" not in message
    assert "no API key was found" not in message


def test_healthy_pool_keeps_the_existing_diagnostic(monkeypatch):
    """A usable credential must not be misreported as exhausted."""
    from agent import agent_init

    _patch_no_routed_client(monkeypatch)
    monkeypatch.setattr(
        "auth.credential_pool.load_pool", lambda p, environment=None: _healthy_pool("openrouter")
    )

    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        agent_init._routed_client_kwargs(_agent("openrouter"), None, 60)


def test_empty_pool_keeps_the_existing_diagnostic(monkeypatch):
    """No read of the pool (empty / unreadable) must not invent an exhaustion verdict."""
    from hermes_cli.config_credentials import credential_pool_environment
    from agent import agent_init

    _patch_no_routed_client(monkeypatch)
    monkeypatch.setattr(
        "auth.credential_pool.load_pool", lambda p, environment=None: CredentialPool("openrouter", [], environment=credential_pool_environment())
    )

    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        agent_init._routed_client_kwargs(_agent("openrouter"), None, 60)


def test_pool_billing_message_names_billing_entry():
    from agent.auxiliary_unavailable import pool_billing_message

    message = pool_billing_message(
        "openrouter", model="z-ai/glm-5.3", pool=_exhausted_pool("openrouter")
    )
    assert message is not None
    assert "'openrouter'" in message
    assert "z-ai/glm-5.3" in message
    assert "402" in message
    assert "can only afford 4282" in message


def test_pool_billing_message_none_when_pool_is_usable():
    from hermes_cli.config_credentials import credential_pool_environment
    from agent.auxiliary_unavailable import pool_billing_message

    assert pool_billing_message("openrouter", pool=_healthy_pool()) is None
    assert pool_billing_message("openrouter", pool=CredentialPool("openrouter", [], environment=credential_pool_environment())) is None


def test_pool_billing_message_defers_non_billing_cooldown():
    """A plain 429/quota bench is not billing: the caller keeps the cooldown wording (#56810)."""
    from hermes_cli.config_credentials import credential_pool_environment
    from agent.auxiliary_unavailable import pool_billing_message

    entry = _credential(
        "openrouter",
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_code=429,
        last_error_message="rate limited",
    )
    pool = CredentialPool("openrouter", [entry], environment=credential_pool_environment())
    assert pool_billing_message("openrouter", pool=pool) is None


def test_openrouter_quota_cooldown_still_reports_the_cooldown(monkeypatch, tmp_path):
    """openrouter 429 pool: the cooldown wording (with reset time) survives, not the setup message.

    Regression guard for the openrouter/custom fall-through this fix widens: previously only named
    providers reached ``pool_cooldown_message``.
    """
    from agent import agent_init

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reset_at = time.time() + 3 * 3600
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "credential_pool": {
                    "openrouter": [
                        {
                            "provider": "openrouter",
                            "id": "or1",
                            "label": "or1",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": "sk-test-0000000000",
                            "last_status": STATUS_EXHAUSTED,
                            "last_status_at": time.time(),
                            "last_error_code": 429,
                            "last_error_reset_at": reset_at,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: z-ai/glm-5.3\n",
        encoding="utf-8",
    )

    _patch_no_routed_client(monkeypatch)
    monkeypatch.setattr(
        "auth.credential_pool.load_pool", lambda p, environment=None: _exhausted_pool_429("openrouter")
    )

    with pytest.raises(RuntimeError) as excinfo:
        agent_init._routed_client_kwargs(_agent("openrouter", "z-ai/glm-5.3"), None, 60)

    message = str(excinfo.value)
    assert "cooling down" in message
    assert "No LLM provider configured" not in message
