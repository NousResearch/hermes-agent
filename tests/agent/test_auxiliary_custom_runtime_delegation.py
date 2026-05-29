"""Task 8 (cpf-zkw.8): auxiliary client delegates custom-provider RESOLUTION
to the unified resolver in ``hermes_cli.runtime_provider``.

The auxiliary client used to keep a divergent twin of provider resolution:
``_resolve_custom_runtime`` called the unified resolver but, on any failure,
fell back to reading ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` straight from the
environment. That was the *last* place in the codebase that consulted
``OPENAI_BASE_URL`` for routing — contradicting the locked decision that
``config.yaml`` is the sole base_url source and ``OPENAI_BASE_URL`` is only
ever warned-about, never consulted (#4165, #5161, plan §3 decision 3/7).

These tests pin:
  * the env-var fallback is gone (resolver is the single source of truth);
  * aux carries the resolver's base_url / api_key / api_mode verbatim;
  * the openrouter default still reads as "no custom endpoint" for aux;
  * a real-resolver round trip resolves the *same* base_url the CLI would
    (test matrix §5 ``test_cli_and_aux_resolve_identical_base_url``, #8919).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
        "CUSTOM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Env var is no longer consulted on resolution failure (#4165 / #5161)
# ---------------------------------------------------------------------------

def test_openai_base_url_not_consulted_on_resolution_failure(monkeypatch):
    """When the unified resolver raises, aux must NOT fall back to the
    ambient OPENAI_BASE_URL/OPENAI_API_KEY env vars — config.yaml is the
    sole base_url source. This is the last consult of OPENAI_BASE_URL."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-poison:9999/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-poison")

    def _boom(**_kwargs):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _boom
    )
    from agent.auxiliary_client import _resolve_custom_runtime

    base, key, mode = _resolve_custom_runtime()
    assert base is None, "OPENAI_BASE_URL env must not be consulted for routing"
    assert key is None
    assert mode is None


def test_openai_base_url_not_consulted_when_resolver_returns_none(monkeypatch):
    """Defensive: when the resolver returns None (failure), aux returns
    'no custom endpoint' rather than reading the env var."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-poison:9999/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-poison")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_k: None,
    )
    from agent.auxiliary_client import _resolve_custom_runtime

    assert _resolve_custom_runtime() == (None, None, None)


# ---------------------------------------------------------------------------
# Aux carries the resolver's decision verbatim (delegation parity)
# ---------------------------------------------------------------------------

def _rp(**kw):
    """Build a ResolvedProvider like the real resolver returns."""
    from hermes_cli.provider_resolution import ResolvedProvider

    base = {
        "provider": "custom", "requested_provider": "custom",
        "api_mode": "chat_completions", "base_url": "", "api_key": "",
        "base_url_source": "config.base_url", "key_source": "config",
    }
    base.update(kw)
    return ResolvedProvider(**base)


def test_delegates_base_url_key_and_api_mode(monkeypatch):
    """A resolved custom endpoint is carried through with its base_url,
    api_key and api_mode exactly as the unified resolver returned them."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_k: _rp(
            base_url="http://localhost:1234/v1",
            api_key="sk-real",
            api_mode="anthropic_messages",
        ),
    )
    from agent.auxiliary_client import _resolve_custom_runtime

    base, key, mode = _resolve_custom_runtime()
    assert base == "http://localhost:1234/v1"
    assert key == "sk-real"
    assert mode == "anthropic_messages"


def test_local_endpoint_gets_placeholder_key(monkeypatch):
    """Local servers (no key) still get the OpenAI-SDK placeholder key."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_k: _rp(base_url="http://localhost:11434/v1", api_key=""),
    )
    from agent.auxiliary_client import _resolve_custom_runtime

    base, key, mode = _resolve_custom_runtime()
    assert base == "http://localhost:11434/v1"
    assert key == "no-key-required"
    assert mode == "chat_completions"


def test_openrouter_default_treated_as_no_custom(monkeypatch):
    """requested='custom' falls back to OpenRouter when nothing is configured;
    aux treats that as 'no custom endpoint'."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_k: _rp(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or",
        ),
    )
    from agent.auxiliary_client import _resolve_custom_runtime

    assert _resolve_custom_runtime() == (None, None, None)


# ---------------------------------------------------------------------------
# Real-resolver round trip — aux resolves the SAME base_url as the CLI (#8919)
# ---------------------------------------------------------------------------

def test_cli_and_aux_resolve_identical_base_url(tmp_path, monkeypatch):
    """A config.yaml custom endpoint resolves to the identical, normalized
    base_url through both the CLI resolver and the auxiliary client — no
    divergent twin (#8919)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: my-local-model\n"
        "  provider: custom\n"
        "  base_url: http://localhost:1234\n"
        "  api_key: sk-cfg\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.runtime_provider import resolve_runtime_provider
    from agent.auxiliary_client import _resolve_custom_runtime

    cli_runtime = resolve_runtime_provider(requested="custom")
    aux_base, aux_key, _aux_mode = _resolve_custom_runtime()

    # Bare custom host gains exactly one /v1 in the shared normalizer (#4600).
    assert cli_runtime["base_url"] == "http://localhost:1234/v1"
    assert aux_base == cli_runtime["base_url"], (
        "aux must resolve the identical base_url the CLI does"
    )
    assert aux_key == cli_runtime["api_key"]
    assert "openrouter.ai" not in (aux_base or "")
