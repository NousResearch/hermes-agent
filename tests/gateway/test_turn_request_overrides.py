"""Regression tests: the gateway must preserve a custom provider's
``request_overrides`` on per-turn agent config.

A ``custom_providers`` entry can carry an ``extra_body`` (e.g.
``chat_template_kwargs`` to toggle a local model's thinking).
``resolve_runtime_provider`` surfaces it as ``request_overrides`` on the
resolved runtime dict, but the gateway used to rebuild the runtime from a
fixed key whitelist that omitted it -- so the provider's configured
``extra_body`` never reached the model on the gateway path, and only
``/fast`` service-tier overrides survived.
"""

import pytest

from gateway.run import GatewayRunner


PROVIDER_OVERRIDES = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


def _runtime_kwargs(**extra):
    base = {
        "api_key": "no-key-required",
        "base_url": "http://10.0.0.1:8000/v1",
        "provider": "custom",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "max_tokens": None,
    }
    base.update(extra)
    return base


def _runner(service_tier=None):
    runner = object.__new__(GatewayRunner)
    runner._service_tier = service_tier
    return runner


def test_provider_request_overrides_preserved_without_service_tier():
    """No /fast: the provider's extra_body must pass straight through."""
    runner = _runner(service_tier=None)
    rk = _runtime_kwargs(request_overrides=PROVIDER_OVERRIDES)
    route = runner._resolve_turn_agent_config("hi", "main", rk)
    assert route["request_overrides"] == PROVIDER_OVERRIDES
    # A copy, not an alias into runtime_kwargs.
    assert route["request_overrides"] is not rk["request_overrides"]


def test_provider_request_overrides_merged_under_fast_mode(monkeypatch):
    """/fast active: provider extra_body AND the service-tier marker both survive."""
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda model_id: {"service_tier": "priority"},
    )
    runner = _runner(service_tier="priority")
    rk = _runtime_kwargs(request_overrides=PROVIDER_OVERRIDES)
    route = runner._resolve_turn_agent_config("hi", "main", rk)
    assert route["request_overrides"]["extra_body"] == PROVIDER_OVERRIDES["extra_body"]
    assert route["request_overrides"]["service_tier"] == "priority"


def test_no_provider_overrides_yields_empty():
    """Regression: absent provider overrides, behaviour is unchanged ({})."""
    runner = _runner(service_tier=None)
    route = runner._resolve_turn_agent_config("hi", "main", _runtime_kwargs())
    assert route["request_overrides"] == {}


def test_resolve_runtime_agent_kwargs_carries_request_overrides(monkeypatch):
    """The module-level runtime resolver must not drop request_overrides."""
    import gateway.run as gateway_run

    fake_runtime = {
        "api_key": "k",
        "base_url": "http://10.0.0.1:8000/v1",
        "provider": "custom",
        "api_mode": "chat_completions",
        "request_overrides": PROVIDER_OVERRIDES,
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda *a, **k: dict(fake_runtime),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config", lambda: {}
    )
    rk = gateway_run._resolve_runtime_agent_kwargs()
    assert rk["request_overrides"] == PROVIDER_OVERRIDES
