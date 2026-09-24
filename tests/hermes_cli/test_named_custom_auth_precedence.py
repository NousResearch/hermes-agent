"""Invariant tests for named custom provider credential resolution.

Contract: main (``resolve_runtime_provider``) and auxiliary
(``resolve_provider_client``) share one precedence — explicit > key_cmd >
inline api_key > key_env > credential pool > host-gated env — and a declared
but unresolvable credential on a non-loopback endpoint fails closed instead
of resolving the ``no-key-required`` placeholder. Keyless local endpoints
(loopback, no declared source) keep the placeholder.
"""

import json

import pytest


def _write_config(home, config_dict):
    import yaml

    (home / "config.yaml").write_text(yaml.dump(config_dict))


@pytest.fixture
def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    (home / "config.yaml").write_text("model:\n  default: test-model\n")
    for var in ("EXAMPLE_API_KEY", "MISSING_NAMED_KEY", "POOL_SHOULD_LOSE"):
        monkeypatch.delenv(var, raising=False)
    return home


def test_configured_key_env_beats_matching_pool_on_both_paths(_home, tmp_path, monkeypatch):
    """A usable key_env beats an already-matching pool on main AND aux (parity)."""
    monkeypatch.setenv("EXAMPLE_API_KEY", "dummy-env-key")
    _write_config(_home, {
        "providers": {
            "example-service": {
                "name": "example-service",
                "base_url": "https://api.example.invalid/v1",
                "key_env": "EXAMPLE_API_KEY",
                "default_model": "example-model",
            },
        },
        "model": {"provider": "custom:example-service", "default": "example-model"},
    })
    (_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {
            "custom:example-service": [{
                "id": "k1", "label": "pool", "auth_type": "api_key",
                "priority": 0, "source": "manual",
                "access_token": "dummy-pool-key",
            }],
        },
    }))

    from hermes_cli import runtime_provider as rp
    from agent import auxiliary_client as aux

    try:
        aux.shutdown_cached_clients()
        main = rp.resolve_runtime_provider(requested="custom:example-service")
        assert main["api_key"] == "dummy-env-key"

        client, model = aux.resolve_provider_client("custom:example-service", "example-model")
        assert client is not None and model == "example-model"
        assert client.api_key == "dummy-env-key"
        assert str(client.base_url).rstrip("/") == main["base_url"].rstrip("/")
    finally:
        aux.shutdown_cached_clients()


def test_declared_missing_key_fails_closed_and_loopback_stays_keyless(
    _home, tmp_path, monkeypatch,
):
    """Missing declared key on a remote endpoint fails closed; loopback stays keyless."""
    _write_config(_home, {
        "providers": {
            "needs-auth": {
                "name": "needs-auth",
                "base_url": "https://api.example.invalid/v1",
                "key_env": "MISSING_NAMED_KEY",
                "default_model": "m",
            },
            "local": {
                "name": "local",
                "base_url": "http://127.0.0.1:8080/v1",
                "default_model": "m",
            },
        },
        "model": {"provider": "custom:needs-auth", "default": "m"},
    })

    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth_constants import AuthError
    from agent import auxiliary_client as aux

    try:
        aux.shutdown_cached_clients()
        with pytest.raises(AuthError) as excinfo:
            rp.resolve_runtime_provider(requested="custom:needs-auth")
        assert excinfo.value.code == "missing_api_key"

        client, _model = aux.resolve_provider_client("needs-auth", "m")
        assert client is None

        # Keyless local endpoint: no declared source, loopback → placeholder survives.
        local = rp.resolve_runtime_provider(requested="custom:local")
        assert local["api_key"] == "no-key-required"
        aux_client, _m = aux.resolve_provider_client("local", "m")
        assert aux_client is not None
        assert aux_client.api_key == "no-key-required"
    finally:
        aux.shutdown_cached_clients()
