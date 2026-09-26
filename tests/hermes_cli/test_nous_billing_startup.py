"""Real stored Nous credentials must respect billing benches at startup."""
import base64
import json
import time
from datetime import datetime, timezone

from agent.credential_pool import load_pool
from hermes_cli.runtime_provider import resolve_runtime_with_fallback


import pytest


@pytest.fixture
def nous_login(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expires = time.time() + 7200
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    token = part({"alg": "none"}) + "." + part({
        "sub": "fixture-account", "scope": "inference:invoke", "exp": expires,
    }) + ".sig"
    state = {
        "access_token": token, "agent_key": token, "refresh_token": "fixture-refresh",
        "scope": "inference:invoke", "token_type": "Bearer", "client_id": "hermes-cli",
        "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(),
        "agent_key_expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(),
        "inference_base_url": "https://inference-api.nousresearch.com/v1",
        "portal_base_url": "https://portal.nousresearch.com",
    }
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "nous", "providers": {"nous": state},
    }))
    pool = load_pool("nous")
    assert pool.select() is not None
    return pool, token


def fallback_config():
    return {"fallback_providers": [{"provider": "custom", "model": "fixture-model",
            "base_url": "http://127.0.0.1:9/v1", "api_key": "fixture-key"}]}


@pytest.mark.parametrize("entrypoint", ["explicit", "configured", "gateway"])
def test_billing_bench_routes_to_configured_fallback(nous_login, tmp_path, entrypoint):
    pool, _ = nous_login
    pool.mark_exhausted_and_rotate(status_code=402, failure_reason="billing")
    assert load_pool("nous").select() is None
    stamp = load_pool("nous").entries()[0].last_status_at
    config = fallback_config()
    if entrypoint != "explicit":
        from hermes_cli.config import atomic_config_write
        config["model"] = {"provider": "nous", "default": "fixture-primary"}
        atomic_config_write(tmp_path / "config.yaml", config)
    if entrypoint == "gateway":
        from gateway.run import _resolve_runtime_agent_kwargs
        runtime = _resolve_runtime_agent_kwargs()
        assert runtime["provider"] == "custom"
        assert runtime["model"] == "fixture-model"
        assert runtime["api_key"] == "fixture-key"
    else:
        runtime, fallback = resolve_runtime_with_fallback(
            config, requested="nous" if entrypoint == "explicit" else None, target_model="fixture-model")
        assert runtime["provider"] == "custom"
        assert fallback == config["fallback_providers"][0]
    assert load_pool("nous").entries()[0].last_status_at == stamp


@pytest.mark.parametrize("mode", ["healthy", "expired", "no_fallback", "explicit_key", "explicit_endpoint", "pool_error", "empty_pool", "fallback_error"])
def test_preserves_primary_routes(nous_login, monkeypatch, mode):
    pool, token = nous_login
    if mode != "healthy":
        pool.mark_exhausted_and_rotate(status_code=402, failure_reason="billing")
    if mode == "expired":
        now = time.time()
        monkeypatch.setattr(time, "time", lambda: now + 3700)
    kwargs = {}
    if mode == "explicit_key":
        kwargs["explicit_api_key"] = token
    if mode == "explicit_endpoint":
        kwargs["explicit_base_url"] = "http://127.0.0.1:9/v1"
    if mode in {"pool_error", "empty_pool"}:
        from hermes_cli import runtime_provider
        def unavailable_pool(_provider):
            if mode == "pool_error":
                raise OSError("fixture pool read failure")
            return None
        monkeypatch.setattr(runtime_provider, "load_pool", unavailable_pool)
    config = {} if mode == "no_fallback" else fallback_config()
    if mode == "fallback_error":
        from hermes_cli.auth import AuthError
        config = {"fallback_providers": [{"provider": "custom", "model": "fixture-model"}]}
        with pytest.raises(AuthError) as exc:
            resolve_runtime_with_fallback(config, requested="nous", target_model="fixture-model")
        assert exc.value.code == "insufficient_credits"
        return
    runtime, fallback = resolve_runtime_with_fallback(
        config, requested="nous", target_model="fixture-model", **kwargs)
    assert runtime["provider"] == "nous"
    assert fallback is None

