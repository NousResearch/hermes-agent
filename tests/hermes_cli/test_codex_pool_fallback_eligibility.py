"""Pool fallback cannot resurrect accounts rejected by model-aware selection."""

import json
from pathlib import Path

import pytest

MODEL = "gpt-5.3-codex"
OTHER = "gpt-5.3-codex-mini"
BASE = "https://chatgpt.com/backend-api/codex"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    (root / "config.yaml").write_text(json.dumps({
        "model": {"provider": "openai-codex", "default": OTHER},
        "fallback_providers": [{"provider": "openrouter", "model": "fallback-model",
                                "api_key": "synthetic-fallback"}],
    }))
    import httpx
    def no_network(*args, **kwargs):
        pytest.fail("Credential routing must not make a provider request")
    monkeypatch.setattr(httpx.Client, "send", no_network)
    return root


def _row(**status):
    return {"id": "account", "label": "account", "source": "manual", "priority": 0,
            "auth_type": "oauth", "access_token": "synthetic-pool", "refresh_token": "synthetic-refresh",
            "expires_at_ms": 4_000_000_000_000, "base_url": BASE, **status}


def _store(home, rows, **extra):
    path = home / "auth.json"
    path.write_text(json.dumps({"credential_pool": {"openai-codex": rows}, **extra}))
    return path


@pytest.mark.parametrize("status", [
    {"model_cooldowns": {MODEL: 4_000_000_000}},
    {"last_status": "dead", "last_status_at": 2_000_000_000},
])
def test_excluded_pool_only_account_walks_configured_fallback(home, status):
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback
    _store(home, [_row(**status)])
    config = json.loads((home / "config.yaml").read_text())
    runtime, fallback = resolve_runtime_with_fallback(config, requested="openai-codex", target_model=MODEL)
    assert fallback is not None
    assert fallback["model"] == "fallback-model"
    assert runtime["provider"] == "openrouter"
    assert runtime["api_key"] == "synthetic-fallback"


@pytest.mark.parametrize("status", [
    {"model_cooldowns": {MODEL: 4_000_000_000}},
    {"last_status": "dead", "last_status_at": 2_000_000_000},
    {"last_status": "exhausted", "last_error_code": 429, "last_error_reset_at": 4_000_000_000_000},
])
def test_unknown_model_diagnostic_skips_excluded_rows_without_writes(home, status):
    from hermes_cli.auth import resolve_codex_runtime_credentials
    path = _store(home, [_row(**status), _row(id="healthy", access_token="synthetic-healthy",
                                           base_url="https://codex-gateway.example/backend-api/codex")])
    before = path.read_bytes()
    runtime = resolve_codex_runtime_credentials(read_only=True, force_refresh=True)
    assert runtime["api_key"] == "synthetic-healthy"
    assert runtime["base_url"] == "https://codex-gateway.example/backend-api/codex"
    assert path.read_bytes() == before
    assert not (home / "auth.lock").exists()


def test_explicit_endpoint_fallback_uses_requested_model(home):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    _store(home, [_row(model_cooldowns={MODEL: 4_000_000_000}),
                  _row(id="eligible", access_token="synthetic-eligible", model_cooldowns={OTHER: 4_000_000_000})])
    runtime = resolve_runtime_provider(requested="openai-codex", target_model=MODEL, explicit_base_url=BASE)
    assert runtime["api_key"] == "synthetic-eligible"
    assert runtime["base_url"] == BASE


def test_explicit_key_is_authoritative_over_pool_bench(home):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    _store(home, [_row(model_cooldowns={MODEL: 4_000_000_000})])
    runtime = resolve_runtime_provider(requested="openai-codex", target_model=MODEL,
                                       explicit_api_key="synthetic-explicit", explicit_base_url=BASE)
    assert runtime["api_key"] == "synthetic-explicit"
    assert runtime.get("credential_pool") is None


def test_valid_independent_singleton_remains_available(home):
    from hermes_cli.auth import resolve_codex_runtime_credentials
    _store(home, [_row(last_status="dead")], providers={"openai-codex": {"tokens": {
        "access_token": "synthetic-singleton", "refresh_token": "synthetic-singleton-refresh"}}})
    runtime = resolve_codex_runtime_credentials(read_only=True)
    assert runtime["api_key"] == "synthetic-singleton"
    assert runtime["source"] == "hermes-auth-store"


def test_healthy_pool_only_account_and_elapsed_bench_remain_available(home):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    _store(home, [_row(model_cooldowns={MODEL: 1})])
    runtime = resolve_runtime_provider(requested="openai-codex", target_model=MODEL)
    assert runtime["api_key"] == "synthetic-pool"
    assert runtime["credential_pool"] is not None


@pytest.mark.parametrize("status", [
    {"model_cooldowns": {MODEL: 4_000_000_000}},
    {"last_status": "dead", "last_status_at": 2_000_000_000},
])
def test_singleton_alias_cannot_resurrect_excluded_pool_account(home, status):
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback
    _store(home, [_row(source="device_code", **status)], providers={"openai-codex": {"tokens": {
        "access_token": "synthetic-pool", "refresh_token": "synthetic-refresh"}}})
    config = json.loads((home / "config.yaml").read_text())
    runtime, fallback = resolve_runtime_with_fallback(config, requested="openai-codex", target_model=MODEL)
    assert fallback is not None
    assert runtime["api_key"] == "synthetic-fallback"


def test_singleton_with_only_unrelated_model_bench_stays_valid_and_read_only(home):
    from hermes_cli.auth import resolve_codex_runtime_credentials
    path = _store(home, [_row(source="device_code", model_cooldowns={OTHER: 4_000_000_000})],
                  providers={"openai-codex": {"tokens": {
                      "access_token": "synthetic-pool", "refresh_token": "synthetic-refresh"}}})
    before = path.read_bytes()
    runtime = resolve_codex_runtime_credentials(target_model=MODEL, read_only=True, force_refresh=True)
    assert runtime["api_key"] == "synthetic-pool"
    assert runtime["source"] == "hermes-auth-store"
    assert path.read_bytes() == before
    assert not (home / "auth.lock").exists()
