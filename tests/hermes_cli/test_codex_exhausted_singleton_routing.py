"""Account-wide pool cooldowns also govern singleton copies of the same token."""

import json
import time

import pytest

from tests.hermes_cli import test_codex_pool_fallback_eligibility as routing_fixtures
from tests.hermes_cli.test_codex_pool_fallback_eligibility import MODEL, OTHER, _row, _store

home = routing_fixtures.home


def _alias(home, **status):
    return _store(home, [_row(source="device_code", **status)], providers={
        "openai-codex": {"tokens": {
            "access_token": "synthetic-pool", "refresh_token": "synthetic-refresh"}}})


@pytest.mark.parametrize("reset", [4_000_000_000, 4_000_000_000_000, "2096-10-02T07:06:40Z"])
@pytest.mark.parametrize("model", [MODEL, None])
def test_exhausted_alias_is_refused_without_diagnostic_mutation(home, reset, model):
    from hermes_cli.auth import AuthError, resolve_codex_runtime_credentials
    path = _alias(home, last_status="exhausted", last_error_code=429, last_error_reset_at=reset)
    before = path.read_bytes()
    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials(read_only=True, force_refresh=True, target_model=model)
    assert path.read_bytes() == before
    assert not (home / "auth.lock").exists()


def test_exhausted_alias_walks_whole_configured_fallback(home, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback
    # Deterministic offline upstream answer: quota has not reopened early.
    monkeypatch.setattr("hermes_cli.auth._probe_codex_quota_restored", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.auth_codex._probe_codex_quota_restored", lambda *a, **kw: False)
    _alias(home, last_status="exhausted", last_error_code=429, last_error_reset_at=4_000_000_000)
    config = json.loads((home / "config.yaml").read_text())
    runtime, fallback = resolve_runtime_with_fallback(config, requested="openai-codex", target_model=MODEL)
    assert fallback is not None
    assert fallback["model"] == "fallback-model"
    assert runtime["provider"] == "openrouter"
    assert runtime["base_url"] == "https://openrouter.ai/api/v1"
    assert runtime["api_key"] == "synthetic-fallback"


def test_exhausted_alias_with_status_timestamp_uses_credential_cooldown(home):
    from hermes_cli.auth import AuthError, resolve_codex_runtime_credentials
    path = _alias(home, last_status="exhausted", last_error_code=429, last_status_at=time.time())
    before = path.read_bytes()
    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials(read_only=True, target_model=MODEL)
    assert path.read_bytes() == before
    assert not (home / "auth.lock").exists()


@pytest.mark.parametrize("status", [
    {"last_status": "exhausted", "last_error_code": 429, "last_error_reset_at": 1_000},
    {"last_status": "exhausted", "last_error_code": 429, "last_status_at": 1},
    {"model_cooldowns": {OTHER: 4_000_000_000}},
])
def test_elapsed_or_unrelated_alias_remains_read_only_and_usable(home, status):
    from hermes_cli.auth import resolve_codex_runtime_credentials
    path = _alias(home, **status)
    before = path.read_bytes()
    runtime = resolve_codex_runtime_credentials(read_only=True, target_model=MODEL)
    assert runtime["api_key"] == "synthetic-pool"
    assert path.read_bytes() == before
    assert not (home / "auth.lock").exists()
