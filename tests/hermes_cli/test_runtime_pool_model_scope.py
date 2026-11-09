"""Model-scoped benches must not disable account rotation for another model."""

import json
import time

import pytest

from hermes_cli import runtime_provider as rp


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: supported-model\n"
        "auth:\n  adopt_external_logins: false\n"
        "credential_pool_strategies:\n  openai-codex: round_robin\n"
    )
    rows = [{
        "id": f"account-{i}", "label": f"account-{i}", "priority": i,
        "auth_type": "api_key", "source": "manual",
        "access_token": f"synthetic-account-{i}-token",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "model_cooldowns": {"unsupported-model": time.time() + 3600},
    } for i in range(2)]
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "credential_pool": {"openai-codex": rows},
    }))
    return home


@pytest.mark.parametrize("target_model", [None, "supported-model"])
@pytest.mark.parametrize("stale_identity", [False, True])
def test_default_and_explicit_models_keep_account_rotation(codex_home, target_model, stale_identity):
    from gateway.run import _resolve_runtime_agent_kwargs, _resolve_runtime_agent_kwargs_for_provider

    runtime = (_resolve_runtime_agent_kwargs() if target_model is None else
               _resolve_runtime_agent_kwargs_for_provider("openai-codex", target_model=target_model))
    pool = runtime["credential_pool"]
    assert pool is not None
    selected_key = runtime["api_key"]

    replacement = pool.mark_exhausted_and_rotate(
        status_code=429, failure_reason="rate_limit", model="supported-model",
        api_key_hint="stale-session-token" if stale_identity else selected_key,
        error_context={"message": "The usage limit has been reached"},
    )

    assert replacement is not None
    assert replacement.access_token != selected_key
    assert pool.select(model="unsupported-model") is None


def test_explicit_model_cannot_bypass_its_own_bench(codex_home, monkeypatch):
    def no_singleton():
        raise rp.AuthError("no usable singleton")

    monkeypatch.setattr(rp, "resolve_codex_runtime_credentials", no_singleton)
    with pytest.raises(rp.AuthError, match="no usable singleton"):
        rp.resolve_runtime_provider(requested="openai-codex", target_model="unsupported-model")
