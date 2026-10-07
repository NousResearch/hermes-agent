"""A fresh root OAuth grant must resolve identically in CLI and multiplexed cron scopes."""
import json
from pathlib import Path
import time

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from cron.scheduler_provider import _profile_cron_scope
from hermes_cli.auth import AuthError, get_auth_status, is_rate_limited_auth_error
from hermes_cli.runtime_provider import resolve_runtime_provider


@pytest.mark.parametrize("cooldown", [False, True])
def test_root_anthropic_oauth_resolves_in_profile_cron_scope(tmp_path, monkeypatch, cooldown):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "install-root"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("auth:\n  adopt_external_logins: false\n")
    (profile / "config.yaml").write_text("auth:\n  adopt_external_logins: false\n")
    (root / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"anthropic": [{
            "id": "fixture", "source": "manual:hermes_pkce", "auth_type": "oauth",
            "priority": 0, "label": "fixture", "access_token": "sk-ant-oat01-fixture",
            "refresh_token": "fixture-refresh", "expires_at_ms": 9999999999999,
            "model_cooldowns": {"claude-opus-5-5": time.time() + 3600} if cooldown else {},
        }]},
    }))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    token = set_secret_scope({}, profile_home=str(profile))
    try:
        assert get_auth_status("anthropic")["logged_in"] is True
        if not cooldown:
            assert resolve_runtime_provider(requested="anthropic", target_model="claude-opus-5-5")["provider"] == "anthropic"
        if cooldown:
            with pytest.raises(AuthError, match="rate-limited.*claude-opus-5-5") as error:
                resolve_runtime_provider(requested="anthropic", target_model="claude-opus-5-5")
            assert is_rate_limited_auth_error(error.value)
        cli_runtime = resolve_runtime_provider(requested="anthropic", target_model="claude-haiku-4-5")
        assert cli_runtime["provider"] == "anthropic"
    finally:
        reset_secret_scope(token)

    monkeypatch.setenv("HERMES_HOME", str(root))
    with _profile_cron_scope(profile):
        if cooldown:
            with pytest.raises(AuthError, match="rate-limited.*claude-opus-5-5") as error:
                resolve_runtime_provider(requested="anthropic", target_model="claude-opus-5-5")
            assert is_rate_limited_auth_error(error.value)
        cron_runtime = resolve_runtime_provider(requested="anthropic", target_model="claude-haiku-4-5")
        assert cron_runtime["provider"] == "anthropic"
        assert cron_runtime["api_key"] == cli_runtime["api_key"]
    assert not (profile / "auth.json").exists()
