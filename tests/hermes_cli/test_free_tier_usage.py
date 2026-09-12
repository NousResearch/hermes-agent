"""Lifetime usage is shared by identity, not copied with profile configuration."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _guest(identity="one"):
    return {"auth_method": "anonymous", "anon_token": f"anon_test_{identity}"}


def test_usage_survives_profiles_refresh_and_concurrent_completions(tmp_path, monkeypatch):
    from hermes_cli import auth, auth_nous
    from hermes_cli import free_tier_usage as usage
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    guest = _guest()
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    identity = usage.current_identity()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: usage.record_completed_tool(identity), range(10)))
    sibling = root / "profiles" / "sibling"
    sibling.mkdir(parents=True)
    token = set_hermes_home_override(str(sibling))
    try:
        from hermes_cli.anon_auth import apply_exchange_to_state
        apply_exchange_to_state(guest, {"access_token": "rotated-fixture", "expires_in": 900})
        auth._save_active_provider_state("nous", guest)
        auth_nous._write_shared_nous_state(guest)  # JWT refresh cannot reset usage
        assert usage.current_identity() == identity
        assert usage.status() == {"tool_calls_used": 10, "tool_call_cap": 10, "capped": True}
        auth._save_active_provider_state("nous", _guest("different"))
        assert usage.status()["tool_calls_used"] == 0
    finally:
        reset_hermes_home_override(token)
    assert usage.status()["capped"] is True
    assert "anon_test" not in usage._usage_path().read_text()


def test_actual_account_not_sign_in_attempt_releases_gate(tmp_path, monkeypatch):
    from hermes_cli import auth, auth_nous
    from hermes_cli import free_tier_usage as usage

    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    guest = _guest()
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    identity = usage.current_identity()
    for _ in range(10):
        usage.record_completed_tool(identity)
    assert usage.status()["capped"] is True
    auth_nous._write_shared_nous_state({"access_token": "test-account", "refresh_token": "test-refresh"})
    assert usage.current_identity() is None
    assert usage.status()["capped"] is False
