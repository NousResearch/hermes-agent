"""Credential-pool rows in `hermes doctor`: burned pool vs config loss (#119533)."""

import json
import time
from pathlib import Path

from hermes_constants import get_hermes_home

_BASE = {
    "id": "key-1",
    "label": "key-1",
    "auth_type": "api_key",
    "priority": 0,
    "source": "manual",
    "access_token": "sk-or-test",
    "base_url": "https://openrouter.ai/api/v1",
}


def _write_openrouter_pool(entries) -> None:
    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"openrouter": entries}}),
        encoding="utf-8",
    )


def test_benched_pool_reports_wait_and_restart_hint(capsys):
    _write_openrouter_pool([{
        **_BASE,
        "last_status": "exhausted",
        "last_status_at": time.time(),
        "last_error_code": 402,
    }])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "Credential Pools" in out  # section header only renders when rows exist
    assert "Credential pool: openrouter" in out
    assert "all 1 entries benched" in out
    assert "restart will NOT clear it" in out
    assert len(finding.manual_issues) == 1, finding.manual_issues
    assert "hermes auth reset openrouter" in finding.manual_issues[0]


def test_dead_pool_without_recovery_time_is_an_issue(capsys):
    _write_openrouter_pool([{**_BASE, "last_status": "dead", "last_status_at": time.time()}])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "all 1 entries unavailable, no recovery time" in out
    assert len(finding.manual_issues) == 1, finding.manual_issues
    assert "hermes auth reset openrouter" in finding.manual_issues[0]


def test_available_pool_reports_ok_without_issues(capsys):
    _write_openrouter_pool([dict(_BASE)])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "Credential pool: openrouter" in out
    assert "at least one available" in out
    assert finding.manual_issues == [], finding.manual_issues


def test_unconfigured_pool_prints_nothing(capsys):
    _write_openrouter_pool([])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "Credential pool" not in out
    assert finding.manual_issues == [], finding.manual_issues


def test_unhydrated_env_reference_is_not_accused_of_being_burned(capsys):
    """An env-sourced row persists as a metadata-only fingerprint; when its secret does not
    resolve in this process the pool reports no available entry WITHOUT any burn state —
    that must stay silent here instead of being called a burned pool (#119533)."""
    _write_openrouter_pool([{
        **_BASE,
        "source": "env:OPENROUTER_API_KEY",
        "access_token": "",
        "extra": {"secret_fingerprint": "abc123"},
    }])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "Credential pool" not in out, out
    assert finding.manual_issues == [], finding.manual_issues


def test_unreadable_pool_warns_and_reports_issue(capsys, monkeypatch):
    """A pool store that raises must produce a warn row + issue instead of killing the
    scan for the remaining providers (one broken provider must not hide the others)."""
    import agent.credential_pool as cp

    def _boom(_pid):
        raise RuntimeError("auth.json malformed")

    monkeypatch.setattr(cp, "load_pool", _boom)
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "unreadable: auth.json malformed" in out
    assert len(finding.manual_issues) >= 1, finding.manual_issues
    assert "auth.json malformed" in finding.manual_issues[0]


def test_mixed_dead_and_exhausted_reports_recovery_time(capsys):
    """dead + timed-exhausted covering every entry: the pool DOES come back (the exhausted
    sibling recovers), so the row must show the recovery window, not the no-recovery variant."""
    _write_openrouter_pool([
        {**_BASE, "last_status": "dead", "last_status_at": time.time()},
        {**_BASE, "id": "key-2", "label": "key-2",
         "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": 402},
    ])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "all 2 entries benched, back in ~" in out
    assert "no recovery time" not in out
    assert len(finding.manual_issues) == 1, finding.manual_issues


def test_check_registered_and_openrouter_coverage():
    """Wiring contract: the check runs from DOCTOR_CHECKS, and openrouter — deliberately absent
    from PROVIDER_REGISTRY (#109397) — is still scanned alongside every api_key registry pool."""
    from hermes_cli import doctor_pools
    from hermes_cli.auth import PROVIDER_REGISTRY
    import hermes_cli.doctor as doctor

    assert any(check is doctor_pools._check_credential_pools for _title, check in doctor.DOCTOR_CHECKS)
    ids = set(doctor_pools._pool_provider_ids())
    assert "openrouter" in ids
    registry_api_keys = {
        pid for pid, pconfig in PROVIDER_REGISTRY.items()
        if getattr(pconfig, "auth_type", "") == "api_key"
    }
    assert registry_api_keys <= ids, registry_api_keys - ids
