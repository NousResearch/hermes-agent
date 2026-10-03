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
    # A 402 is billing: `hermes auth reset` clears the local stamp and the next call
    # re-402s, so advising it here would be a non-remedy.
    assert "add credits" in finding.manual_issues[0], finding.manual_issues[0]
    assert "hermes auth add openrouter" in finding.manual_issues[0], finding.manual_issues[0]


def test_dead_pool_without_recovery_time_is_an_issue(capsys):
    _write_openrouter_pool([{**_BASE, "last_status": "dead", "last_status_at": time.time()}])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "all 1 entries unavailable, no recovery time" in out
    assert len(finding.manual_issues) == 1, finding.manual_issues
    # DEAD never re-enters via TTL — only a write-side re-auth clears it.
    assert "DEAD never recovers on its own" in finding.manual_issues[0], finding.manual_issues[0]
    assert "hermes auth add openrouter" in finding.manual_issues[0], finding.manual_issues[0]


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
    """A pool store that raises OSError must produce a warn row + issue instead of killing the
    scan for the remaining providers (one broken provider must not hide the others)."""
    import agent.credential_pool as cp

    def _boom(_pid):
        # load_pool propagates OSError from _load_auth_store; unparseable JSON is absorbed
        # there and never reaches the caller.
        raise OSError("auth.json unreadable")

    monkeypatch.setattr(cp, "load_pool", _boom)
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "unreadable: auth.json unreadable" in out
    assert len(finding.manual_issues) >= 1, finding.manual_issues
    assert "auth.json unreadable" in finding.manual_issues[0]


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



def test_partially_burned_pool_reports_the_burn_not_the_env_reference(capsys):
    """The #119533 false-burn guard, at the shape that actually regressed.

    Two entries: one exhausted (timed) and one env-sourced reference whose secret does not
    resolve in this process (``access_token: ""``, no burn stamp). The unhydrated reference
    must NOT be counted as burned, and the pool must NOT be reported as fully benched — but
    the one real burn must still surface. The old guard (``burned < total`` -> silence) hid
    it entirely; dropping the guard entirely accused the env reference of being burned.
    The correct answer is in between, and this pins it."""
    _write_openrouter_pool([
        {**_BASE, "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": 429},
        {
            **_BASE, "id": "key-2", "label": "key-2", "access_token": "",
            "source": "env:OPENROUTER_API_KEY", "extra": {"secret_fingerprint": "abc123"},
        },
    ])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "Credential pool: openrouter" in out, "the real burn must not be hidden:\n%s" % out
    assert "1 of 2 entries" in out, out
    assert "1 unavailable with no burn state" in out, out
    assert "all 2 entries" not in out, "the env reference must not be counted as burned:\n%s" % out
    assert len(finding.manual_issues) == 1, finding.manual_issues
    assert "1 of 2 entries" in finding.manual_issues[0], finding.manual_issues[0]


def test_fully_unhydrated_pool_stays_silent(capsys):
    """No burn state at all: an env reference that cannot resolve here is a key-presence
    question, owned by the env/connectivity checks. Reporting it would be a false burn."""
    _write_openrouter_pool([{
        **_BASE, "access_token": "",
        "source": "env:OPENROUTER_API_KEY", "extra": {"secret_fingerprint": "abc123"},
    }])
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "Credential pool: openrouter" not in out, out
    assert finding.manual_issues == [], finding.manual_issues


def test_one_unreadable_pool_does_not_hide_a_burned_one(capsys, monkeypatch):
    """Isolation: the scan must continue past a broken provider. ``_boom`` here raises for
    openrouter ONLY, so a second provider with a burned pool must still get its row — a
    ``continue`` -> ``break`` mutation loses it silently."""
    import agent.credential_pool as cp

    real = cp.load_pool
    burned_entry = {
        **_BASE, "id": "key-gmi", "base_url": "https://generativelanguage.googleapis.com",
        "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": 402,
    }

    def _selective(pid):
        if pid == "openrouter":
            raise OSError("auth.json unreadable")
        if pid == "gemini":
            return real(pid)
        return real(pid)

    _write_openrouter_pool([])
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {
            "openrouter": [{**_BASE, "access_token": ""}],
            "gemini": [burned_entry],
        }}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cp, "load_pool", _selective)
    from hermes_cli.doctor_pools import _check_credential_pools

    _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "unreadable" in out, "the broken provider should still warn:\n%s" % out
    assert "Credential pool: gemini" in out, (
        "a healthy provider after the broken one was never scanned (continue -> break?):\n%s" % out
    )


def test_no_section_header_when_every_pool_is_healthy_or_absent(capsys):
    """The "Credential Pools" header renders only when rows exist. An unconditional header
    prints a dangling section on every clean install."""
    _write_openrouter_pool([{**_BASE}])  # configured, available -> an OK row, not empty
    from hermes_cli.doctor_pools import _check_credential_pools

    _check_credential_pools(False)
    healthy = capsys.readouterr().out
    _write_openrouter_pool([{**_BASE, "access_token": ""}])  # nothing to report
    _check_credential_pools(False)
    empty = capsys.readouterr().out
    assert "Credential Pools" in healthy, "a row must render its section header:\n%s" % healthy
    assert "Credential Pools" not in empty, (
        "no rows must mean no section header:\n%s" % empty
    )


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


def test_programming_bug_is_not_disguised_as_unreadable(capsys, monkeypatch):
    """``except OSError`` must stay narrow: a TypeError from a real bug must surface (via the
    check's own error handling), not be reported to the user as a corrupt credential store."""
    import agent.credential_pool as cp

    def _bug(_pid):
        raise TypeError("pool.entries() changed shape")

    monkeypatch.setattr(cp, "load_pool", _bug)
    from hermes_cli.doctor_pools import _check_credential_pools

    finding = _check_credential_pools(False)
    out = capsys.readouterr().out
    assert "unreadable" not in out, "a programming bug must not be reported as unreadable:\n%s" % out
    assert not finding.manual_issues or all("unreadable" not in i for i in finding.manual_issues), \
        finding.manual_issues


def test_burned_pool_is_warned_never_green(capsys):
    """Severity is part of the contract: a fully benched pool must render as a warning.
    ``check_ok`` here would show a dead provider as healthy green."""
    _write_openrouter_pool([{
        **_BASE, "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": 429,
    }])
    from hermes_cli.doctor_pools import _check_credential_pools

    _check_credential_pools(False)
    row = [l for l in capsys.readouterr().out.splitlines() if "Credential pool: openrouter" in l]
    assert row, "no row rendered"
    assert "⚠" in row[0] and "✓" not in row[0], "a burned pool must warn, not pass:\n%s" % row[0]
